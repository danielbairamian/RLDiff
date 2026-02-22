import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torch.distributions import Beta
import numpy as np

from src.MonteCarloLayer import MonteCarloLayer
from src.latent_encoder.VisionEncoder import VisionEncoder


# Beta distribution naturally lives in [0,1] — no clamping or correction terms needed.
# α, β > 1  → unimodal (enforced by softplus + 1 below)
# mean       = α / (α + β)        ← where the agent wants to step
# confidence = α + β              ← how certain it is  (higher = tighter)
# entropy    = analytic, clean signal for the entropy bonus


class NeRFEmbedder(nn.Module):
    """
    NeRF-style sinusoidal positional encoding.
    For each input scalar, produces [sin(2^0 * pi * x), cos(2^0 * pi * x), ..., sin(2^(L-1) * pi * x), cos(2^(L-1) * pi * x)]
    Output dim per scalar: 2 * L
    """
    def __init__(self, L: int):
        super().__init__()
        self.L = L
        freqs = 2.0 ** torch.arange(L, dtype=torch.float32)  # (L,)
        self.register_buffer('freqs', freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_freq = x.unsqueeze(-1) * self.freqs * torch.pi  # (..., D, L)
        sins = torch.sin(x_freq)
        coss = torch.cos(x_freq)
        embedded = torch.stack([sins, coss], dim=-1)       # (..., D, L, 2)
        return embedded.flatten(start_dim=-3)              # (..., D * L * 2)

    @property
    def out_dim_per_scalar(self) -> int:
        return 2 * self.L


class Backbone_Encoder(nn.Module):
    def __init__(self, state_dim: int, fused_dims: int, time_encoder_dims: List[int], projection_dims: List[int]):
        super(Backbone_Encoder, self).__init__()

        self.nerf_embedder = NeRFEmbedder(L=16)
        nerf_out_dim = self.nerf_embedder.out_dim_per_scalar * 2  # alpha and steps

        self.time_encoder = nn.ModuleList()
        input_dim = 2
        for i in range(len(time_encoder_dims)):
            output_dim = time_encoder_dims[i]
            self.time_encoder.append(
                nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.SiLU(),
                )
            )
            input_dim = output_dim

        self.fused_latent = nn.Bilinear(state_dim, 2, fused_dims)

        self.projection_encoder = nn.Sequential()
        input_dim = fused_dims + time_encoder_dims[-1] + state_dim + 2 # + nerf_out_dim
        for i in range(len(projection_dims)):
            output_dim = projection_dims[i]
            self.projection_encoder.append(
                nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.SiLU(),
                )
            )
            input_dim = output_dim

        self.backbone_out_dim = projection_dims[-1]

    def forward(self, state, alpha, steps):
        time_inputs = torch.cat([alpha.unsqueeze(-1), steps.unsqueeze(-1)], dim=-1)

        time_encoding = time_inputs
        for layer in self.time_encoder:
            time_encoding = layer(time_encoding)

        fused = self.fused_latent(state, time_inputs)
        # nerf_embedded = self.nerf_embedder(time_inputs)
        # combined = torch.cat([fused, time_encoding, state, time_inputs, nerf_embedded], dim=-1)
        combined = torch.cat([fused, time_encoding, state, time_inputs], dim=-1)

        for layer in self.projection_encoder:
            combined = layer(combined)

        return combined


# ---------------------------------------------------------------------------
# Beta Policy Head
# ---------------------------------------------------------------------------
# The network outputs raw_alpha and raw_beta (unconstrained).
# We map them to concentration params via:
#   α = softplus(raw_alpha) + 1     → α > 1  always
#   β = softplus(raw_beta)  + 1     → β > 1  always
# This forces a *unimodal* distribution (no U-shape pathology).
#
# Intuition guide:
#   α ≈ β ≈ 1  →  near-uniform  (high entropy, lots of exploration)
#   α ≈ β >> 1 →  tight bell at 0.5
#   α >> β     →  mass near 1.0  (big step)
#   β >> α     →  mass near 0.0  (small step)
#
# To push the initial mean toward a target `m` with concentration `c`:
#   raw_alpha_bias  ≈  softplus_inv(m * c - 1)
#   raw_beta_bias   ≈  softplus_inv((1-m) * c - 1)
# ---------------------------------------------------------------------------

def _softplus_inv(x: float) -> float:
    """Inverse of softplus: log(exp(x) - 1)"""
    return np.log(np.exp(x) - 1.0)


class PPOAgent(nn.Module):
    def __init__(
        self,
        vision_encoder: VisionEncoder,
        state_dim: int,
        fused_dims: int,
        time_encoder_dims: List[int],
        projection_dims: List[int],
        action_dim: int,
        # act_min / act_max are kept for env.step() reference, but the policy
        # natively outputs in [0, 1].  Scale inside the env if needed.
        act_min: float = 0.0,
        act_max: float = 1.0,
        # Target initial mean of the action distribution
        mean_action_init: float = 0.1,
        # Initial concentration (α+β).  Higher → tighter initial exploration.
        # 2.0 means α=β≈1 → near-uniform.  Try 4–8 for moderate exploration.
        concentration_init: float = 4.0,
    ):
        super(PPOAgent, self).__init__()

        self.register_buffer('act_min', torch.tensor(act_min, dtype=torch.float32))
        self.register_buffer('act_max', torch.tensor(act_max, dtype=torch.float32))

        self.vision_encoder = vision_encoder
        self.backbone = Backbone_Encoder(state_dim, fused_dims, time_encoder_dims, projection_dims)
        backbone_dim = self.backbone.backbone_out_dim

        # Two heads: raw concentration parameters for the Beta distribution.
        self.alpha_head = nn.Linear(backbone_dim, action_dim)   # → α (after softplus+1)
        self.beta_head  = nn.Linear(backbone_dim, action_dim)   # → β (after softplus+1)

        # Initialise biases so the distribution starts near mean_action_init
        # with the requested concentration.
        #   desired α = mean_action_init * concentration_init
        #   desired β = (1 - mean_action_init) * concentration_init
        # Both must be > 1 for unimodality, so clip the lower bound.
        desired_alpha = max(1.01, mean_action_init * concentration_init)
        desired_beta  = max(1.01, (1.0 - mean_action_init) * concentration_init)
        raw_alpha_bias = _softplus_inv(desired_alpha - 1.0)   # softplus(bias) + 1 = desired_alpha
        raw_beta_bias  = _softplus_inv(desired_beta  - 1.0)

        with torch.no_grad():
            self.alpha_head.bias.normal_(raw_alpha_bias, 0.01)
            self.alpha_head.weight.normal_(0, 0.01)
            self.beta_head.bias.normal_(raw_beta_bias, 0.01)
            self.beta_head.weight.normal_(0, 0.01)

        self.critic = nn.Linear(backbone_dim, 1)
        self.mc_layer = MonteCarloLayer(
            self.critic,
            dropout_p=0.05, mc_samples=256,
            attention_mode='attention', attend_mode='inputs',
            num_heads=8, embedding_size=backbone_dim // 2,
            query_mode='per_sample',
        )

    # ------------------------------------------------------------------
    # Helper: raw network outputs → Beta concentration params
    # ------------------------------------------------------------------
    def _concentration_params(self, combined: torch.Tensor):
        """
        Returns (alpha, beta) — both > 1, shape (B, action_dim).
        softplus ensures positivity; +1 pushes above 1 for unimodality.
        """
        alpha = F.softplus(self.alpha_head(combined)) + 1.0
        beta  = F.softplus(self.beta_head(combined))  + 1.0
        return alpha, beta

    # ------------------------------------------------------------------
    # forward  (used during rollout collection)
    # ------------------------------------------------------------------
    def forward(self, state, alpha_t, steps, deterministic=False):
        """
        alpha_t : current diffusion alpha  (B,)
        steps   : current step count       (B,)

        Returns
        -------
        action      : (B, A)  — sampled or mean, in [0, 1]
        value       : (B,)
        log_prob    : (B,)
        conc_alpha  : (B, A)  — α concentration (interpretable)
        conc_beta   : (B, A)  — β concentration (interpretable)
        """
        state_enc = self.vision_encoder.encode(state)
        combined  = self.backbone(state_enc, alpha_t, steps)

        conc_alpha, conc_beta = self._concentration_params(combined)
        dist  = Beta(conc_alpha, conc_beta)
        value = self.mc_layer.get_mean_only(combined)

        if deterministic:
            # Mode of Beta: (α-1)/(α+β-2)  — valid because α,β > 1
            # action = (conc_alpha - 1.0) / (conc_alpha + conc_beta - 2.0)
            # Mean of Beta: α / (α + β)
            action = conc_alpha / (conc_alpha + conc_beta)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, value.squeeze(-1), log_prob, conc_alpha, conc_beta

    # ------------------------------------------------------------------
    # evaluate_actions  (used during PPO update)
    # ------------------------------------------------------------------
    def evaluate_actions(self, state, alpha_t, steps, actions):
        """
        actions : (B, A)  — the raw actions stored in the rollout buffer,
                            already in [0, 1] (no squashing correction needed)
        """
        state_enc = self.vision_encoder.encode(state)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        # Clamp actions away from hard boundaries to avoid log(0) in Beta.log_prob
        actions = actions.clamp(1e-6, 1.0 - 1e-6)

        combined = self.backbone(state_enc, alpha_t, steps)

        conc_alpha, conc_beta = self._concentration_params(combined)
        dist  = Beta(conc_alpha, conc_beta)
        value = self.mc_layer.get_mean_only(combined)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)

        return log_prob, value.squeeze(-1), entropy, conc_alpha, conc_beta