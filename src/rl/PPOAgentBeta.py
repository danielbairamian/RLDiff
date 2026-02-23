import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torch.distributions import Beta
import numpy as np

from src.MonteCarloLayer import MonteCarloLayer
from src.latent_encoder.VisionEncoder import VisionEncoder


# ---------------------------------------------------------------------------
# Beta Policy Head — Log-Space Parameterization
# ---------------------------------------------------------------------------
# The network outputs raw_alpha, raw_beta (unconstrained).
# We map them to concentration params via:
#
#   α = exp(raw_alpha) + 1     → α > 1  always
#   β = exp(raw_beta)  + 1     → β > 1  always
#
# Why log-space over softplus:
#   - Concentration spans many orders of magnitude (2 → 1000+) during training
#   - In log-space, AdamW updates are scale-invariant: a step of δ in raw
#     space always means a ~δ*100% multiplicative change in α, whether
#     α is 5 or 5000. softplus is approximately linear at large values,
#     giving fixed *absolute* steps regardless of current scale.
#
# Why no saturation:
#   - Hard clamps kill gradients outside the range
#   - Smooth saturation (tanh-based) introduces an arbitrary scale constant
#   - The training loop already has clip_grad_norm_(0.5) which is the
#     correct place to handle runaway updates — no need to also saturate
#     inside the network
#
# Deterministic action = mode = (α-1)/(α+β-2)
#   - Always valid since α,β > 1 is guaranteed by construction
#   - The principled MAP estimate — the single most probable action
#   - Mean was only ever a workaround for early training instability;
#     if initialization is correct, mode is always the right choice
# ---------------------------------------------------------------------------


class NeRFEmbedder(nn.Module):
    """
    NeRF-style sinusoidal positional encoding.
    For each input scalar, produces [sin(2^0 * pi * x), cos(2^0 * pi * x), ..., sin(2^(L-1) * pi * x), cos(2^(L-1) * pi * x)]
    Output dim per scalar: 2 * L
    """
    def __init__(self, L: int):
        super().__init__()
        self.L = L
        freqs = 2.0 ** torch.arange(L, dtype=torch.float32)
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
        input_dim = fused_dims + time_encoder_dims[-1] + state_dim + 2
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
        combined = torch.cat([fused, time_encoding, state, time_inputs], dim=-1)

        for layer in self.projection_encoder:
            combined = layer(combined)

        return combined


class PPOAgent(nn.Module):
    def __init__(
        self,
        vision_encoder: VisionEncoder,
        state_dim: int,
        fused_dims: int,
        time_encoder_dims: List[int],
        projection_dims: List[int],
        action_dim: int,
        act_min: float = 0.0,
        act_max: float = 1.0,
        mean_action_init: float = 0.1,
        concentration_init: float = 4.0,
    ):
        super(PPOAgent, self).__init__()

        self.register_buffer('act_min', torch.tensor(act_min, dtype=torch.float32))
        self.register_buffer('act_max', torch.tensor(act_max, dtype=torch.float32))

        self.vision_encoder = vision_encoder
        self.backbone = Backbone_Encoder(state_dim, fused_dims, time_encoder_dims, projection_dims)
        backbone_dim = self.backbone.backbone_out_dim

        self.alpha_head = nn.Linear(backbone_dim, action_dim)
        self.beta_head  = nn.Linear(backbone_dim, action_dim)

        # Log-space initialization:
        # We want exp(raw_bias) + 1 = desired_param  →  raw_bias = log(desired - 1)
        desired_alpha = max(1.01, mean_action_init * concentration_init)
        desired_beta  = max(1.01, (1.0 - mean_action_init) * concentration_init)
        raw_alpha_bias = np.log(desired_alpha - 1.0)
        raw_beta_bias  = np.log(desired_beta  - 1.0)

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
    # Helper: raw network outputs → Beta concentration params (log-space)
    # ------------------------------------------------------------------
    def _concentration_params(self, combined: torch.Tensor):
        """
        Returns (alpha, beta) — both > 1, shape (B, action_dim).
        exp maps unconstrained raw outputs to (0, ∞), +1 ensures > 1.
        Gradient clipping in the training loop (clip_grad_norm_) is the
        sole protection against explosion — no saturation inside the network.
        """
        alpha = torch.exp(self.alpha_head(combined)) + 1.0 + 1e-6  # Add small epsilon to ensure strictly > 1, preventing numerical issues in Beta distribution
        beta  = torch.exp(self.beta_head(combined))  + 1.0 + 1e-6  # Same epsilon for beta to ensure numerical stability, even if it starts near 1.0
        return alpha, beta

    # ------------------------------------------------------------------
    # forward  (used during rollout collection)
    # ------------------------------------------------------------------
    def forward(self, state, alpha_t, steps, deterministic=False):
        state_enc = self.vision_encoder.encode(state)
        combined  = self.backbone(state_enc, alpha_t, steps)

        conc_alpha, conc_beta = self._concentration_params(combined)
        dist  = Beta(conc_alpha, conc_beta)
        value = self.mc_layer.get_mean_only(combined)

        if deterministic:
            # Mode = (α-1)/(α+β-2) — always valid since α,β > 1 by construction.
            # This is the MAP estimate: the single most probable action.
            action_mode = (conc_alpha - 1.0) / (conc_alpha + conc_beta - 2.0)
            action_mean = conc_alpha / (conc_alpha + conc_beta)
            # action = action_mode
            action = action_mean 
        else:
            action = dist.sample()

        action_for_logprob = action.clamp(1e-6, 1.0 - 1e-6)
        log_prob = dist.log_prob(action_for_logprob).sum(dim=-1)

        return action, value.squeeze(-1), log_prob, conc_alpha, conc_beta

    # ------------------------------------------------------------------
    # evaluate_actions  (used during PPO update)
    # ------------------------------------------------------------------
    def evaluate_actions(self, state, alpha_t, steps, actions):
        state_enc = self.vision_encoder.encode(state)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        # Clamp stored actions away from hard boundaries — this is data,
        # not network outputs, so killing gradients here is harmless.
        actions = actions.clamp(1e-6, 1.0 - 1e-6)

        combined = self.backbone(state_enc, alpha_t, steps)

        conc_alpha, conc_beta = self._concentration_params(combined)
        dist  = Beta(conc_alpha, conc_beta)
        value = self.mc_layer.get_mean_only(combined)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)

        return log_prob, value.squeeze(-1), entropy, conc_alpha, conc_beta