import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torch.distributions import Beta
import numpy as np

from src.MonteCarloLayer import MonteCarloLayer
from src.latent_encoder.VisionEncoder import VisionEncoder


# ---------------------------------------------------------------------------
# Beta Policy Head — Joint α/β Estimation
# ---------------------------------------------------------------------------
# A single head outputs 2 * action_dim values, chunked into raw_alpha, raw_beta.
#
#   α = softplus(clamp(raw_alpha, min=RAW_MIN)) + 1.0
#   β = softplus(clamp(raw_beta,  min=RAW_MIN)) + 1.0
#
# Why joint over two separate heads:
#   - α and β are not independent given the state: a larger desired step (high α)
#     should come with lower relative confidence (higher β) since big steps are riskier
#   - A single weight matrix encodes the α/β relationship directly rather than
#     forcing two independent heads to discover the correlation from scratch
#   - Same parameter count as two separate heads, no extra cost
#
# Why clamp(min=RAW_MIN) on the input to softplus:
#   - softplus(x) → 0 as x → -∞, gradient → 0 → α → 1 exactly → stuck forever
#   - RAW_MIN = -9.21 floors input: softplus(-9.21) ≈ 0.0001 → α,β ≥ 1.0001
#   - gradient at floor: sigmoid(-9.21) ≈ 0.0001 — tiny but nonzero, never exactly zero
#   - no upper clamp: softplus grows linearly, gradient clipping handles explosion
#
# RAW_MIN = log(10000) = 9.21: chosen so softplus(RAW_MIN) ≈ 0.0001
# Unimodality: α,β > 1 always ✓
# Mode = (α-1)/(α+β-2) — always valid, used for deterministic action
# ---------------------------------------------------------------------------

KAPPA_MIN = 2.005   # Minimum κ = α+β; ensures α, β > 1 (unimodal Beta)

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
        mean_action_init: float = 0.5,
        concentration_init: float = 2.1,
    ):
        super(PPOAgent, self).__init__()

        self.action_dim = action_dim

        self.register_buffer('act_min', torch.tensor(act_min, dtype=torch.float32))
        self.register_buffer('act_max', torch.tensor(act_max, dtype=torch.float32))

        self.vision_encoder = vision_encoder
        self.backbone = Backbone_Encoder(state_dim, fused_dims, time_encoder_dims, projection_dims)
        backbone_dim = self.backbone.backbone_out_dim

        self.mean_concentration_head = nn.Linear(backbone_dim, 2 * action_dim)

        self.critic = nn.Linear(backbone_dim, 1)
        self.mc_layer = MonteCarloLayer(
            self.critic,
            dropout_p=0.05, mc_samples=256,
            attention_mode='attention', attend_mode='inputs',
            num_heads=8, embedding_size=backbone_dim // 2,
            query_mode='per_sample',
        )

    # ------------------------------------------------------------------
    # Helper: raw outputs → α, β
    # ------------------------------------------------------------------
    def _concentration_params(self, combined: torch.Tensor):
        raw_mean, raw_conc = self.mean_concentration_head(combined).chunk(2, dim=-1) 
        mu = torch.sigmoid(raw_mean)  # (0,1)
        kappa = F.softplus(raw_conc) + KAPPA_MIN  # > KAPPA_MIN to ensure unimodality
        excess = kappa - 2.0  # κ = α + β, so excess over 2 is split between α and β    
        alpha = 1.0 + mu * excess
        beta  = 1.0 + (1.0 - mu) * excess
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
            # Mode = (α-1)/(α+β-2) — always valid since α,β > 1
            # action = (conc_alpha - 1.0) / (conc_alpha + conc_beta - 2.0)
            action = conc_alpha / (conc_alpha + conc_beta)  # mean action, not mode — more stable for early training
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

        actions = actions.clamp(1e-6, 1.0 - 1e-6)

        combined = self.backbone(state_enc, alpha_t, steps)

        conc_alpha, conc_beta = self._concentration_params(combined)
        dist  = Beta(conc_alpha, conc_beta)
        value = self.mc_layer.get_mean_only(combined)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)

        return log_prob, value.squeeze(-1), entropy, conc_alpha, conc_beta