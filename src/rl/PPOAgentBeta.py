import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torch.distributions import Beta
import numpy as np

from src.MonteCarloLayer import MonteCarloLayer
from src.latent_encoder.VisionEncoder import VisionEncoder


# ---------------------------------------------------------------------------
# Beta Policy Head — Mean/Concentration Parameterization
# ---------------------------------------------------------------------------
# Instead of outputting α and β directly, the network outputs:
#
#   m = sigmoid(raw_mean)                    ∈ (0, 1)   — where to step
#   c = softplus(raw_concentration) + 2     ∈ (2, ∞)   — how confident
#
# Then α = m * c  and  β = (1-m) * c
#
# Why this is better than raw α/β:
#   - Mean head is bounded by sigmoid — cannot explode by construction
#   - Concentration head uses softplus which grows linearly for large inputs:
#     to get c=1e5, the raw output must itself be ~1e5. With exp, only log(1e5)≈11.5
#     is needed — a small number easily reached in a few hundred updates.
#     Softplus makes explosion practically unreachable under normal gradient clipping.
#   - Mean and concentration are decoupled — a large concentration update
#     doesn't corrupt the mean, and vice versa. Previously α and β could
#     drift independently, producing nonsensical mean/concentration combinations.
#
# Deterministic action = mode = (α-1)/(α+β-2)
#   - Always valid since α = m*c > 1 and β = (1-m)*c > 1 when c > 2, m ∈ (0,1)
#   - The principled MAP estimate — the single most probable action
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

        # Two decoupled heads with activations matched to their natural ranges
        self.mean_head          = nn.Linear(backbone_dim, action_dim)  # sigmoid  → (0, 1)
        self.concentration_head = nn.Linear(backbone_dim, action_dim)  # softplus → (0, ∞), +2 ensures α,β > 1

        # Initialization:
        #   sigmoid(mean_bias) = mean_action_init
        #     → mean_bias = log(m / (1-m))   (logit)
        #   softplus(conc_bias) + 2 = concentration_init
        #     → conc_bias = log(exp(concentration_init - 2) - 1)   (softplus inverse)
        mean_bias  = np.log(mean_action_init / (1.0 - mean_action_init))
        conc_bias  = np.log(np.exp(max(concentration_init - 2.0, 1e-6)) - 1.0)

        with torch.no_grad():
            self.mean_head.bias.normal_(mean_bias, 0.01)
            self.mean_head.weight.normal_(0, 0.01)
            self.concentration_head.bias.normal_(conc_bias, 0.01)
            self.concentration_head.weight.normal_(0, 0.01)

        self.critic = nn.Linear(backbone_dim, 1)
        self.mc_layer = MonteCarloLayer(
            self.critic,
            dropout_p=0.05, mc_samples=256,
            attention_mode='attention', attend_mode='inputs',
            num_heads=8, embedding_size=backbone_dim // 2,
            query_mode='per_sample',
        )

    # ------------------------------------------------------------------
    # Helper: network outputs → Beta α and β
    # ------------------------------------------------------------------
    def _concentration_params(self, combined: torch.Tensor):
        """
        mean          = sigmoid(raw_mean)            ∈ (0, 1)   — bounded, cannot explode
        concentration = softplus(raw_conc) + 2       ∈ (2, ∞)   — linear growth, slow to explode
        alpha         = mean * concentration          > 1  always (since c > 2 and m > 0)
        beta          = (1 - mean) * concentration    > 1  always (since c > 2 and m < 1)
        """
        mean          = torch.sigmoid(self.mean_head(combined))
        concentration = F.softplus(self.concentration_head(combined)) + 2.0
        alpha = mean * concentration
        beta  = (1.0 - mean) * concentration
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
            # Mode = (α-1)/(α+β-2) — always valid since α,β > 1 by construction
            # action = (conc_alpha - 1.0) / (conc_alpha + conc_beta - 2.0)
            action = conc_alpha  / (conc_alpha + conc_beta + 1e-8)  # Mean action for logging — more interpretable than mode
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