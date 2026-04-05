import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torch.distributions import Beta
import numpy as np

from src.MonteCarloLayer import MonteCarloLayer
from src.latent_encoder.VisionEncoder import VisionEncoder


# ---------------------------------------------------------------------------
# Beta Policy Head — Mean / Log-Concentration Parameterization
# ---------------------------------------------------------------------------
# A single head outputs 2 * action_dim values, chunked into (raw_mean, raw_conc).
#
#   μ     = sigmoid(raw_mean)                    — Beta mode, ∈ (0, 1)
#   κ     = exp(raw_conc) + KAPPA_MIN            — total concentration α+β
#   α     = 1 + μ · (κ − 2)
#   β     = 1 + (1 − μ) · (κ − 2)
#
# Key identities:
#   mode  = (α−1)/(α+β−2) = μ                   — mode equals sigmoid output
#   mean  = α / (α+β) = (1 + μ(κ−2)) / κ        — ≈ μ for large κ
#   α + β = κ                                    — concentration is explicit
#   α, β  > 1 always                             — unimodal Beta guaranteed
#
# Why exp for concentration (not softplus):
#   - Exp gives true log-space parameterization: raw_conc = log(κ − κ_min).
#     κ = 10 → 100 takes the same optimizer step as κ = 100 → 1000.
#   - For budget=100, actions ~ 0.01: need κ ~ 10000 for tight distributions.
#     softplus needs raw_conc ~ 10000 to reach that; exp needs raw_conc ~ 9.2.
#   - RAW_CONC_MAX=10 → κ_max ≈ 22000, covers budget range 5–100 comfortably.
#
# STE clamp on raw_conc:
#   - Forward: clamps raw_conc to [RAW_CONC_MIN, RAW_CONC_MAX] for numerical safety.
#   - Backward: straight-through gradient (d/d(raw_conc) = 1 always).
#   - This prevents exp explosion without killing gradients when the network
#     saturates against the ceiling.
#
# KAPPA_MIN = 2.005: floor on total concentration.
#   Guarantees κ − 2 ≥ 0.005 > 0, so α, β > 1 strictly.
# ---------------------------------------------------------------------------

KAPPA_MIN = 2.005
RAW_MEAN_MIN = -8.0
RAW_MEAN_MAX = 8.0

RAW_CONC_MIN = -8.0
RAW_CONC_MAX = 10.0   # exp(10) ≈ 22026 → κ_max ≈ 22028, sufficient for budget=100


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

        self.nerf_embedder = NeRFEmbedder(L=10)  # 10 frequencies per scalar → 20 dims per scalar

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

        self.fused_latent = nn.Linear(state_dim + 2, fused_dims)
        self.projection_encoder = nn.Sequential()
        input_dim = fused_dims + time_encoder_dims[-1] + state_dim + 2 + self.nerf_embedder.out_dim_per_scalar * 2

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

        fused = self.fused_latent(torch.cat([state, time_inputs], dim=-1))
        fused = F.silu(fused)

        nerf_embeddings = self.nerf_embedder(time_inputs)
        combined = torch.cat([fused, time_encoding, state, time_inputs, nerf_embeddings], dim=-1)

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
        concentration_init: float = 4.0,
    ):
        super(PPOAgent, self).__init__()

        self.action_dim = action_dim

        self.register_buffer('act_min', torch.tensor(act_min, dtype=torch.float32))
        self.register_buffer('act_max', torch.tensor(act_max, dtype=torch.float32))

        self.vision_encoder = vision_encoder
        self.backbone = Backbone_Encoder(state_dim, fused_dims, time_encoder_dims, projection_dims)
        self.backbone_v = Backbone_Encoder(state_dim, fused_dims, time_encoder_dims, projection_dims)
        backbone_dim = self.backbone.backbone_out_dim

        self.mean_head = nn.Linear(backbone_dim, action_dim)
        self.conc_head = nn.Linear(backbone_dim, action_dim)

        mean_action_init_raw = np.log(mean_action_init / (1 - mean_action_init))  # inverse sigmoid
        conc_init_raw = np.log(concentration_init - KAPPA_MIN)                    # inverse of exp + KAPPA_MIN

        with torch.no_grad():
            self.mean_head.bias.normal_(mean_action_init_raw, 0.01)
            self.mean_head.weight.normal_(0, 0.01)

            self.conc_head.bias.normal_(conc_init_raw, 0.01)
            self.conc_head.weight.normal_(0, 0.01)

        self.critic = nn.Linear(backbone_dim, 1)
        self.mc_layer = MonteCarloLayer(
            self.critic,
            dropout_p=0.1, mc_samples=512,
            attention_mode='attention', attend_mode='inputs',
            num_heads=4, embedding_size=backbone_dim // 2,
            query_mode='per_sample',
        )

    # ------------------------------------------------------------------
    # Helper: raw outputs → α, β
    # ------------------------------------------------------------------

    def _alpha_beta_params(self, combined: torch.Tensor):
        raw_mean = self.mean_head(combined)
        raw_conc = self.conc_head(combined)

        mu = torch.sigmoid(raw_mean)

        # STE clamp: forward is numerically safe, backward is straight-through (grad = 1)
        # This allows the network to saturate against the ceiling without killing gradients,
        # which matters for budget=100 where κ needs to be large.
        raw_conc_clamped = raw_conc + (raw_conc.clamp(RAW_CONC_MIN, RAW_CONC_MAX) - raw_conc).detach()
        kappa = torch.exp(raw_conc_clamped) + KAPPA_MIN

        alpha = 1.0 + mu * (kappa - 2.0)
        beta  = 1.0 + (1.0 - mu) * (kappa - 2.0)

        return alpha, beta, {'kappa': kappa, 'mu': mu}

    # ------------------------------------------------------------------
    # forward  (used during rollout collection)
    # ------------------------------------------------------------------
    def forward(self, state, alpha_t, steps, deterministic=False):
        state_enc = self.vision_encoder.encode(state)
        combined  = self.backbone(state_enc, alpha_t, steps)
        combined_v = self.backbone_v(state_enc, alpha_t, steps)

        conc_alpha, conc_beta, net_dict = self._alpha_beta_params(combined)
        dist  = Beta(conc_alpha, conc_beta)
        value = self.mc_layer.get_mean_only(combined_v)

        if deterministic:
            action = conc_alpha / (conc_alpha + conc_beta)  # true mean — well-behaved at all κ
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
        combined_v = self.backbone_v(state_enc, alpha_t, steps)

        conc_alpha, conc_beta, net_dict = self._alpha_beta_params(combined)
        dist  = Beta(conc_alpha, conc_beta)

        value = self.mc_layer.get_mean_only(combined_v)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)

        return log_prob, value.squeeze(-1), entropy, conc_alpha, conc_beta, net_dict