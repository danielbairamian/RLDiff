import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional
from torch.distributions import Beta
import numpy as np

from src.MonteCarloLayer import MonteCarloLayer
from src.latent_encoder.VisionEncoder_Res import VisionEncoder


# ---------------------------------------------------------------------------
# Beta Policy Head — Mean / Log-Concentration Parameterization
# ---------------------------------------------------------------------------
# A single head outputs 2 * action_dim values, chunked into (raw_mean, raw_conc).
#
#   μ     = sigmoid(raw_mean)                    — Beta mode, ∈ (0, 1)
#   κ     = exp(raw_conc) + KAPPA_MIN            — total concentration α+β  [soft=False]
#   κ     = softplus(raw_conc) + KAPPA_MIN       — total concentration α+β  [soft=True]
#   α     = 1 + μ · (κ − 2)
#   β     = 1 + (1 − μ) · (κ − 2)
#
# Key identities:
#   mode  = (α−1)/(α+β−2) = μ                   — mode equals sigmoid output
#   mean  = α / (α+β) = (1 + μ(κ−2)) / κ        — ≈ μ for large κ
#   α + β = κ                                    — concentration is explicit
#   α, β  > 1 always                             — unimodal Beta guaranteed
#
# soft=False (exp):
#   - True log-space parameterization: equal optimizer steps → multiplicative κ changes.
#   - Required for budget=100 where κ ~ 10000 is needed for tight distributions.
#     exp needs raw_conc ~ 9.2; softplus would need raw_conc ~ 10000.
#   - STE clamp prevents exp explosion while keeping straight-through gradients.
#   - conc_init: raw_conc = log(κ_init - KAPPA_MIN)
#
# soft=True (softplus):
#   - Smoother, more stable gradients. Practically equivalent to exp for κ < ~100.
#   - Suitable for budget=5–20 where κ ~ 50–200 suffices.
#   - Cannot realistically reach κ > ~1000 from normal network outputs.
#   - conc_init: raw_conc = log(exp(κ_init - KAPPA_MIN) - 1)  [softplus inverse]
#
# KAPPA_MIN = 2.005: floor on total concentration.
#   Guarantees κ − 2 ≥ 0.005 > 0, so α, β > 1 strictly.
# ---------------------------------------------------------------------------

KAPPA_MIN = 2.005
RAW_MEAN_MIN = -8.0
RAW_MEAN_MAX = 8.0

RAW_CONC_MIN = -8.0
RAW_CONC_MAX = 10.0   # exp(10) ≈ 22026 → κ_max ≈ 22028, sufficient for budget=100


# ---------------------------------------------------------------------------
# NeRFEmbedder
# ---------------------------------------------------------------------------

class NeRFEmbedder(nn.Module):
    """
    NeRF-style sinusoidal positional encoding for a single scalar input.
    Produces [sin(2^0·π·x), cos(2^0·π·x), ..., sin(2^(L-1)·π·x), cos(2^(L-1)·π·x)].
    Output dim: 2 * L
    """
    def __init__(self, L: int):
        super().__init__()
        self.L = L
        freqs = 2.0 ** torch.arange(L, dtype=torch.float32)
        self.register_buffer('freqs', freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,) -> (B, 2*L)
        x_freq = x.unsqueeze(-1) * self.freqs * torch.pi
        return torch.stack([torch.sin(x_freq), torch.cos(x_freq)], dim=-1).flatten(start_dim=-2)

    @property
    def out_dim(self) -> int:
        return 2 * self.L


def _build_mlp(input_dim: int, layer_dims: List[int]) -> nn.Sequential:
    """Utility: build a SiLU-activated MLP from input_dim through layer_dims."""
    layers = nn.Sequential()
    for out_dim in layer_dims:
        layers.append(nn.Linear(input_dim, out_dim))
        layers.append(nn.SiLU())
        input_dim = out_dim
    return layers


# ---------------------------------------------------------------------------
# AdaLN — Adaptive Layer Normalisation
# ---------------------------------------------------------------------------
# Upgrades FiLM by normalising state_enc before modulation:
#   output = gamma(time_enc) * LayerNorm(state_enc) + beta(time_enc)
#
# Why better than FiLM:
#   FiLM applies scale/shift to raw activations, so the conditioning signal
#   must contend with whatever scale the vision encoder happens to output.
#   AdaLN normalises first, giving gamma/beta a stable, zero-mean unit-variance
#   input at every training step. This is the standard choice in DiT and SD-UNet.
#
# Init: gamma -> N(1, 0.01), beta -> N(0, 0.01) — near-identity at init,
# but with per-unit noise to break symmetry and avoid coupled gradient updates.
# ---------------------------------------------------------------------------

class FiLM(nn.Module):
    def __init__(self, time_dim: int, state_dim: int):
        super().__init__()
        self.gamma = nn.Linear(time_dim, state_dim)
        self.beta  = nn.Linear(time_dim, state_dim)

        self.gamma.weight.data.normal_(0.0, 0.01)
        self.gamma.bias.data.normal_(0.0, 0.01)
        self.beta.weight.data.normal_(0.0, 0.01)
        self.beta.bias.data.normal_(0.0, 0.01)

    def forward(self, state_enc: torch.Tensor, time_enc: torch.Tensor) -> torch.Tensor:
        return (1.0 + self.gamma(time_enc)) * state_enc + self.beta(time_enc)


# ---------------------------------------------------------------------------
# Backbone_Encoder  (shared trunk)
# ---------------------------------------------------------------------------
# Time encoding:
#   alpha and steps are concatenated after their respective NeRF embeddings
#   and passed through a single time_encoder MLP. A single MLP is sufficient
#   because both signals need to interact before conditioning the state — 
#   separating them into two MLPs then merging adds parameters without benefit.
#
#   alpha uses L=10 (continuous ratio in [0,1], benefits from high frequencies).
#   steps uses L=6  (coarser integer count, lower frequencies sufficient).
#
#   time_enc : cat[NeRF(alpha, L=10), NeRF(steps, L=6)] → time_encoder MLP → (time_encoder_dims[-1],)
#
# Cross-modal fusion — AdaLN:
#   AdaLN(state_enc, time_enc) → state_mod
#   Normalises state_enc before conditioning so gamma/beta always see a
#   stable distribution regardless of vision encoder output scale.
#
# Fusion bottleneck:
#   cat[state_mod, time_enc] → Linear(fused_dims) + SiLU
#   state_mod already encodes state via AdaLN, so concatenating raw state
#   again is redundant — it would give the optimizer three correlated views
#   of the same information and inflate the bottleneck unnecessarily.
# ---------------------------------------------------------------------------

class Backbone_Encoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        fused_dims: int,
        time_encoder_dims: List[int],
        projection_dims: List[int],
    ):
        super().__init__()

        # --- time encoders ---
        self.nerf_alpha = NeRFEmbedder(L=10)  # 20D — fine-grained continuous ratio
        self.nerf_steps = NeRFEmbedder(L=10)   # 20D — coarser integer count

        time_in_dim = self.nerf_alpha.out_dim + self.nerf_steps.out_dim  # 40D
        self.time_encoder = _build_mlp(time_in_dim, time_encoder_dims)
        time_enc_dim = time_encoder_dims[-1]

        # --- AdaLN cross-modal conditioning ---
        self.film = FiLM(time_dim=time_enc_dim, state_dim=state_dim)

        # --- fusion bottleneck ---
        # state_mod + time_enc only; raw state excluded (already encoded in state_mod)
        self.fusion_bottleneck = nn.Sequential(
            nn.Linear(state_dim * 2 + time_enc_dim, fused_dims),
            nn.SiLU(),
        )

        # --- shared trunk ---
        self.projection_encoder = _build_mlp(fused_dims, projection_dims)
        self.trunk_out_dim = projection_dims[-1]

    def forward(self, state: torch.Tensor, alpha: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        # --- time encoding ---
        time_emb = torch.cat([self.nerf_alpha(alpha), self.nerf_steps(steps)], dim=-1)
        time_enc = self.time_encoder(time_emb)                           # (B, time_enc_dim)

        # --- AdaLN cross-modal conditioning ---
        state_mod = self.film(state, time_enc)                          # (B, state_dim)

        # --- fusion bottleneck ---
        fused = self.fusion_bottleneck(torch.cat([state, state_mod, time_enc], dim=-1))

        # --- shared trunk ---
        return self.projection_encoder(fused)


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------
# Shared trunk + separate private heads for actor and critic.
#
# Why separate heads:
#   Actor and critic solve different problems: the actor parameterises a
#   distribution over actions; the critic estimates a scalar value. Sharing
#   the entire backbone forces both tasks to compete for the same internal
#   representation. Private MLP layers after the shared trunk let each head
#   develop specialised features while still benefiting from the shared
#   visual + temporal representation.
#
#   trunk  (Backbone_Encoder — shared)
#     ├── actor_private  (1 layer, width = projection_dims[-1])  →  mean_head, conc_head
#     └── critic_private (1 layer, width = projection_dims[-1])  →  mc_layer
#
# Private head width is implicitly projection_dims[-1] — no extra arg.
# ---------------------------------------------------------------------------

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
        soft: bool = True,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.soft = soft

        self.register_buffer('act_min', torch.tensor(act_min, dtype=torch.float32))
        self.register_buffer('act_max', torch.tensor(act_max, dtype=torch.float32))

        # --- shared backbone ---
        self.vision_encoder = vision_encoder
        self.backbone = Backbone_Encoder(state_dim, fused_dims, time_encoder_dims, projection_dims)
        trunk_dim = self.backbone.trunk_out_dim

        # --- private heads ---
        head_out_dim = projection_dims[-1]
        self.actor_private  = _build_mlp(trunk_dim, [head_out_dim])
        self.critic_private = _build_mlp(trunk_dim, [head_out_dim])

        # --- actor output layers ---
        self.mean_head = nn.Linear(head_out_dim, action_dim)
        self.conc_head = nn.Linear(head_out_dim, action_dim)

        mean_action_init_raw = np.log(mean_action_init / (1.0 - mean_action_init))

        if soft:
            conc_init_raw = np.log(np.exp(concentration_init - KAPPA_MIN) - 1.0)
        else:
            conc_init_raw = np.log(concentration_init - KAPPA_MIN)

        with torch.no_grad():
            self.mean_head.bias.normal_(mean_action_init_raw, 0.01)
            self.mean_head.weight.normal_(0.0, 0.01)
            self.conc_head.bias.normal_(conc_init_raw, 0.01)
            self.conc_head.weight.normal_(0.0, 0.01)

        # --- critic output layer (mc_layer wraps self.critic) ---
        self.critic = nn.Linear(head_out_dim, 1)
        # self.mc_layer = MonteCarloLayer(
        #     self.critic,
        #     dropout_p=0.05, mc_samples=512,
        #     attention_mode='attention', attend_mode='inputs',
        #     num_heads=4, embedding_size=head_out_dim // 2,
        #     query_mode='per_sample',
        # )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trunk(self, state, alpha_t, steps):
        state_enc = self.vision_encoder.encode(state)
        return self.backbone(state_enc, alpha_t, steps)

    def _alpha_beta_params(self, actor_feat: torch.Tensor):
        raw_mean = self.mean_head(actor_feat)
        raw_conc = self.conc_head(actor_feat)

        mu = torch.sigmoid(raw_mean)

        if self.soft:
            kappa = F.softplus(raw_conc) + KAPPA_MIN
        else:
            # STE clamp: numerically safe forward, straight-through backward.
            raw_conc_clamped = raw_conc + (raw_conc.clamp(RAW_CONC_MIN, RAW_CONC_MAX) - raw_conc).detach()
            kappa = torch.exp(raw_conc_clamped) + KAPPA_MIN

        alpha = 1.0 + mu * (kappa - 2.0)
        beta  = 1.0 + (1.0 - mu) * (kappa - 2.0)

        return alpha, beta, {'kappa': kappa, 'mu': mu}

    # ------------------------------------------------------------------
    # forward  (rollout collection)
    # ------------------------------------------------------------------
    def forward(self, state, alpha_t, steps, deterministic=False):
        trunk       = self._trunk(state, alpha_t, steps)
        actor_feat  = self.actor_private(trunk)
        critic_feat = self.critic_private(trunk)

        conc_alpha, conc_beta, net_dict = self._alpha_beta_params(actor_feat)
        dist  = Beta(conc_alpha, conc_beta)
        # value = self.mc_layer.get_mean_only(critic_feat)
        value = self.critic(critic_feat)  # value head is separate from MC layer for sampling efficiency; MC used only in PPO update

        if deterministic:
            action = conc_alpha / (conc_alpha + conc_beta)  # true mean, well-behaved at all κ
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action.clamp(1e-6, 1.0 - 1e-6)).sum(dim=-1)

        return action, value.squeeze(-1), log_prob, conc_alpha, conc_beta

    # ------------------------------------------------------------------
    # evaluate_actions  (PPO update)
    # ------------------------------------------------------------------
    def evaluate_actions(self, state, alpha_t, steps, actions):
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        actions = actions.clamp(1e-6, 1.0 - 1e-6)

        trunk       = self._trunk(state, alpha_t, steps)
        actor_feat  = self.actor_private(trunk)
        critic_feat = self.critic_private(trunk)

        conc_alpha, conc_beta, net_dict = self._alpha_beta_params(actor_feat)
        dist  = Beta(conc_alpha, conc_beta)
        # value = self.mc_layer.get_mean_only(critic_feat)
        value = self.critic(critic_feat)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)

        return log_prob, value.squeeze(-1), entropy, conc_alpha, conc_beta, net_dict