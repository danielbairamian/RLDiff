import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torch.distributions import Normal
import numpy as np  

from src.MonteCarloLayer import MonteCarloLayer
from src.latent_encoder.VisionEncoder import VisionEncoder


class NeRFEmbedder(nn.Module):
    """
    NeRF-style sinusoidal positional encoding.
    For each input scalar, produces [sin(2^0 * pi * x), cos(2^0 * pi * x), ..., sin(2^(L-1) * pi * x), cos(2^(L-1) * pi * x)]
    Output dim per scalar: 2 * L
    """
    def __init__(self, L: int):
        super().__init__()
        self.L = L
        # Precompute frequency bands: [2^0, 2^1, ..., 2^(L-1)]
        freqs = 2.0 ** torch.arange(L, dtype=torch.float32)  # (L,)
        self.register_buffer('freqs', freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., D) — any number of leading dims, D input scalars
        returns: (..., D * 2 * L)
        """
        # x: (..., D) -> (..., D, 1) * (L,) -> (..., D, L)
        x_freq = x.unsqueeze(-1) * self.freqs * torch.pi  # (..., D, L)
        sins = torch.sin(x_freq)   # (..., D, L)
        coss = torch.cos(x_freq)   # (..., D, L)
        # interleave: (..., D, 2L) -> (..., D*2L)
        embedded = torch.stack([sins, coss], dim=-1)       # (..., D, L, 2)
        return embedded.flatten(start_dim=-3)              # (..., D * L * 2)

    @property
    def out_dim_per_scalar(self) -> int:
        return 2 * self.L

class Backbone_Encoder(nn.Module):
    def __init__(self, state_dim:int, fused_dims:int, time_encoder_dims:List[int], projection_dims:List[int]):
        super(Backbone_Encoder, self).__init__()

        self.nerf_embedder = NeRFEmbedder(L=16)  # Example with L=4, can be tuned
        nerf_out_dim = self.nerf_embedder.out_dim_per_scalar * 2  # alpha and steps
        self.time_encoder = nn.ModuleList()
        input_dim = 2 # alpha and steps
        for i in range(len(time_encoder_dims)):
            output_dim = time_encoder_dims[i]
            self.time_encoder.append(
                nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.SiLU(),
                )
            )
            input_dim = output_dim
        
        
        # takes the state and time encoding as input and outputs action logits
        self.fused_latent = nn.Bilinear(state_dim, 2, fused_dims)

        self.projection_encoder = nn.Sequential()
        input_dim = fused_dims + time_encoder_dims[-1] + state_dim + 2  + nerf_out_dim # fused + time encoding + state + raw time inputs
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

        nerf_embedded = self.nerf_embedder(time_inputs)
        combined = torch.cat([fused, time_encoding, state, time_inputs, nerf_embedded], dim=-1)

        for layer in self.projection_encoder:
            combined = layer(combined)
        
        return combined


class PPOAgent(nn.Module):
    def __init__(self, vision_encoder:VisionEncoder, state_dim:int, fused_dims:int, time_encoder_dims:List[int], projection_dims:List[int], action_dim:int, act_min:float=0.0, act_max:float=1.0, std_init:float=0.3, mean_action_init:float=0.1):
        super(PPOAgent, self).__init__()

        # Keep these to reference in your env.step() clipping later
        self.register_buffer('act_min', torch.tensor(act_min, dtype=torch.float32))
        self.register_buffer('act_max', torch.tensor(act_max, dtype=torch.float32))

        self.vision_encoder = vision_encoder
        self.backbone = Backbone_Encoder(state_dim, fused_dims, time_encoder_dims, projection_dims)
        
        # 1. State-Dependent Mean
        self.action_mean = nn.Linear(self.backbone.backbone_out_dim, action_dim)
        
        # 2. State-Independent Log Std (Global Parameter)
        # Shape: (1, action_dim). Initialized to log_std_init.
        log_std_init = np.log(std_init)

        self.action_log_std = nn.Linear(self.backbone.backbone_out_dim, action_dim)

        # inverse sigmoid for init
        # sigmoid = 1 / (1 + e^-x) --> x = log(p / (1 - p))
        # mean_action_init = np.log(mean_action_init / (1 - mean_action_init))

        # Initialize action mean to output near mean_action_init
        with torch.no_grad():
            self.action_mean.bias.normal_(mean_action_init, 0.01)
            self.action_mean.weight.normal_(0, 0.01)
            self.action_log_std.bias.normal_(log_std_init, 0.01)
            self.action_log_std.weight.normal_(0, 0.01)

        self.critic = nn.Linear(self.backbone.backbone_out_dim, 1)
        self.mc_layer = MonteCarloLayer(self.critic, 
                                        dropout_p=0.05, mc_samples=256, 
                                        attention_mode='attention', attend_mode='inputs', 
                                        num_heads=8, embedding_size=self.backbone.backbone_out_dim//2, 
                                        query_mode='per_sample')

    
    def forward(self, state, alpha, steps, deterministic=False):
        state = self.vision_encoder.encode(state)   
        combined = self.backbone(state, alpha, steps)
        
        # Mean comes from the network
        action_mean = self.action_mean(combined)
        # action_mean = torch.sigmoid(action_mean) * (self.act_max - self.act_min) + self.act_min

        # Log std is expanded to match the batch size: [1, A] -> [B, A]
        action_log_std = self.action_log_std(combined)
        # action_log_std = torch.clamp(action_log_std, LOG_MIN, LOG_MAX)  # Clamp for numerical stability

        probs = Normal(action_mean, torch.exp(action_log_std))
        value = self.mc_layer.get_mean_only(combined)
        
        if deterministic:
            action = action_mean
        else:
            action = probs.sample() 
            
        # Clean math: no squashing, no correction terms
        log_prob = probs.log_prob(action).sum(dim=-1)

        return action, value.squeeze(-1), log_prob, action_mean, action_log_std
    
    def evaluate_actions(self, state, alpha, steps, actions):
        state = self.vision_encoder.encode(state)
        # Buffer shape safety check
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        combined = self.backbone(state, alpha, steps)

        action_mean = self.action_mean(combined)
        # action_mean = torch.sigmoid(action_mean) * (self.act_max - self.act_min) + self.act_min
        action_log_std = self.action_log_std(combined)
        # action_log_std = torch.clamp(action_log_std, LOG_MIN, LOG_MAX)  # Clamp for numerical stability
        
        probs = Normal(action_mean, torch.exp(action_log_std))
        value = self.mc_layer.get_mean_only(combined)

        # Standard Gaussian evaluation against the RAW unclipped actions
        log_prob = probs.log_prob(actions).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)

        return log_prob, value.squeeze(-1), entropy, action_mean, action_log_std