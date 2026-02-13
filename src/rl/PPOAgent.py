import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torch.distributions import Normal
import numpy as np


EPSILON = 1e-6
LOG_MIN = -20
LOG_MAX = 2

class PPOAgent(nn.Module):
    def __init__(self, state_dim:int, fused_dims:int, time_encoder_dims:List[int], projection_dims:List[int], action_dim:int, act_min:float=0.0, act_max:float=1.0, log_std_init:float=-2.0, mean_action_init:float=0.1):
        super(PPOAgent, self).__init__()


        self.register_buffer('act_min', torch.tensor(act_min, dtype=torch.float32))
        self.register_buffer('act_max', torch.tensor(act_max, dtype=torch.float32))
        self.register_buffer('act_scale', torch.tensor((act_max - act_min) / 2.0, dtype=torch.float32))
        self.register_buffer('act_bias', torch.tensor((act_max + act_min) / 2.0, dtype=torch.float32))

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
        input_dim = fused_dims + time_encoder_dims[-1] + state_dim
        for i in range(len(projection_dims)):
            output_dim = projection_dims[i]
            self.projection_encoder.append(
                nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.SiLU(),
                )
            )
            input_dim = output_dim
        
        self.action_mean = nn.Linear(input_dim, action_dim)
        self.action_log_std = nn.Linear(input_dim, action_dim)

        # Initialize action mean to output small actions
        with torch.no_grad():
            # We want tanh(x) * 0.5 + 0.5 ≈ mean_action_init
            # So tanh(x) ≈ (mean_action_init - 0.5) / 0.5
            # So x ≈ atanh((mean_action_init - 0.5) / 0.5)
            target_tanh = (mean_action_init - self.act_bias) / self.act_scale
            target_tanh = torch.clamp(target_tanh, -0.99, 0.99)  # Keep in valid range
            target_pretanh = torch.atanh(target_tanh)
            
            # Bias the output to this value
            self.action_mean.bias.normal_(target_pretanh.item(), 0.01)  # Small noise around the target pre-tanh value
            # Small weights so state dependency builds up gradually
            self.action_mean.weight.normal_(0, 0.01)
        
        # Initialize log_std to low exploration
        with torch.no_grad():
            self.action_log_std.bias.normal_(log_std_init, 0.1)  # e.g., -2.0 → std ≈ 0.135
            self.action_log_std.weight.normal_(0, 0.01)

        self.critic = nn.Linear(input_dim, 1)
    
    def backbone(self, state, alpha, steps):
        # Time encoding inputs: alpha and steps
        time_inputs = torch.cat([alpha.unsqueeze(-1), steps.unsqueeze(-1)], dim=-1)
        
        # Encode time inputs
        time_encoding = time_inputs
        for layer in self.time_encoder:
            time_encoding = layer(time_encoding)
   
        # Fuse state and time encoding (time is pre-encoded)
        fused = self.fused_latent(state, time_inputs)

        # combine fused representation with time encoding and state for final projection
        combined = torch.cat([fused, time_encoding, state], dim=-1) # Use last time encoding for projection

        for layer in self.projection_encoder:
            combined = layer(combined)
        
        return combined
    

    def _tanh_squash_correction(self, unsquashed_action):
        """
        Compute the log probability correction for tanh squashing.
        
        Uses OpenAI Spinning Up's numerically stable formula:
        correction = 2 * (log(2) - u - softplus(-2*u))
        
        where u is the unsquashed action.
        
        This is equivalent to log(1 - tanh²(u)) but more numerically stable.
        
        Derivation:
        -----------
        1. We want: log(1 - tanh²(u))
        2. Since 1 - tanh²(u) = 4/(e^u + e^(-u))²
        3. log(1 - tanh²(u)) = log(4) - 2*log(e^u + e^(-u))
        4. Using logsumexp: log(e^u + e^(-u)) = u + log(1 + e^(-2u)) = u + softplus(-2u)
        5. Therefore: log(1 - tanh²(u)) = 2*log(2) - 2*(u + softplus(-2u))
                                         = 2*(log(2) - u - softplus(-2u))
        
        For multi-dimensional actions, we sum over action dimensions.
        We also need to account for scaling from [-1,1] to [act_min, act_max].
        """
        # Correction for tanh squashing: 2*(log(2) - u - softplus(-2*u))
        correction = 2.0 * (np.log(2) - unsquashed_action - F.softplus(-2.0 * unsquashed_action))
        correction = correction.sum(dim=-1)
        
        # Additional correction for scaling from [-1,1] to [act_min, act_max]
        # When we scale by act_scale, we need to subtract log(act_scale) for each dimension
        scale_correction = torch.log(self.act_scale) * unsquashed_action.shape[-1]
        
        return correction - scale_correction
    
    def forward(self, state, alpha, steps, deterministic=False):
        combined = self.backbone(state, alpha, steps)
        
        action_mean = self.action_mean(combined)
        action_log_std = self.action_log_std(combined)
        action_log_std = torch.clamp(action_log_std, -20, 2) # clamp log std for numerical stability

        
        probs = Normal(action_mean, torch.exp(action_log_std))
        value = self.critic(combined)
        
        if deterministic:
            unsquashed_action = action_mean
        else:
            unsquashed_action = probs.rsample() # reparameterization trick for sampling
        log_prob = probs.log_prob(unsquashed_action).sum(dim=-1) # sum log probs if action_dim > 1 
        log_prob -= self._tanh_squash_correction(unsquashed_action)

        action = torch.tanh(unsquashed_action) * self.act_scale + self.act_bias

        return action, value.squeeze(-1), log_prob, torch.tanh(action_mean) * self.act_scale + self.act_bias, action_log_std
    
    def evaluate_actions(self, state, alpha, steps, actions):
        combined = self.backbone(state, alpha, steps)

        action_mean = self.action_mean(combined)
        action_log_std = self.action_log_std(combined)
        action_log_std = torch.clamp(action_log_std, LOG_MIN, LOG_MAX) # clamp log std for numerical stability
        
        probs = Normal(action_mean, torch.exp(action_log_std))
        value = self.critic(combined)

        # Unsquash actions: u = atanh((a - act_bias) / act_scale)
        normalized = (actions - self.act_bias) / self.act_scale
        # Clamp to valid tanh range to avoid numerical issues
        normalized = torch.clamp(normalized, -1.0 + EPSILON, 1.0 - EPSILON)
        unsquashed_actions = torch.atanh(normalized)
        
        # Compute log probability in unsquashed space
        log_prob = probs.log_prob(unsquashed_actions).sum(dim=-1)
        
        # Apply tanh squashing correction
        log_prob = log_prob - self._tanh_squash_correction(unsquashed_actions)
        
        # Entropy is computed in the unsquashed space
        # (the entropy of the base Gaussian distribution)
        entropy = probs.entropy().sum(dim=-1)

        return log_prob, value.squeeze(-1), entropy

if __name__ == "__main__":
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    # dummy input dimensions
    state_dim = 512
    fused_dims = 256
    time_encoder_dims = [32, 64, 128]
    action_dim = 1
    projection_dims = [512, 256, 64]

    agent = PPOAgent(state_dim, fused_dims, time_encoder_dims, projection_dims, action_dim).to(device)

    # dummy inputs
    batch_size = 32
    state = torch.randn(batch_size, state_dim).to(device)
    alpha = torch.randn(batch_size).to(device)
    steps = torch.randn_like(alpha).to(device)

    action, value, log_prob, action_mean, action_log_std = agent(state, alpha, steps)
    print("Action Shape: ", action.shape)
    print("Value Shape: ", value.shape) 
    print("Log Prob Shape: ", log_prob.shape)