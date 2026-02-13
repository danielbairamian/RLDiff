import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torch.distributions import Normal
import numpy as np  

from src.MonteCarloLayer import MonteCarloLayer

class Backbone_Encoder(nn.Module):
    def __init__(self, state_dim:int, fused_dims:int, time_encoder_dims:List[int], projection_dims:List[int]):
        super(Backbone_Encoder, self).__init__()

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
        
        self.backbone_out_dim = projection_dims[-1]
    
    def forward(self, state, alpha, steps):
        time_inputs = torch.cat([alpha.unsqueeze(-1), steps.unsqueeze(-1)], dim=-1)
        
        time_encoding = time_inputs
        for layer in self.time_encoder:
            time_encoding = layer(time_encoding)
   
        fused = self.fused_latent(state, time_inputs)

        combined = torch.cat([fused, time_encoding, state], dim=-1)

        for layer in self.projection_encoder:
            combined = layer(combined)
        
        return combined


class PPOAgent(nn.Module):
    def __init__(self, state_dim:int, fused_dims:int, time_encoder_dims:List[int], projection_dims:List[int], action_dim:int, act_min:float=0.0, act_max:float=1.0, std_init:float=0.3, mean_action_init:float=0.1):
        super(PPOAgent, self).__init__()

        # Keep these to reference in your env.step() clipping later
        self.register_buffer('act_min', torch.tensor(act_min, dtype=torch.float32))
        self.register_buffer('act_max', torch.tensor(act_max, dtype=torch.float32))

        self.backbone = Backbone_Encoder(state_dim, fused_dims, time_encoder_dims, projection_dims)
        
        # 1. State-Dependent Mean
        self.action_mean = nn.Linear(self.backbone.backbone_out_dim, action_dim)
        
        # 2. State-Independent Log Std (Global Parameter)
        # Shape: (1, action_dim). Initialized to log_std_init.
        log_std_init = np.log(std_init)
        self.action_log_std = nn.Parameter(torch.full((1, action_dim), log_std_init, dtype=torch.float32))

        # inverse sigmoid for init
        # sigmoid = 1 / (1 + e^-x) --> x = log(p / (1 - p))
        # mean_action_init = np.log(mean_action_init / (1 - mean_action_init))

        # Initialize action mean to output near mean_action_init
        with torch.no_grad():
            self.action_mean.bias.normal_(mean_action_init, 0.01)
            self.action_mean.weight.normal_(0, 0.01)

        self.critic = nn.Linear(self.backbone.backbone_out_dim, 1)
        self.mc_layer = MonteCarloLayer(self.critic, 
                                        dropout_p=0.1, mc_samples=256, 
                                        attention_mode='attention', attend_mode='inputs', 
                                        num_heads=4, embedding_size=self.backbone.backbone_out_dim//4, 
                                        query_mode='per_sample')

    
    def forward(self, state, alpha, steps, deterministic=False):
        combined = self.backbone(state, alpha, steps)
        
        # Mean comes from the network
        action_mean = self.action_mean(combined)
        # action_mean = torch.sigmoid(action_mean) * (self.act_max - self.act_min) + self.act_min

        # Log std is expanded to match the batch size: [1, A] -> [B, A]
        action_log_std = self.action_log_std.expand_as(action_mean)
        
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
        # Buffer shape safety check
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        combined = self.backbone(state, alpha, steps)

        action_mean = self.action_mean(combined)
        # action_mean = torch.sigmoid(action_mean) * (self.act_max - self.act_min) + self.act_min
        action_log_std = self.action_log_std.expand_as(action_mean)
        
        probs = Normal(action_mean, torch.exp(action_log_std))
        value = self.mc_layer.get_mean_only(combined)

        # Standard Gaussian evaluation against the RAW unclipped actions
        log_prob = probs.log_prob(actions).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)

        return log_prob, value.squeeze(-1), entropy, action_mean, action_log_std

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    state_dim = 512
    fused_dims = 256
    time_encoder_dims = [32, 64, 128]
    action_dim = 1
    projection_dims = [512, 256, 64]

    agent = PPOAgent(state_dim, fused_dims, time_encoder_dims, projection_dims, action_dim).to(device)

    batch_size = 32
    state = torch.randn(batch_size, state_dim).to(device)
    alpha = torch.randn(batch_size).to(device)
    steps = torch.randn_like(alpha).to(device)

    action, value, log_prob, action_mean, action_log_std = agent(state, alpha, steps)
    print("Action Shape: ", action.shape)
    print("Value Shape: ", value.shape) 
    print("Log Prob Shape: ", log_prob.shape)