import torch
import torch.nn as nn
from typing import List
from torch.distributions import Normal

class PPOAgent(nn.Module):
    def __init__(self, state_dim:int, fused_dims:int, time_encoder_dims:List[int], projection_dims:List[int], action_dim:int):
        super(PPOAgent, self).__init__()
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
    
    def forward(self, state, alpha, steps, deterministic=False):
        combined = self.backbone(state, alpha, steps)
        
        action_mean = self.action_mean(combined)
        action_log_std = self.action_log_std(combined)
        
        probs = Normal(action_mean, torch.exp(action_log_std))
        value = self.critic(combined)
        
        if deterministic:
            action = action_mean
        else:
            action = probs.rsample() # reparameterization trick for sampling
        log_prob = probs.log_prob(action).sum(dim=-1) # sum log probs if action_dim > 1 

        return action, value.squeeze(-1), log_prob
    
    def evaluate_actions(self, state, alpha, steps, actions):
        combined = self.backbone(state, alpha, steps)
        
        action_mean = self.action_mean(combined)
        action_log_std = self.action_log_std(combined)
        
        probs = Normal(action_mean, torch.exp(action_log_std))
        value = self.critic(combined)

        log_prob = probs.log_prob(actions).sum(dim=-1) # sum log probs if action_dim > 1 
        entropy = probs.entropy().sum(dim=-1) # sum entropy if action_dim > 1

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

    action, value, log_prob = agent(state, alpha, steps)
    print("Action Shape: ", action.shape)
    print("Value Shape: ", value.shape) 
    print("Log Prob Shape: ", log_prob.shape)