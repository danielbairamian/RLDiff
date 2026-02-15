import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import time

class DiffusionEnv:
    def __init__(self, dataloader, iadb_model, AE, device, order=1, budget=10, sample_multiplier=16, denorm_fn=None):
        self.dataloader = dataloader
        self.iadb_model = iadb_model
        self.AE = AE
        self.device = device
        self.dataloader_iterator = iter(self.dataloader)
        self.denorm_fn = denorm_fn
        self.order = order
        self.budget = budget // self.order  # Adjust budget based on the order of the method
        self.sample_multiplier = sample_multiplier # how many x0 samples to generate per x1 sample, to increase batch size for RL training
        
        # LPIPS perceptual loss network (lower = more similar)
        self.lpips_net = lpips.LPIPS(net='vgg', verbose=False).to(device).eval()
        for param in self.lpips_net.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def step(self, action):
        # action shape: (B,)

        action = action.clamp(0, 1)
        
        
        # update x0 and alpha based on the action taken
        new_alpha = self.alpha + action
        new_alpha = new_alpha.clamp(0, 1)

        # if the agent is about to take the last step, and new alpha is < 1.0, we force it to take the last step to ensure the episode ends
        new_alpha = torch.where(self.steps >= self.budget - 1, torch.ones_like(new_alpha), new_alpha)
        
        # self.x0 = self.x0 + d*(new_alpha - self.alpha).view(-1, 1, 1, 1)
        # Freeze done episodes - don't update their state


        if self.order == 1:
            d = self.iadb_model(self.x0, self.alpha)['sample']
        elif self.order == 2:
            # Midpoint alpha for the second-order method
            mid_alpha = (new_alpha + self.alpha) / 2
            
            # intermediate step (x_t+1/2)
            d_1 = self.iadb_model(self.x0, self.alpha)['sample']
            x_mid = self.x0 + d_1 * (mid_alpha - self.alpha).view(-1, 1, 1, 1)
            # second step (x_t+1)
            d = self.iadb_model(x_mid, mid_alpha)['sample']
        

        self.x0 = torch.where(
            self.dones.view(-1, 1, 1, 1), 
            self.x0,  # Keep old x0 if done
            self.x0 + d * (new_alpha - self.alpha).view(-1, 1, 1, 1)  # Update if active
        )

        
        # update alpha
        self.alpha = torch.where(self.dones, self.alpha, new_alpha)

        # counter for episode termination per sample in the batch, only if it's not done
        self.steps = self.steps + (~self.dones).int()
        # episode is done if alpha >= 1.0 or if we exceed the budget of steps
        self.dones = self.dones | (self.alpha >= 1.0) | (self.steps >= self.budget)
        
        # encode the new x0 in the latent space using the AE encoder
        self.x0_encoded = self.AE.encode(self.x0)

        # LPIPS perceptual reward computation
        # LPIPS expects inputs in [-1, 1] range
        x0_lpips = self.denorm_fn(self.x0) * 2 - 1  # [0,1] -> [-1,1]
        x1_lpips = self.denorm_fn(self.x1) * 2 - 1  # shape: (B_x1, C, H, W)
        
        gen_mean = self.x0_encoded.mean(dim=-1, keepdim=True)  # (B_x0, 1)
        real_mean = self.x1_encoded.mean(dim=-1, keepdim=True)  # (B_x1, 1)

        z_gen_centered = self.x0_encoded - gen_mean  # (B_x0, latent_dim)
        z_real_centered = self.x1_encoded - real_mean  # (B_x1, latent_dim)

        z_gen_centered = F.normalize(z_gen_centered, dim=-1)  # (B_x0, latent_dim)
        z_real_centered = F.normalize(z_real_centered, dim=-1)  # (B_x1, latent_dim)


        correlation_matrix = torch.matmul(z_gen_centered, z_real_centered.T)  # (B_x0, B_x1)
        _, correlation_indices = correlation_matrix.max(dim=1)  # (B_x0,)   
        
        nearest_x1_lpips = x1_lpips[correlation_indices]  # shape: (B_x0, C, H, W)
        lpips_dist = self.lpips_net(x0_lpips, nearest_x1_lpips).squeeze()  # shape: (B_x0,)        
        # lpips_reward = torch.exp(-min_lpips_dist)
        lpips_reward = -lpips_dist  # Using negative distance directly as reward, since LPIPS is already a perceptual similarity metric where lower is better

        rewards = lpips_reward * (self.dones).float()

        return {'x0_encoded': self.x0_encoded, 'alpha': self.alpha, 'steps': self.steps, 'x0': self.x0}, rewards, self.dones

    @torch.no_grad()
    def reset(self):
        try:
            x1, _ = next(self.dataloader_iterator)
        except StopIteration:
            self.dataloader_iterator = iter(self.dataloader)
            x1, _ = next(self.dataloader_iterator)
        
        # x1 shape: (B, C, H, W)
        self.x1 = x1.to(self.device)
        # x1_encoded shape: (B, latent_dim)
        self.x1_encoded = self.AE.encode(self.x1)

        # x0 shape: (B, C, H, W)
        self.x0 = torch.randn_like(self.x1).to(self.device)
        self.x0 = self.x0[:self.x1.shape[0]//self.sample_multiplier] # ensure x0 and x1 have the same batch size
        # x0_encoded shape: (B, latent_dim)
        self.x0_encoded = self.AE.encode(self.x0)
        self.alpha = torch.zeros(self.x0.shape[0], device=self.device) # Start at alpha=0 (x0)
        self.dones = torch.zeros(self.x0.shape[0], dtype=torch.bool, device=self.device)
        self.steps = torch.zeros(self.x0.shape[0], dtype=torch.int, device=self.device)

        return {'x0_encoded': self.x0_encoded, 'alpha': self.alpha, 'steps': self.steps, 'x0': self.x0}

