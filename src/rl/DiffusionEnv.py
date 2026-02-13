import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionEnv:
    def __init__(self, dataloader, iadb_model, AE, device, order=1, budget=10, sample_multiplier=16):
        self.dataloader = dataloader
        self.iadb_model = iadb_model
        self.AE = AE
        self.device = device
        self.dataloader_iterator = iter(self.dataloader)
        self.order = order
        self.budget = budget
        self.step_count = 0
        self.sample_multiplier = sample_multiplier # how many x0 samples to generate per x1 sample, to increase batch size for RL training

    @torch.no_grad()
    def step(self, action):
        # action shape: (B,)

        action = action.clamp(0, 1)
        
        # take a step in the environment using the IADB model
        d = self.iadb_model(self.x0, self.alpha)['sample']
        # update x0 and alpha based on the action taken
        new_alpha = self.alpha + action
        new_alpha = new_alpha.clamp(0, 1)

        # if the agent is about to take the last step, and new alpha is < 1.0, we force it to take the last step to ensure the episode ends
        new_alpha = torch.where(self.steps >= self.budget - 1, torch.ones_like(new_alpha), new_alpha)
        
        # self.x0 = self.x0 + d*(new_alpha - self.alpha).view(-1, 1, 1, 1)
        # Freeze done episodes - don't update their state
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
        self.encoded_x0 = self.AE.encode(self.x0)


        # L2 REWARD


        # # compute pairwise distances in the latent space between x0 and x1
        # # self.encoded_x0 shape: (B, latent_dim)
        # # self.x1_encoded shape: (B, latent_dim)
        # # dist shape: (B, B) where dist[i,j] is the distance between encoded_x0[i] and x1_encoded[j]
        # dist = torch.cdist(self.encoded_x0, self.x1_encoded, p=2)
        # # for each sample in the batch, find the minimum distance to any of the x1 samples
        # min_dist, _ = torch.min(dist, dim=-1)
        # min_dist *= 1e-2 # scale down the distance to keep rewards in a reasonable range

        # # rewards = torch.log1p(min_dist)
        # rewards = -min_dist # we want to minimize the distance, so we take the negative log distance as the reward
        # # if not done, reward is 0 (we only give a reward at the end of the episode based on how close we got to x1)
        # # if the episode timed out (steps >= budget), we still give the reward based on the final distance to x1

        # # COSINE SIM REWARD

        # # compute cosine similarity in the latent space between x0 and x1
        # # self.encoded_x0 shape: (B, latent_dim)
        # # self.x1_encoded shape: (B, latent_dim)
        # z_x0 = F.normalize(self.encoded_x0, dim=-1)
        # z_x1 = F.normalize(self.x1_encoded, dim=-1)

        # cosine_sim_matrix = torch.mm(z_x0, z_x1.t())  # Shape: (B, B)

        # # for each sample in the batch, find the maximum cosine similarity to any of the x1 samples
        # max_cosine_sim, _ = torch.max(cosine_sim_matrix, dim=-1)
        # rewards = max_cosine_sim


        # --- PEARSON CORRELATION REWARD ---
        
        # 1. Center the vectors (Subtract the mean of EACH vector individually)
        # We compute mean along dim=1 (the feature dimension 512)
        gen_mean = self.x0_encoded.mean(dim=1, keepdim=True)
        real_mean = self.x1_encoded.mean(dim=1, keepdim=True)
        
        z_gen_centered = self.x0_encoded - gen_mean
        z_real_centered = self.x1_encoded - real_mean
        
        # 2. Normalize the CENTERED vectors
        z_gen_norm = F.normalize(z_gen_centered, p=2, dim=1)
        z_real_norm = F.normalize(z_real_centered, p=2, dim=1)
        
        # 3. Compute Cosine on Centered vectors (= Pearson Correlation)
        # Result is strictly [-1, 1], measuring purely linear correlation
        correlation_matrix = torch.mm(z_gen_norm, z_real_norm.t())
        
        # Nearest Neighbor in Correlation Space
        rewards, _ = torch.max(correlation_matrix, dim=1)


        rewards = rewards * (self.dones).float()

        return {'x0_encoded': self.encoded_x0, 'alpha': self.alpha, 'steps': self.steps, 'x0': self.x0}, rewards, self.dones

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
