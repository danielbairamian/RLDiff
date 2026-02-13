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
        self.x0_encoded = self.AE.encode(self.x0)


        # L2 REWARD


        # compute pairwise distances in the latent space between x0 and x1
        # self.x0_encoded shape: (B, latent_dim)
        # self.x1_encoded shape: (B, latent_dim)
        # dist shape: (B, B) where dist[i,j] is the distance between x0_encoded[i] and x1_encoded[j]
        # dist = torch.cdist(self.x0_encoded, self.x1_encoded, p=2)
        # # for each sample in the batch, find the minimum distance to any of the x1 samples
        # min_dist, _ = torch.min(dist, dim=-1)
        # min_dist *= 1e-2 # scale down the distance to keep rewards in a reasonable range

        # # rewards = torch.log1p(min_dist)
        # rewards = -min_dist # we want to minimize the distance, so we take the negative log distance as the reward
    
        # # COSINE SIM REWARD

        # # compute cosine similarity in the latent space between x0 and x1
        # # self.x0_encoded shape: (B, latent_dim)
        # # self.x1_encoded shape: (B, latent_dim)
        # z_x0 = F.normalize(self.x0_encoded, dim=-1)
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


        # # intrinsic reward: decoder consistency
        # x0_decoded = self.AE.decode(self.x0_encoded)
        # reconstruction_error = F.l1_loss(x0_decoded, self.x0, reduction='none').mean(dim=[1,2,3]) # shape: (B,)
        # intrinsic_reward = -reconstruction_error # we want to minimize reconstruction error, so intrinsic reward is negative error
        # rewards = rewards + intrinsic_reward # combine extrinsic
        
        
        # if not done, reward is 0 (we only give a reward at the end of the episode based on how close we got to x1)
        # if the episode timed out (steps >= budget), we still give the reward based on the final distance to x1
        rewards = rewards * (self.dones).float()

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



if __name__ == "__main__":
    # --- DUMMY CLASSES ---
    class DummyAE(nn.Module):
        def __init__(self, latent_dim=512):
            super().__init__()
            self.latent_dim = latent_dim
            
        def encode(self, x):
            # Outputs a random latent vector of the correct shape [B, Latent_Dim]
            return torch.ones(x.shape[0], self.latent_dim, device=x.device)
            
        def decode(self, z):
            # Outputs a random image of the correct shape [B, 3, 32, 32]
            return torch.ones(z.shape[0], 3, 32, 32, device=z.device)

    class DummyIADB(nn.Module):
        def forward(self, x, alpha):
            # The diffusion model predicts a direction 'd' of the same shape as the image
            return {'sample': torch.randn_like(x)}

    # --- DUMMY DATA ---
    # Creates a simple list that acts as an iterator yielding (image_batch, labels)
    batch_size = 32
    sample_mult = 4
    img_channels, img_size = 3, 32
    dummy_dataloader = [(torch.randn(batch_size, img_channels, img_size, img_size), None) for _ in range(10)]

    # --- INITIALIZATION ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Testing on device: {device}\n")

    ae = DummyAE(latent_dim=512).to(device)
    iadb = DummyIADB().to(device)

    env = DiffusionEnv(
        dataloader=dummy_dataloader,
        iadb_model=iadb,
        AE=ae,
        device=device,
        order=1,
        budget=10,
        sample_multiplier=sample_mult
    )

    # --- RUN LOCAL TEST ---
    print("--- Testing Reset ---")
    obs = env.reset()
    expected_x0_batch = batch_size // sample_mult
    
    print(f"Real Image Batch (x1): {env.x1.shape}")
    print(f"Generated Image Batch (x0): {obs['x0'].shape}")
    print(f"Latent Shape (x0_encoded): {obs['x0_encoded'].shape}")
    print(f"Alpha Initial: {obs['alpha']}\n")

    assert obs['x0'].shape[0] == expected_x0_batch, "Sample multiplier slicing failed."

    print("--- Testing Step ---")
    # Dummy action: Agent wants to take a step size of ~0.1
    dummy_actions = torch.rand(expected_x0_batch, device=device) * 0.2 
    
    next_obs, rewards, dones = env.step(dummy_actions)
    
    print(f"Actions Taken: {dummy_actions}")
    print(f"New Alphas: {next_obs['alpha']}")
    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}\n")
    
    print("--- Testing Episode Loop (Fast-Forward to Done) ---")
    # Force the environment to hit the budget
    for _ in range(10):
        next_obs, rewards, dones = env.step(dummy_actions)
        if dones.all():
            break
            
    print(f"Final Alphas: {next_obs['alpha']}")
    print(f"Final Rewards: {rewards}")
    print(f"Final Steps: {next_obs['steps']}")
    print("Test Complete. No crashes!")