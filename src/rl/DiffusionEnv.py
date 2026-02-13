import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

class DiffusionEnv:
    def __init__(self, dataloader, iadb_model, AE, device, order=1, budget=10, sample_multiplier=16, denorm_fn=None):
        self.dataloader = dataloader
        self.iadb_model = iadb_model
        self.AE = AE
        self.device = device
        self.dataloader_iterator = iter(self.dataloader)
        self.denorm_fn = denorm_fn
        self.order = order
        self.budget = budget
        self.sample_multiplier = sample_multiplier # how many x0 samples to generate per x1 sample, to increase batch size for RL training
        
        # LPIPS perceptual loss network (lower = more similar)
        self.lpips_net = lpips.LPIPS(net='vgg', verbose=False).to(device).eval()
        for param in self.lpips_net.parameters():
            param.requires_grad = False

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

        # LPIPS perceptual reward computation
        # LPIPS expects inputs in [-1, 1] range
        x0_lpips = self.denorm_fn(self.x0) * 2 - 1  # [0,1] -> [-1,1]
        x1_lpips = self.denorm_fn(self.x1) * 2 - 1  # shape: (B_x1, C, H, W)
        
        gen_mean = self.x0_encoded.mean(dim=0, keepdim=True)  # (1, latent_dim)
        real_mean = self.x1_encoded.mean(dim=0, keepdim=True)  # (1, latent_dim)

        z_gen_centered = self.x0_encoded - gen_mean  # (B_x0, latent_dim)
        z_real_centered = self.x1_encoded - real_mean  # (B_x1, latent_dim)

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
    dummy_denorm_fn = lambda x: x.clamp(0, 1)  # Clamp to [0,1] for LPIPS compatibility
    env = DiffusionEnv(
        dataloader=dummy_dataloader,
        iadb_model=iadb,
        AE=ae,
        device=device,
        order=1,
        budget=10,
        sample_multiplier=sample_mult,
        denorm_fn=dummy_denorm_fn
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
    print("Test Complete. LPIPS reward working!")
    print("Test Complete. No crashes!")