import torch


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

        # compute pairwise distances in the latent space between x0 and x1
        # self.encoded_x0 shape: (B, latent_dim)
        # self.x1_encoded shape: (B, latent_dim)
        # dist shape: (B, B) where dist[i,j] is the distance between encoded_x0[i] and x1_encoded[j]
        dist = torch.cdist(self.encoded_x0, self.x1_encoded, p=2)
        # for each sample in the batch, find the minimum distance to any of the x1 samples
        min_dist, _ = torch.min(dist, dim=-1) / self.encoded_x0.shape[-1] # normalize by latent dimension for better scaling of rewards
        # rewards = torch.log1p(min_dist)
        rewards = -min_dist # we want to minimize the distance, so we take the negative log distance as the reward
        # if not done, reward is 0 (we only give a reward at the end of the episode based on how close we got to x1)
        # if the episode timed out (steps >= budget), we still give the reward based on the final distance to x1
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



if __name__ == "__main__":
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    # dummy dataloader
    dataloader = [(torch.randn(4, 3, 32, 32).to(device), torch.randint(0, 10, (8,)).to(device)) for _ in range(10)]
    
    # dummy AE
    class DummyAE(torch.nn.Module):
        def __init__(self, input_dim=3072, latent_dim=128):
            super(DummyAE, self).__init__()
            # Use a fixed projection so it's deterministic but not just returning 'x'
            self.proj = torch.nn.Linear(input_dim, latent_dim)
            torch.nn.init.orthogonal_(self.proj.weight)
            self.proj.requires_grad_(False)

        def encode(self, x):
            # x shape: (B, 3, 32, 32) -> flatten to (B, 3072)
            flat_x = x.view(x.size(0), -1)
            with torch.no_grad():
                return self.proj(flat_x)

    AE = DummyAE().to(device)
    # dummy IADB model
    class DummyIADBModel(torch.nn.Module):
        def forward(self, x, alpha):
            return {'sample': torch.randn_like(x)}
    iadb_model = DummyIADBModel().to(device)

    env = DiffusionEnv(dataloader, iadb_model, AE, device, budget=10)
    obs = env.reset()
    done = torch.zeros(obs['x0_encoded'].shape[0], dtype=torch.bool, device=device)
    while not done.all():
        action = torch.randn(obs['alpha'].shape[0]).to(device) * 0.01 # small random action
        obs, rewards, done = env.step(action)
        print(f"Rewards: {rewards}, Done: {done}", "Steps: ", obs['steps'], "Alpha: ", obs['alpha'])  