import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

# ==========================================
# 1. DUMMY COMPONENTS FOR TESTING
# ==========================================
class DummyIADB(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x, alpha):
        # Ensure alpha is a tensor for conditioning
        out = self.net(x)
        return {'sample': out}

def get_dummy_dataloader(batch_size=8, img_size=64, num_batches=5):
    for _ in range(num_batches):
        images = torch.rand((batch_size, 3, img_size, img_size))
        labels = torch.zeros(batch_size)
        yield images, labels

# ==========================================
# 2. DIFFUSION ENVIRONMENT (TRUE N-TH ORDER)
# ==========================================
class DiffusionEnvNthOrder:
    def __init__(self, dataloader, iadb_model, device, budget=10, order=-1, sample_multiplier=2, denorm_fn=None):
        self.dataloader = dataloader
        self.iadb_model = iadb_model
        self.device = device
        self.dataloader_iterator = iter(self.dataloader)
        self.denorm_fn = denorm_fn if denorm_fn else lambda x: x 
        self.budget = budget 
        self.sample_multiplier = sample_multiplier 
        self.max_order = self.budget if order == -1 else order 
        
        print("Loading InceptionV3 (Official FID Feature Space)...")
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Identity()
        self.inception = self.inception.to(device).eval()
        
        for param in self.inception.parameters():
            param.requires_grad = False
            
        self.inception_preprocess = T.Compose([
            T.Resize((299, 299), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def get_inception_features(self, x):
        x_clean = self.denorm_fn(x) 
        if x_clean.shape[1] == 1:
            x_clean = x_clean.repeat(1, 3, 1, 1)
        x_preprocessed = self.inception_preprocess(x_clean)
        return self.inception(x_preprocessed)

    @torch.no_grad()
    def step(self, action, order_action):
        action = action.clamp(0, 1)
        order_action = order_action.clamp(0, 1)
        
        old_dones = self.dones.clone()
        active_mask = ~self.dones
        
        # 1. Map continuous order to [1, self.max_order]
        requested_order = torch.round(order_action * (self.max_order - 1)) + 1
        requested_order = requested_order.int()
        
        # 2. Constraints: Cap at remaining budget
        remaining_budget = self.budget - self.steps
        actual_order = torch.min(requested_order, remaining_budget).max(torch.ones_like(requested_order))
        
        # 3. Setup Integration targets
        # h is the full step size for this RL action
        h = action 
        target_alpha = (self.alpha + h).clamp(0, 1)
        
        # Force alpha to 1.0 if budget is spent
        target_alpha = torch.where(actual_order >= remaining_budget, torch.ones_like(target_alpha), target_alpha)
        
        # Calculate the actual delta for this specific step
        actual_h = target_alpha - self.alpha

        if active_mask.any():
            max_steps = actual_order[active_mask].max().item()
            
            x_start = self.x0.clone()  # Original state before integration
            x_probe = self.x0.clone()  # Working state for Runge-Kutta stages
            
            # --- MULTI-STAGE N-TH ORDER INTEGRATION ---
            # If actual_order=2, this mimics Midpoint: 1st pass at start, 2nd pass at mid.
            for i in range(max_steps):
                step_mask = active_mask & (actual_order > i)
                if not step_mask.any():
                    break
                
                # Determine probing alpha for this stage
                # For Midpoint (N=2): i=0 -> alpha_start; i=1 -> alpha_start + h/2
                # For Euler (N=1): i=0 -> alpha_start
                fraction = (i / actual_order[step_mask]).float()
                # For standard Midpoint/RK logic, we evaluate at the midpoint for the final update
                # Here we use a 0.5 shift for all intermediate stages
                if i > 0:
                    fraction = 0.5 

                probe_alpha = self.alpha[step_mask] + (actual_h[step_mask] * fraction)
                
                # Forward pass on subset
                d_sub = self.iadb_model(x_probe[step_mask], probe_alpha)['sample']
                h_view = actual_h[step_mask].view(-1, 1, 1, 1)
                
                if i < max_steps - 1:
                    # PROBE: Move x_probe to the midpoint for the next evaluation
                    x_probe[step_mask] = x_start[step_mask] + d_sub * (h_view * 0.5)
                else:
                    # FINAL: Use the last calculated slope to move the real state the full distance h
                    self.x0[step_mask] = x_start[step_mask] + d_sub * h_view
            
        # 4. Update environment states
        self.alpha = torch.where(active_mask, target_alpha, self.alpha)
        self.steps = torch.where(active_mask, self.steps + actual_order, self.steps)
        self.dones = self.dones | (self.alpha >= 1.0) | (self.steps >= self.budget)

        # --- REWARD COMPUTATION ---
        just_done = self.dones & ~old_dones
        if just_done.any():
            x0_finished = self.x0[just_done]
            z_gen_inception = self.get_inception_features(x0_finished)
            z_gen_norm = F.normalize(z_gen_inception, p=2.0, dim=1)
            
            sim_matrix = torch.matmul(z_gen_norm, self.z_real_norm.T)
            topk_sim, _ = torch.topk(sim_matrix, self.k, dim=1)
            self.episode_rewards[just_done] = topk_sim.mean(dim=1)

        rewards = self.episode_rewards * self.dones.float()
        return {'alpha': self.alpha, 'steps': self.steps, 'x0': self.x0}, rewards, self.dones

    @torch.no_grad()
    def reset(self):
        try:
            x1, _ = next(self.dataloader_iterator)
        except (StopIteration, AttributeError):
            self.dataloader_iterator = iter(self.dataloader)
            x1, _ = next(self.dataloader_iterator)
        
        self.x1 = x1.to(self.device)
        x1_inception = self.get_inception_features(self.x1)
        self.z_real_norm = F.normalize(x1_inception, p=2.0, dim=1)
        self.k = 1 

        self.x0 = torch.randn_like(self.x1).to(self.device)
        self.x0 = self.x0[:self.x1.shape[0]//self.sample_multiplier] 
        
        self.alpha = torch.zeros(self.x0.shape[0], device=self.device) 
        self.dones = torch.zeros(self.x0.shape[0], dtype=torch.bool, device=self.device)
        self.steps = torch.zeros(self.x0.shape[0], dtype=torch.int, device=self.device)
        self.episode_rewards = torch.zeros(self.x0.shape[0], device=self.device)

        return {'alpha': self.alpha, 'steps': self.steps, 'x0': self.x0}

# ==========================================
# 3. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_model = DummyIADB().to(device)
    dummy_loader = get_dummy_dataloader(batch_size=8, img_size=64)
    
    env = DiffusionEnvNthOrder(
        dataloader=dummy_loader, 
        iadb_model=dummy_model, 
        device=device, 
        budget=10, 
        sample_multiplier=2,
        order=2, # Testing with max order of 2
    )
    
    state = env.reset()
    num_agents = state['x0'].shape[0]
    
    for step_num in range(1, 10):
        print(f"\n=== Env Step {step_num} ===")
        action = torch.full((num_agents,), 0.2, device=device) + torch.randn_like(state['alpha']) * 0.05
        order_action = torch.randn(num_agents, device=device)

        print(f"Raw Action: {action.cpu().numpy().round(3)}")
        print(f"Raw Order Action: {order_action.cpu().numpy().round(3)}")
        
        state, rewards, dones = env.step(action, order_action)
        
        print(f"Alphas: {state['alpha'].cpu().numpy().round(3)}")
        print(f"Steps:  {state['steps'].cpu().numpy()}")
        print(f"Dones:  {dones.cpu().numpy()}")
        print(f"Rewards: {rewards.cpu().numpy().round(3)}")
        
        if dones.all(): break