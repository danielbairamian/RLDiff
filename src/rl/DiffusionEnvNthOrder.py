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
        # A simple conv layer to simulate the diffusion model forward pass
        self.net = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x, alpha):
        # Dummy model: just returns a tensor of the same shape
        # alpha is ignored here for simplicity, but normally conditions the network
        out = self.net(x)
        return {'sample': out}

def get_dummy_dataloader(batch_size=8, img_size=64, num_batches=5):
    # Yields random images and dummy labels
    for _ in range(num_batches):
        images = torch.rand((batch_size, 3, img_size, img_size))
        labels = torch.zeros(batch_size)
        yield images, labels

# ==========================================
# 2. DIFFUSION ENVIRONMENT
# ==========================================
class DiffusionEnvNthOrder:
    def __init__(self, dataloader, iadb_model, device, budget=10, order=-1, sample_multiplier=2, denorm_fn=None):
        self.dataloader = dataloader
        self.iadb_model = iadb_model
        
        self.device = device
        self.dataloader_iterator = iter(self.dataloader)
        self.denorm_fn = denorm_fn if denorm_fn else lambda x: x # Default to identity
        self.budget = budget 
        self.sample_multiplier = sample_multiplier 
        self.max_order =  self.budget if order == -1 else order # <-- too costly to allow full budget in one step, so we will cap at a reasonable max order (e.g., 5)
        
        # --- THE GOLD-STANDARD REWARD (InceptionV3) ---
        print("Loading InceptionV3 (Official FID Feature Space)...")
        # Suppress warnings for testing by using the current weights API
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
        """Helper to denorm, resize, and extract FID embeddings."""
        x_clean = self.denorm_fn(x) 
        if x_clean.shape[1] == 1:
            x_clean = x_clean.repeat(1, 3, 1, 1)
        x_preprocessed = self.inception_preprocess(x_clean)
        return self.inception(x_preprocessed)

    @torch.no_grad()
    def step(self, action, order_action):
        # 1. Clamp actions
        action = action.clamp(0, 1)
        order_action = order_action.clamp(0, 1)
        
        old_dones = self.dones.clone()
        active_mask = ~self.dones
        
        # 2. Map continuous order to [1, self.max_order]
        requested_order = torch.round(order_action * (self.max_order - 1)) + 1
        requested_order = requested_order.int()
        
        # 3. Apply Constraints: Cap at remaining budget
        remaining_budget = self.budget - self.steps
        actual_order = torch.min(requested_order, remaining_budget)
        actual_order = torch.max(actual_order, torch.ones_like(actual_order)) # Safety net
        
        # 4. Compute target alpha for this step
        new_alpha = self.alpha + action
        
        # Force alpha to 1.0 if this action exhausts their remaining budget
        new_alpha = torch.where(actual_order >= remaining_budget, torch.ones_like(new_alpha), new_alpha)
        new_alpha = new_alpha.clamp(0, 1)

        if active_mask.any():
            # How many max sub-steps do we need to unroll for the active batch items?
            max_order_in_batch = actual_order[active_mask].max().item()
            
            # Distance per sub-step
            delta_alpha = (new_alpha - self.alpha) / actual_order.float()
            
            x_current = self.x0.clone()
            alpha_current = self.alpha.clone()
            
            # --- EFFICIENT SUB-BATCHING INTEGRATION ---
            for i in range(max_order_in_batch):
                # Identify which items still need to step at iteration `i`
                step_mask = active_mask & (actual_order > i)
                
                if not step_mask.any():
                    break
                    
                # Slice ONLY the tensors that need an update to save compute
                x_sub = x_current[step_mask]
                alpha_sub = alpha_current[step_mask]
                
                # Forward pass on the reduced subset
                d_sub = self.iadb_model(x_sub, alpha_sub)['sample']
                
                # Reshape delta_alpha to [Subset_Size, 1, 1, 1] for broadcasting
                da = delta_alpha[step_mask].view(-1, 1, 1, 1)
                
                # Euler update for the sub-step
                x_current[step_mask] = x_sub + d_sub * da
                alpha_current[step_mask] = alpha_sub + delta_alpha[step_mask]
                
            # Commit the integration results back to self.x0 for active items
            self.x0 = torch.where(active_mask.view(-1, 1, 1, 1), x_current, self.x0)
            
        # 5. Update environment states
        self.alpha = torch.where(active_mask, new_alpha, self.alpha)
        self.steps = torch.where(active_mask, self.steps + actual_order, self.steps)
        self.dones = self.dones | (self.alpha >= 1.0) | (self.steps >= self.budget)

        # --- OPTIMIZED REWARD COMPUTATION ---
        just_done = self.dones & ~old_dones

        if just_done.any():
            x0_finished = self.x0[just_done]
            z_gen_inception = self.get_inception_features(x0_finished)
            
            z_gen_norm = F.normalize(z_gen_inception, p=2.0, dim=1)
            
            sim_matrix = torch.matmul(z_gen_norm, self.z_real_norm.T)
            topk_sim, _ = torch.topk(sim_matrix, self.k, dim=1)
            mean_topk_sim = topk_sim.mean(dim=1)
            
            self.episode_rewards[just_done] = mean_topk_sim

        rewards = self.episode_rewards * self.dones.float()
            
        return {'alpha': self.alpha, 'steps': self.steps, 'x0': self.x0}, rewards, self.dones

    @torch.no_grad()
    def reset(self):
        try:
            x1, _ = next(self.dataloader_iterator)
        except StopIteration:
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
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize dummy components
    dummy_model = DummyIADB().to(device)
    dummy_loader = get_dummy_dataloader(batch_size=8, img_size=64)
    
    # Initialize Environment
    # We use sample_multiplier=2, so batch_size 8 -> 4 acting agents
    env = DiffusionEnvNthOrder(
        dataloader=dummy_loader, 
        iadb_model=dummy_model, 
        device=device, 
        budget=10, 
        sample_multiplier=2
    )
    
    # Reset Environment
    print("\n--- Resetting Environment ---")
    state = env.reset()
    num_agents = state['x0'].shape[0]
    
    print(f"Agents initialized: {num_agents}")
    print(f"Initial Alphas: {state['alpha'].cpu().numpy()}")
    print(f"Initial Steps:  {state['steps'].cpu().numpy()}")
    
    # Run a few dummy steps
    for step_num in range(1, 5):
        print(f"\n=== Env Step {step_num} ===")
        
        # Simulate RL actor generating actions (shape: [num_agents])
        # Random step size between 0.1 and 0.4
        action = torch.rand(num_agents, device=device) * 0.3 + 0.1 
        
        # Random order action mapped in [0, 1] range
        # Some might pick high order (closer to 1), some low (closer to 0)
        order_action = torch.rand(num_agents, device=device) 
        
        print(f"Requested Actions (Delta Alpha): {action.cpu().numpy().round(2)}")
        print(f"Requested Order Actions (0-1):   {order_action.cpu().numpy().round(2)}")
        
        # Take a step in the environment
        state, rewards, dones = env.step(action, order_action)
        
        # Observe the constrained/updated state
        print(f"Current Alphas: {state['alpha'].cpu().numpy().round(3)}")
        print(f"Steps Consumed: {state['steps'].cpu().numpy()}")
        print(f"Dones:          {dones.cpu().numpy()}")
        print(f"Rewards:        {rewards.cpu().numpy().round(3)}")
        
        if dones.all():
            print("\nAll agents have reached their budget or alpha=1.0!")
            break