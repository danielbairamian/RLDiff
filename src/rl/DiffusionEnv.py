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
    def step_optimized(self, action):
        action = action.clamp(0, 1)
        
        new_alpha = self.alpha + action
        new_alpha = new_alpha.clamp(0, 1)
        # new_alpha = torch.where(self.steps >= self.budget - 1, torch.ones_like(new_alpha), new_alpha)
        
        # 1. IDENTIFY ACTIVE ENVIRONMENTS
        # Only compute physics for episodes that haven't finished yet
        active_mask = ~self.dones
        
        # --- OPTIMIZATION 1: MASKED DIFFUSION PHYSICS ---
        d = torch.zeros_like(self.x0)
        
        if active_mask.any():
            # Slice out only the active tensors
            x0_act = self.x0[active_mask]
            alpha_act = self.alpha[active_mask]
            
            if self.order == 1:
                d[active_mask] = self.iadb_model(x0_act, alpha_act)['sample']
            elif self.order == 2:
                new_alpha_act = new_alpha[active_mask]
                mid_alpha_act = (new_alpha_act + alpha_act) / 2
                
                d_1 = self.iadb_model(x0_act, alpha_act)['sample']
                x_mid = x0_act + d_1 * (mid_alpha_act - alpha_act).view(-1, 1, 1, 1)
                d[active_mask] = self.iadb_model(x_mid, mid_alpha_act)['sample']

        # Apply physics to x0 (Inactive envs just safely add 0.0)
        self.x0 = torch.where(
            self.dones.view(-1, 1, 1, 1), 
            self.x0,  
            self.x0 + d * (new_alpha - self.alpha).view(-1, 1, 1, 1)  
        )
        
        # Update trackers
        self.alpha = torch.where(self.dones, self.alpha, new_alpha)
        self.steps = self.steps + active_mask.int()
        
        # 2. IDENTIFY NEWLY DONE ENVIRONMENTS
        # Find who crossed the finish line on THIS EXACT STEP
        new_dones = self.dones | (self.alpha >= 1.0) | (self.steps >= self.budget)
        just_done = new_dones & ~self.dones
        self.dones = new_dones

        # --- OPTIMIZATION 2: MASKED LATENT UPDATES ---
        # The PPO agent needs the new state to make its next decision, 
        # but we ONLY need to re-encode the environments that actually moved.
        if active_mask.any():
            self.x0_encoded[active_mask] = self.AE.encode(self.x0[active_mask])

        # --- OPTIMIZATION 3: SPARSE TERMINAL REWARDS ---
        # ONLY calculate LPIPS/Targeting if someone actually finished
        if just_done.any():
            # Extract ONLY the images and latents that just finished
            x0_encoded_fin = self.x0_encoded[just_done]
            x0_fin = self.x0[just_done]
            
            # Target Matching (Pearson Correlation) for the finished subset
            gen_mean = x0_encoded_fin.mean(dim=-1, keepdim=True)
            real_mean = self.x1_encoded.mean(dim=-1, keepdim=True)

            z_gen_centered = F.normalize(x0_encoded_fin - gen_mean, dim=-1)
            z_real_centered = F.normalize(self.x1_encoded - real_mean, dim=-1)

            correlation_matrix = torch.matmul(z_gen_centered, z_real_centered.T) 
            _, correlation_indices = correlation_matrix.max(dim=1) 
            
            # LPIPS Computation for the finished subset
            x0_lpips_fin = self.denorm_fn(x0_fin) * 2 - 1
            x1_lpips = self.denorm_fn(self.x1) * 2 - 1
            nearest_x1_lpips = x1_lpips[correlation_indices]
            
            # .view(-1) is a critical safety catch here. 
            # If exactly 1 agent finishes, .squeeze() would destroy the batch dimension.
            lpips_dist = self.lpips_net(x0_lpips_fin, nearest_x1_lpips).view(-1)

            # Save the calculated reward into the cache
            self.terminal_rewards_cache[just_done] = torch.exp(-lpips_dist)
            
        # Inject the calculated reward into the correct indices
        rewards = self.terminal_rewards_cache * self.dones.float()

        return {'x0_encoded': self.x0_encoded, 'alpha': self.alpha, 'steps': self.steps, 'x0': self.x0}, rewards, self.dones

    @torch.no_grad()
    def step(self, action):
        # action shape: (B,)

        action = action.clamp(0, 1)
        
        
        # update x0 and alpha based on the action taken
        new_alpha = self.alpha + action
        new_alpha = new_alpha.clamp(0, 1)

        # if the agent is about to take the last step, and new alpha is < 1.0, we force it to take the last step to ensure the episode ends
        # new_alpha = torch.where(self.steps >= self.budget - 1, torch.ones_like(new_alpha), new_alpha)
        
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
        lpips_reward = torch.exp(lpips_reward)

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

        # Cache for terminal rewards so we can broadcast them on dead steps without recalculating
        self.terminal_rewards_cache = torch.zeros(self.x0.shape[0], device=self.device)

        return {'x0_encoded': self.x0_encoded, 'alpha': self.alpha, 'steps': self.steps, 'x0': self.x0}


if __name__ == "__main__":
    # --- DUMMY CLASSES ---
    # We must make these strictly deterministic based on inputs so we can compare the runs
    class DummyAE(nn.Module):
        def __init__(self, latent_dim=512):
            super().__init__()
            self.latent_dim = latent_dim
            
        def encode(self, x):
            # Deterministic: Just flatten and slice/pad
            flat = x.view(x.shape[0], -1)
            if flat.shape[1] < self.latent_dim:
                return F.pad(flat, (0, self.latent_dim - flat.shape[1]))
            return flat[:, :self.latent_dim]
            
        def decode(self, z):
            return torch.ones(z.shape[0], 3, 32, 32, device=z.device)

    class DummyIADB(nn.Module):
        def forward(self, x, alpha):
            # Simulate a heavy U-Net forward pass (Batch size dictates compute time)
            B = x.shape[0]
            if B > 0:
                # Do 10 heavy matrix multiplications
                for _ in range(10):
                    _ = torch.matmul(
                        torch.randn(B, 1000, 1000, device=x.device), 
                        torch.randn(B, 1000, 1000, device=x.device)
                    )
            return {'sample': x * 0.1}

    # --- DUMMY DATA ---
    batch_size = 128  # Increased batch size to highlight the speedup
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
        budget=20,
        sample_multiplier=sample_mult,
        denorm_fn=dummy_denorm_fn
    )

    expected_x0_batch = batch_size // sample_mult
    MAX_STEPS = 25

    # 1. PRE-GENERATE STAGGERED ACTIONS
    # We assign different step speeds to different elements in the batch.
    # Some will reach 1.0 in 5 steps, others in 20. This triggers the dynamic masking optimization!
    base_action_rates = torch.linspace(0.04, 0.25, expected_x0_batch, device=device)
    action_sequence = [base_action_rates + torch.rand_like(base_action_rates) * 0.01 for _ in range(MAX_STEPS)]

    print("--- Capturing Initial State ---")
    env.reset()
    init_state = {
        'x1': env.x1.clone(),
        'x1_encoded': env.x1_encoded.clone(),
        'x0': env.x0.clone(),
        'x0_encoded': env.x0_encoded.clone(),
        'alpha': env.alpha.clone(),
        'dones': env.dones.clone(),
        'steps': env.steps.clone()
    }

    # --- RUN 1: STANDARD STEP ---
    print("\n--- Running STANDARD step() ---")
    standard_history = []
    
    # Warmup LPIPS (PyTorch sometimes has overhead on the first execution)
    _ = env.lpips_net(torch.randn(1, 3, 32, 32, device=device), torch.randn(1, 3, 32, 32, device=device))
    
    start_time = time.time()
    for step_idx in range(MAX_STEPS):
        if env.dones.all():
            break
        obs, rewards, dones = env.step(action_sequence[step_idx])
        
        # Save exact copies of the outputs
        standard_history.append({
            'x0': obs['x0'].clone(),
            'x0_encoded': obs['x0_encoded'].clone(),
            'alpha': obs['alpha'].clone(),
            'steps': obs['steps'].clone(),
            'rewards': rewards.clone(),
            'dones': dones.clone()
        })
    standard_time = time.time() - start_time
    print(f"Standard loop finished in {len(standard_history)} steps.")


    # --- RESTORE EXACT INITIAL STATE ---
    env.x1 = init_state['x1'].clone()
    env.x1_encoded = init_state['x1_encoded'].clone()
    env.x0 = init_state['x0'].clone()
    env.x0_encoded = init_state['x0_encoded'].clone()
    env.alpha = init_state['alpha'].clone()
    env.dones = init_state['dones'].clone()
    env.steps = init_state['steps'].clone()


    # --- RUN 2: OPTIMIZED STEP ---
    print("\n--- Running OPTIMIZED step_optimized() ---")
    opt_history = []
    
    start_time = time.time()
    for step_idx in range(MAX_STEPS):
        if env.dones.all():
            break
        obs, rewards, dones = env.step_optimized(action_sequence[step_idx])
        
        # Save exact copies of the outputs
        opt_history.append({
            'x0': obs['x0'].clone(),
            'x0_encoded': obs['x0_encoded'].clone(),
            'alpha': obs['alpha'].clone(),
            'steps': obs['steps'].clone(),
            'rewards': rewards.clone(),
            'dones': dones.clone()
        })
    opt_time = time.time() - start_time
    print(f"Optimized loop finished in {len(opt_history)} steps.")


    # --- VERIFICATION ---
    print("\n--- Verifying Outputs ---")
    assert len(standard_history) == len(opt_history), "Episode lengths mismatch!"
    
    for step_idx, (std, opt) in enumerate(zip(standard_history, opt_history)):
        try:
            torch.testing.assert_close(std['x0'], opt['x0'], msg=f"x0 mismatch at step {step_idx}")
            torch.testing.assert_close(std['x0_encoded'], opt['x0_encoded'], msg=f"x0_encoded mismatch at step {step_idx}")
            torch.testing.assert_close(std['alpha'], opt['alpha'], msg=f"alpha mismatch at step {step_idx}")
            torch.testing.assert_close(std['steps'], opt['steps'], msg=f"steps mismatch at step {step_idx}")
            torch.testing.assert_close(std['rewards'], opt['rewards'], msg=f"rewards mismatch at step {step_idx}")
            torch.testing.assert_close(std['dones'], opt['dones'], msg=f"dones mismatch at step {step_idx}")
        except AssertionError as e:
            print(f"❌ FAILED! {e}")
            exit(1)
            
    print("✅ SUCCESS! All outputs from step() and step_optimized() are exactly mathematically identical.")
    
    print(f"\n--- Performance Results (Batch Size: {expected_x0_batch}) ---")
    print(f"Standard Time:  {standard_time:.4f}s")
    print(f"Optimized Time: {opt_time:.4f}s")
    speedup = (standard_time / opt_time) if opt_time > 0 else float('inf')
    print(f"Speedup:        {speedup:.2f}x faster")