import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

class DiffusionEnv:
    def __init__(self, dataloader, iadb_model, AE, device, order=1, budget=10, sample_multiplier=4, denorm_fn=None):
        self.dataloader = dataloader
        self.iadb_model = iadb_model
        
        # --- 1. THE FAST STATE ENCODER ---
        self.AE = AE 
        
        self.device = device
        self.dataloader_iterator = iter(self.dataloader)
        self.denorm_fn = denorm_fn
        self.order = order
        self.budget = budget // self.order
        self.sample_multiplier = sample_multiplier 
        
        # --- 2. THE GOLD-STANDARD REWARD (InceptionV3) ---
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
        """Helper to denorm, resize, and extract FID embeddings."""
        x_clean = self.denorm_fn(x) 
        x_preprocessed = self.inception_preprocess(x_clean)
        return self.inception(x_preprocessed)

    @torch.no_grad()
    def step(self, action):
        action = action.clamp(0, 1)
        
        new_alpha = self.alpha + action
        new_alpha = new_alpha.clamp(0, 1)
        new_alpha = torch.where(self.steps >= self.budget - 1, torch.ones_like(new_alpha), new_alpha)
        
        # Track who is already done before this step
        old_dones = self.dones.clone()
        
        if self.order == 1:
            d = self.iadb_model(self.x0, self.alpha)['sample']
        elif self.order == 2:
            mid_alpha = (new_alpha + self.alpha) / 2
            d_1 = self.iadb_model(self.x0, self.alpha)['sample']
            x_mid = self.x0 + d_1 * (mid_alpha - self.alpha).view(-1, 1, 1, 1)
            d = self.iadb_model(x_mid, mid_alpha)['sample']
        
        self.x0 = torch.where(
            self.dones.view(-1, 1, 1, 1), 
            self.x0,  
            self.x0 + d * (new_alpha - self.alpha).view(-1, 1, 1, 1)  
        )
        
        self.alpha = torch.where(self.dones, self.alpha, new_alpha)
        self.steps = self.steps + (~self.dones).int()
        self.dones = self.dones | (self.alpha >= 1.0) | (self.steps >= self.budget)
        
        # --- 1. FAST STATE ENCODING ---
        self.x0_encoded = self.AE.encode(self.x0)

        # --- 2. OPTIMIZED REWARD COMPUTATION ---
        # Who just crossed the finish line exactly on this step?
        just_done = self.dones & ~old_dones

        if just_done.any():
            # Run Inception ONLY on the agents that just finished
            x0_finished = self.x0[just_done]
            z_gen_inception = self.get_inception_features(x0_finished)
            
            z_gen_norm = F.normalize(z_gen_inception, p=2.0, dim=1)
            
            sim_matrix = torch.matmul(z_gen_norm, self.z_real_norm.T)
            topk_sim, _ = torch.topk(sim_matrix, self.k, dim=1)
            mean_topk_sim = topk_sim.mean(dim=1)
            
            # Permanently cache their score
            self.episode_rewards[just_done] = mean_topk_sim

        # Output the persistent cache. 
        # train.py's active_mask will automatically zero out duplicates for the math, 
        # but the final step will retain the correct values for TensorBoard logging.
        rewards = self.episode_rewards * self.dones.float()
            
        return {'x0_encoded': self.x0_encoded, 'alpha': self.alpha, 'steps': self.steps, 'x0': self.x0}, rewards, self.dones

    @torch.no_grad()
    def reset(self):
        try:
            x1, _ = next(self.dataloader_iterator)
        except StopIteration:
            self.dataloader_iterator = iter(self.dataloader)
            x1, _ = next(self.dataloader_iterator)
        
        self.x1 = x1.to(self.device)
        
        # Pre-compute and normalize real Inception features ONCE per batch
        x1_inception = self.get_inception_features(self.x1)
        self.z_real_norm = F.normalize(x1_inception, p=2.0, dim=1)
        
        self.k = 1 # max(1, int(0.01 * self.x1.shape[0]))

        self.x0 = torch.randn_like(self.x1).to(self.device)
        self.x0 = self.x0[:self.x1.shape[0]//self.sample_multiplier] 
        
        self.x0_encoded = self.AE.encode(self.x0)
        
        self.alpha = torch.zeros(self.x0.shape[0], device=self.device) 
        self.dones = torch.zeros(self.x0.shape[0], dtype=torch.bool, device=self.device)
        self.steps = torch.zeros(self.x0.shape[0], dtype=torch.int, device=self.device)
        
        # Persistent reward cache
        self.episode_rewards = torch.zeros(self.x0.shape[0], device=self.device)

        return {'x0_encoded': self.x0_encoded, 'alpha': self.alpha, 'steps': self.steps, 'x0': self.x0}