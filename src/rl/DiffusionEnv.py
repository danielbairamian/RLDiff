import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

class DiffusionEnv:
    def __init__(self, dataloader, iadb_model, device, order=1, budget=10, sample_multiplier=4, denorm_fn=None, eval_mode=False, feature_extractor="DINO"):
        self.dataloader = dataloader
        self.iadb_model = iadb_model
        
        self.device = device
        self.dataloader_iterator = iter(self.dataloader)
        self.denorm_fn = denorm_fn
        self.order = order
        self.budget = budget // self.order
        self.sample_multiplier = sample_multiplier 
        self.eval_mode = eval_mode # eval mode to disable rewards and just get trajectories
        self.feature_extractor = feature_extractor.upper()

        if self.feature_extractor == "IV3":
            # --- InceptionV3 (Official FID Feature Space) ---
            print("Loading InceptionV3 (Official FID Feature Space)...")
            self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Identity()
            self.model = self.model.to(device).eval()

            for param in self.model.parameters():
                param.requires_grad = False

            self.preprocess = T.Compose([
                T.Resize((299, 299), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        elif self.feature_extractor == "DINO":
            # --- DINOv2 ViT-S/14 (smallest available model, 21M params) ---
            print("Loading DINOv2 ViT-S/14 (smallest DINOv2 model)...")
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.model = self.model.to(device).eval()

            for param in self.model.parameters():
                param.requires_grad = False

            # DINOv2 expects 224x224 (multiple of patch size 14), ImageNet stats
            self.preprocess = T.Compose([
                T.Resize((224, 224), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        else:
            raise ValueError(f"Unknown feature_extractor '{feature_extractor}'. Choose 'IV3' or 'DINO'.")

    @torch.no_grad()
    def get_features(self, x):
        """Denorm, resize, and extract embeddings using the chosen feature extractor."""
        x_clean = self.denorm_fn(x)
        # Ensure 3-channel input
        if x_clean.shape[1] == 1:
            x_clean = x_clean.repeat(1, 3, 1, 1)
        x_preprocessed = self.preprocess(x_clean)
        return self.model(x_preprocessed)

    @torch.no_grad()
    def step(self, action):
        action = action.clamp(0, 1)
        
        new_alpha = self.alpha + action
        new_alpha = new_alpha.clamp(0, 1)
        # to force termination <-- vvv uncomment line below vvv -->
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

        # --- OPTIMIZED REWARD COMPUTATION ---
        # Who just crossed the finish line exactly on this step?
        just_done = self.dones & ~old_dones

        if just_done.any() and not self.eval_mode:
            # Run feature extractor ONLY on the agents that just finished
            x0_finished = self.x0[just_done]
            z_gen = self.get_features(x0_finished)
            
            z_gen_norm = F.normalize(z_gen, p=2.0, dim=1)
            
            sim_matrix = torch.matmul(z_gen_norm, self.z_real_norm.T)
            topk_sim, _ = torch.topk(sim_matrix, self.k, dim=1)
            mean_topk_sim = topk_sim.mean(dim=1)
            
            # Permanently cache their score
            self.episode_rewards[just_done] = mean_topk_sim

        # Output the persistent cache. 
        # train.py's active_mask will automatically zero out duplicates for the math, 
        # but the final step will retain the correct values for TensorBoard logging.
        rewards = self.episode_rewards * self.dones.float()
        # rewards = self.episode_rewards * ( self.alpha >= 1.0 ).float()  # Only give reward if they actually reached the end, not just ran out of steps
            
        return {'alpha': self.alpha, 'steps': self.steps, 'x0': self.x0}, rewards, self.dones

    @torch.no_grad()
    def reset(self):
        try:
            x1, _ = next(self.dataloader_iterator)
        except StopIteration:
            self.dataloader_iterator = iter(self.dataloader)
            x1, _ = next(self.dataloader_iterator)
        
        self.x1 = x1.to(self.device)
        
        # Pre-compute and normalize real features ONCE per batch
        if not self.eval_mode:
            x1_features = self.get_features(self.x1)
            self.z_real_norm = F.normalize(x1_features, p=2.0, dim=1)
        else:
            self.z_real_norm = None  # No rewards in eval mode, so no need to compute or store this
        
        # self.k = 1 # max(1, int(0.01 * self.x1.shape[0]))
        self.k = max(1, int(0.01 * self.x1.shape[0])) # Top 1% of the batch 

        self.x0 = torch.randn_like(self.x1).to(self.device)
        self.x0 = self.x0[:self.x1.shape[0]//self.sample_multiplier] 
        
        self.alpha = torch.zeros(self.x0.shape[0], device=self.device) 
        self.dones = torch.zeros(self.x0.shape[0], dtype=torch.bool, device=self.device)
        self.steps = torch.zeros(self.x0.shape[0], dtype=torch.int, device=self.device)
        
        # Persistent reward cache
        self.episode_rewards = torch.zeros(self.x0.shape[0], device=self.device)

        return {'alpha': self.alpha, 'steps': self.steps, 'x0': self.x0}