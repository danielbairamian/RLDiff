import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

from src.diffusion.DDIM_inference import DDIMWrapper


class DiffusionEnv:
    """
    Base RL environment for diffusion models.

    Subclasses must implement:
        _init_model(model)          — store the model, set up any schedule caches
        _denoise_step(new_alpha)    — given new_alpha (N,), return x_new (N,C,H,W)
                                      using self.x0 and self.alpha as current state
    """

    def __init__(self, dataloader, model, device,
                 order=1, budget=10, sample_multiplier=4,
                 denorm_fn=None, eval_mode=False, feature_extractor="DINO"):

        self.dataloader          = dataloader
        self.device              = device
        self.dataloader_iterator = iter(self.dataloader)
        self.denorm_fn           = denorm_fn
        self.order               = order
        self.budget              = budget // self.order
        self.sample_multiplier   = sample_multiplier
        self.eval_mode           = eval_mode
        self.feature_extractor   = feature_extractor.upper()

        self._init_model(model)

        if self.feature_extractor == "IV3":
            print("Loading InceptionV3 (Official FID Feature Space)...")
            self.feat_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
            self.feat_model.fc = nn.Identity()
            self.feat_model = self.feat_model.to(device).eval()
            for param in self.feat_model.parameters():
                param.requires_grad = False
            self.preprocess = T.Compose([
                T.Resize((299, 299), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        elif self.feature_extractor == "DINO":
            print("Loading DINOv2 ViT-S/14 (smallest DINOv2 model)...")
            self.feat_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.feat_model = self.feat_model.to(device).eval()
            for param in self.feat_model.parameters():
                param.requires_grad = False
            self.preprocess = T.Compose([
                T.Resize((224, 224), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        else:
            raise ValueError(f"Unknown feature_extractor '{feature_extractor}'. Choose 'IV3' or 'DINO'.")

    def _init_model(self, model):
        raise NotImplementedError

    def _denoise_step(self, new_alpha: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def get_features(self, x):
        x_clean = self.denorm_fn(x)
        if x_clean.shape[1] == 1:
            x_clean = x_clean.repeat(1, 3, 1, 1)
        return self.feat_model(self.preprocess(x_clean))

    @torch.no_grad()
    def step(self, action):
        action    = action.clamp(0, 1)
        new_alpha = (self.alpha + action).clamp(0, 1)
        new_alpha = torch.where(
            self.steps >= self.budget - 1,
            torch.ones_like(new_alpha),
            new_alpha
        )

        old_dones = self.dones.clone()

        x_new = self._denoise_step(new_alpha)

        self.x0    = torch.where(self.dones.view(-1, 1, 1, 1), self.x0, x_new)
        self.alpha = torch.where(self.dones, self.alpha, new_alpha)
        self.steps = self.steps + (~self.dones).int()
        self.dones = self.dones | (self.alpha >= 1.0) | (self.steps >= self.budget)

        just_done = self.dones & ~old_dones
        if just_done.any() and not self.eval_mode:
            z_gen      = self.get_features(self.x0[just_done])
            z_gen_norm = F.normalize(z_gen, p=2.0, dim=1)
            sim_matrix = torch.matmul(z_gen_norm, self.z_real_norm.T)
            topk_sim, _ = torch.topk(sim_matrix, self.k, dim=1)
            self.episode_rewards[just_done] = topk_sim.mean(dim=1) - 1

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

        if not self.eval_mode:
            self.z_real_norm = F.normalize(self.get_features(self.x1), p=2.0, dim=1)
        else:
            self.z_real_norm = None

        self.k  = max(1, int(0.02 * self.x1.shape[0]))
        self.x0 = torch.randn_like(self.x1)[:self.x1.shape[0] // self.sample_multiplier]

        self.alpha           = torch.zeros(self.x0.shape[0], device=self.device)
        self.dones           = torch.zeros(self.x0.shape[0], dtype=torch.bool,  device=self.device)
        self.steps           = torch.zeros(self.x0.shape[0], dtype=torch.int,   device=self.device)
        self.episode_rewards = torch.zeros(self.x0.shape[0], device=self.device)

        return {'alpha': self.alpha, 'steps': self.steps, 'x0': self.x0}


class IADBDiffusionEnv(DiffusionEnv):

    def _init_model(self, model):
        self.iadb_model = model

    def _denoise_step(self, new_alpha: torch.Tensor) -> torch.Tensor:
        if self.order == 1:
            d = self.iadb_model(self.x0, self.alpha)['sample']

        elif self.order == 2:
            mid_alpha = (new_alpha + self.alpha) / 2
            d1    = self.iadb_model(self.x0, self.alpha)['sample']
            x_mid = self.x0 + d1 * (mid_alpha - self.alpha).view(-1, 1, 1, 1)
            d     = self.iadb_model(x_mid, mid_alpha)['sample']

        return self.x0 + d * (new_alpha - self.alpha).view(-1, 1, 1, 1)


class DDIMDiffusionEnv(DiffusionEnv):
    """
    DDIM-specific RL environment with support for first and second-order 
    sampling, progress-guaranteed discretization, and numerical stability.
    """

    def _init_model(self, model):
        """Initializes the DDIM wrapper and caches the alpha schedule."""
        self.ddim_wrapper = model
        self._ac = model.scheduler.alphas_cumprod.to(self.device)
        self.T   = model.T

    def _alpha_to_t(self, alpha: torch.Tensor) -> torch.Tensor:
        """Maps the continuous [0, 1] alpha to discrete [T-1, 0] timesteps."""
        return ((1.0 - alpha) * (self.T - 1)).round().long().clamp(0, self.T - 1)

    def _ddim_step(self, x, t_cur, t_next, eps):
        """
        Performs a single deterministic DDIM step with static thresholding 
        on the predicted clean image (x0) to prevent color-blowing.
        """
        ac_cur  = self._ac[t_cur].view(-1, 1, 1, 1)
        
        # ac_next is 1.0 if t_next is -1, otherwise retrieved from the schedule
        ac_next = torch.where(
            (t_next < 0).view(-1, 1, 1, 1), 
            torch.ones(1, device=self.device).view(-1, 1, 1, 1), 
            self._ac[t_next.clamp(min=0)].view(-1, 1, 1, 1)
        )
        
        # 1. Estimate clean image (x0)
        x0_pred = (x - torch.sqrt(1.0 - ac_cur) * eps) / torch.sqrt(ac_cur)
        
        # 2. Static Thresholding: Clamp to [-1, 1] to maintain numerical stability
        x0_pred = x0_pred.clamp(-1.0, 1.0)
        
        # 3. Compute x_next using the clamped x0 prediction
        return torch.sqrt(ac_next) * x0_pred + torch.sqrt(1.0 - ac_next) * eps

    def _denoise_step(self, new_alpha: torch.Tensor) -> torch.Tensor:
        """
        Executes the denoising step using a mask to skip terminal or 
        redundant computations.
        """
        # Determine active samples that are not yet finished
        active = ~self.dones
        x_out  = self.x0.clone()

        if not active.any():
            return x_out

        # Slice state to active samples only
        x_active = self.x0[active]
        t_cur    = self._alpha_to_t(self.alpha[active])
        
        # Guarantee at least one discrete step forward if not already finished
        t_next   = self._alpha_to_t(new_alpha[active]) - 1

        if self.order == 1:
            # First order: standard DDIM prediction
            eps   = self.ddim_wrapper.predict_eps(x_active, t_cur)
            x_new = self._ddim_step(x_active, t_cur, t_next, eps)

        elif self.order == 2:
            # Second order (Midpoint/PLMS style): uses two UNet calls
            # 1. Proposal step to t_next
            eps_cur = self.ddim_wrapper.predict_eps(x_active, t_cur)
            x_tmp   = self._ddim_step(x_active, t_cur, t_next, eps_cur)
            
            # 2. Correction step at t_next (clamped to 0 for the UNet input)
            eps_next = self.ddim_wrapper.predict_eps(x_tmp, t_next.clamp(min=0))
            eps_avg  = 0.5 * (eps_cur + eps_next)
            x_new    = self._ddim_step(x_active, t_cur, t_next, eps_avg)

        # Update only the active samples in the batch
        x_out[active] = x_new
        return x_out

# class DDIMDiffusionEnv(DiffusionEnv):

#     def _init_model(self, model: DDIMWrapper):
#         self.ddim_wrapper = model
#         self._ac = model.scheduler.alphas_cumprod.to(self.device)
#         self.T   = model.T

#     def _alpha_to_t(self, alpha: torch.Tensor) -> torch.Tensor:
#         return ((1.0 - alpha) * (self.T - 1)).round().long().clamp(0, self.T - 1)

#     def _ddim_step(self, x, t_cur, t_next, eps):
#         ac_cur  = self._ac[t_cur].view(-1, 1, 1, 1)
#         ac_next = self._ac[t_next.clamp(min=0)].view(-1, 1, 1, 1)
#         ac_next = torch.where((t_next < 0).view(-1, 1, 1, 1), torch.ones_like(ac_next), ac_next)
#         x0_pred = (x - torch.sqrt(1.0 - ac_cur) * eps) / torch.sqrt(ac_cur)
#         x0_pred = x0_pred.clamp(-1.0, 1.0)
#         return torch.sqrt(ac_next) * x0_pred + torch.sqrt(1.0 - ac_next) * eps

#     def _denoise_step(self, new_alpha: torch.Tensor) -> torch.Tensor:
#         t_cur    = self._alpha_to_t(self.alpha)
#         t_target = self._alpha_to_t(new_alpha)
#         t_next   = torch.where(new_alpha >= 1.0 - 1e-8, torch.full_like(t_target, -1), t_target)
#         no_step_non_terminal = (t_next == t_cur) & (t_cur > 0)
#         t_next = torch.where(no_step_non_terminal, t_cur - 1, t_next)

#         if self.order == 1:
#             eps = self.ddim_wrapper.predict_eps(self.x0, t_cur)
#             return self._ddim_step(self.x0, t_cur, t_next, eps)

#         elif self.order == 2:
#             eps_cur  = self.ddim_wrapper.predict_eps(self.x0, t_cur)
#             x_tmp    = self._ddim_step(self.x0, t_cur, t_next, eps_cur)
#             eps_next = self.ddim_wrapper.predict_eps(x_tmp, t_next.clamp(min=0))
#             eps_avg  = 0.5 * (eps_cur + eps_next)
#             return self._ddim_step(self.x0, t_cur, t_next, eps_avg)