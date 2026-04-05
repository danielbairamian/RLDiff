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
                 denorm_fn=None, denorm_fn_data=None, eval_mode=False, feature_extractor="DINO"):

        self.dataloader          = dataloader
        self.device              = device
        self.dataloader_iterator = iter(self.dataloader)
        self.denorm_fn           = denorm_fn
        self.denorm_fn_data      = denorm_fn_data  # separate denorm for data if needed
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
    def get_features(self, x, denorm_func):
        x_clean = denorm_func(x)
        x_clean = x_clean.clamp(0.0, 1.0)
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
            z_gen      = self.get_features(self.x0[just_done], self.denorm_fn)
            z_gen_norm = F.normalize(z_gen, p=2.0, dim=1)
            sim_matrix = torch.matmul(z_gen_norm, self.z_real_norm.T)
            topk_sim, _ = torch.topk(sim_matrix, self.k, dim=1)
            self.episode_rewards[just_done] = topk_sim.mean(dim=1)

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
            self.z_real_norm = F.normalize(self.get_features(self.x1, self.denorm_fn_data), p=2.0, dim=1)
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

    def _init_model(self, model: DDIMWrapper):
        self.ddim_wrapper = model
        self._ac = model.scheduler.alphas_cumprod.to(self.device)
        self.T   = model.T
        self.t_state = None   # explicit DDIM timestep state — set in reset()

    def _alpha_to_t(self, alpha: torch.Tensor) -> torch.Tensor:
        return ((1.0 - alpha) * (self.T - 1)).round().long().clamp(0, self.T - 1)

    def _ddim_step(self, x, t_cur, t_next, eps):
        ac_cur  = self._ac[t_cur].view(-1, 1, 1, 1)
        ac_next = self._ac[t_next.clamp(min=0)].view(-1, 1, 1, 1)
        ac_next = torch.where((t_next < 0).view(-1, 1, 1, 1), torch.ones_like(ac_next), ac_next)
        x0_pred = (x - torch.sqrt(1.0 - ac_cur) * eps) / torch.sqrt(ac_cur)
        x0_pred = x0_pred.clamp(-1.0, 1.0)  # clip to valid range for stability
        return torch.sqrt(ac_next) * x0_pred + torch.sqrt(1.0 - ac_next) * eps

    def _denoise_step(self, new_alpha: torch.Tensor) -> torch.Tensor:
        # Gate on active (non-terminated) samples.
        # For IADB, zero action → delta_alpha=0 → x unchanged implicitly.
        # For DDIM, _ddim_step always moves x, so we must gate explicitly AND
        # guard against tiny actions that don't produce a discrete timestep change.
        active = ~self.dones
        x_out  = self.x0.clone()

        if not active.any():
            return x_out

        x_a    = self.x0[active]
        t_cur  = self.t_state[active]                        # ground-truth DDIM position
        t_next = self._alpha_to_t(new_alpha[active]) - 1    # target after action

        # If the action is so small that it doesn't advance a single discrete
        # timestep, skip the UNet entirely — keeps x and t_state in sync.
        # This mirrors IADB where delta_alpha=0 is a no-op.
        makes_progress = t_next < t_cur                      # t decreases = progress
        x_new  = x_a.clone()

        if not makes_progress.any():
            # No sample moves — write back unchanged and return early
            x_out[active] = x_new
            return x_out

        # Only run the UNet on active samples that actually make progress
        prog   = makes_progress
        x_p    = x_a[prog]
        tc_p   = t_cur[prog]
        tn_p   = t_next[prog]

        if self.order == 1:
            eps              = self.ddim_wrapper.predict_eps(x_p, tc_p)
            x_new[prog]      = self._ddim_step(x_p, tc_p, tn_p, eps)

        elif self.order == 2:
            # Midpoint (RK2): evaluate eps at t_mid (halfway) for stability
            # with variable RL step sizes. Falls back to first-order when the
            # gap is only 1 (t_mid would collapse onto an endpoint).
            gap     = tc_p - tn_p.clamp(min=0)
            use_mid = gap >= 2

            x_step = torch.empty_like(x_p)

            if use_mid.any():
                x_m    = x_p[use_mid];  tc_m = tc_p[use_mid];  tn_m = tn_p[use_mid]
                tmid_m = ((tc_m.float() + tn_m.clamp(min=0).float()) / 2).round().long().clamp(min=0)

                eps_cur            = self.ddim_wrapper.predict_eps(x_m, tc_m)
                x_mid              = self._ddim_step(x_m, tc_m, tmid_m, eps_cur)
                eps_mid            = self.ddim_wrapper.predict_eps(x_mid, tmid_m)
                x_step[use_mid]    = self._ddim_step(x_m, tc_m, tn_m, eps_mid)

            if (~use_mid).any():
                x_f    = x_p[~use_mid];  tc_f = tc_p[~use_mid];  tn_f = tn_p[~use_mid]
                eps                = self.ddim_wrapper.predict_eps(x_f, tc_f)
                x_step[~use_mid]   = self._ddim_step(x_f, tc_f, tn_f, eps)

            x_new[prog] = x_step

        x_out[active] = x_new

        # Update t_state for samples that made progress — this is the ground truth
        # DDIM position and must stay in sync with the actual noise level of x.
        # Using alpha alone to reconstruct t would drift when actions are small.
        new_t = self.t_state.clone()
        active_indices        = active.nonzero(as_tuple=True)[0]
        prog_indices          = active_indices[prog]
        new_t[prog_indices]   = tn_p.clamp(min=-1)   # -1 = terminal (fully clean)
        self.t_state          = new_t

        return x_out

    def reset(self):
        state = super().reset()
        # Initialise t_state to T-1 (pure noise) for all samples
        self.t_state = torch.full((self.x0.shape[0],), self.T - 1,
                                  dtype=torch.long, device=self.device)
        return state