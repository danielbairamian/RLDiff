import torch
import math
import argparse
import os
from diffusers import DDIMPipeline, DDIMScheduler
from torchvision.utils import save_image


# ---------------------------------------------------------------------------
# DDIMWrapper
# ---------------------------------------------------------------------------

class DDIMWrapper:
    """
    Wraps a HuggingFace DDIMPipeline.

    Why the previous approach (faking IADB directions from DDIM's UNet) failed:
        DDIM's UNet was trained on inputs distributed as:
            x_t = sqrt(α̅_t)·x1 + sqrt(1-α̅_t)·ε          (DDIM forward process)
        The IADB samplers feed it inputs distributed as:
            x_α  = α·x1 + (1-α)·x0                         (IADB interpolation)
        These are different distributions — the UNet produces garbage on IADB inputs,
        and the error explodes over many steps.

    Correct approach:
        Each sample_* function below uses DDIM's own scheduler.step() internally,
        keeping the UNet's inputs on-distribution at all times.
        The outer signature sample_fn(wrapper, x0, nb_step) → x1 is preserved,
        so the rest of the codebase sees the same interface as IADB.

    Schedule / order variants map to DDIM as follows:
        linear_first  — uniform timestep spacing,  1 UNet call per step  (standard DDIM)
        cosine_first  — cosine timestep spacing,   1 UNet call per step
        linear_second — uniform timestep spacing,  2 UNet calls per step (midpoint/RK2)
        cosine_second — cosine timestep spacing,   2 UNet calls per step (midpoint/RK2)
    """

    def __init__(self, pipeline: DDIMPipeline):
        self.unet      = pipeline.unet
        self.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        self.T         = self.scheduler.config.num_train_timesteps

    def to(self, device):
        self.unet.to(device)
        return self

    @property
    def device(self):
        return next(self.unet.parameters()).device

    def predict_eps(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Single UNet forward. t is an integer tensor broadcastable to (N,)."""
        t_batch = t.expand(x.shape[0]).to(self.device)
        return self.unet(x, t_batch).sample


# ---------------------------------------------------------------------------
# Timestep schedule helpers
# ---------------------------------------------------------------------------

def _linear_timesteps(T: int, nb_step: int) -> list[int]:
    """Uniformly spaced timesteps from T-1 → 0, inclusive."""
    return [round(T - 1 - i * (T - 1) / (nb_step - 1)) for i in range(nb_step)]


def _cosine_timesteps(T: int, nb_step: int) -> list[int]:
    """Cosine-spaced timesteps from T-1 → 0.
    Spends more steps near the noise end (high t), fewer near the data end.
    """
    timesteps = []
    for i in range(nb_step):
        cos_val = math.cos((i / (nb_step - 1)) * math.pi / 2)
        t = round(cos_val * (T - 1))
        timesteps.append(t)
    return timesteps


# ---------------------------------------------------------------------------
# Core DDIM step (shared by all samplers)
# ---------------------------------------------------------------------------

def _ddim_step(wrapper: DDIMWrapper,
               x: torch.Tensor,
               t: int,
               t_prev: int,
               eps_pred: torch.Tensor) -> torch.Tensor:
    """
    One DDIM step from timestep t → t_prev using a pre-computed eps_pred.
    Deterministic (eta=0).
    """
    ac = wrapper.scheduler.alphas_cumprod.to(x.device)
    alpha_bar_t      = ac[t]
    alpha_bar_t_prev = ac[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x.device)

    x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
    x0_pred = x0_pred.clamp(-1.0, 1.0)  # static thresholding: x1 lives in [-1,1]
    x_prev  = (torch.sqrt(alpha_bar_t_prev) * x0_pred
               + torch.sqrt(1 - alpha_bar_t_prev) * eps_pred)
    return x_prev


# ---------------------------------------------------------------------------
# IADB-compatible sampler functions
# Same outer signature: sample_fn(model, x0, nb_step, return_trajectory=False)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_ddim_linear_first_order(wrapper: DDIMWrapper, x0: torch.Tensor,
                                   nb_step: int, return_trajectory: bool = False):
    timesteps = _linear_timesteps(wrapper.T, nb_step)
    B = x0.shape[0]
    x0_states = torch.zeros((nb_step + 1, B, *x0.shape[1:]))
    alphas    = torch.zeros(nb_step + 1, B)
    x0_states[0] = x0.cpu()

    x = x0.clone().to(wrapper.device)
    for i, (t, t_prev) in enumerate(zip(timesteps, timesteps[1:] + [-1])):
        t_tensor = torch.tensor(t, device=wrapper.device)
        eps = wrapper.predict_eps(x, t_tensor)
        x   = _ddim_step(wrapper, x, t, t_prev, eps)

        alpha = 1.0 - t / (wrapper.T - 1)
        x0_states[i + 1] = x.cpu()
        alphas[i + 1]     = alpha

    if return_trajectory:
        return x, {'states': x0_states, 'alphas': alphas}
    return x


@torch.no_grad()
def sample_ddim_cosine_first_order(wrapper: DDIMWrapper, x0: torch.Tensor,
                                   nb_step: int, return_trajectory: bool = False):
    timesteps = _cosine_timesteps(wrapper.T, nb_step)
    B = x0.shape[0]
    x0_states = torch.zeros((nb_step + 1, B, *x0.shape[1:]))
    alphas    = torch.zeros(nb_step + 1, B)
    x0_states[0] = x0.cpu()

    x = x0.clone().to(wrapper.device)
    for i, (t, t_prev) in enumerate(zip(timesteps, timesteps[1:] + [-1])):
        t_tensor = torch.tensor(t, device=wrapper.device)
        eps = wrapper.predict_eps(x, t_tensor)
        x   = _ddim_step(wrapper, x, t, t_prev, eps)

        alpha = 1.0 - t / (wrapper.T - 1)
        x0_states[i + 1] = x.cpu()
        alphas[i + 1]     = alpha

    if return_trajectory:
        return x, {'states': x0_states, 'alphas': alphas}
    return x


@torch.no_grad()
def sample_ddim_linear_second_order(wrapper: DDIMWrapper, x0: torch.Tensor,
                                    nb_step: int, return_trajectory: bool = False):
    """
    Second-order midpoint (RK2): evaluate eps at t_mid between t and t_prev,
    then use that eps for the full step. Matches IADB's second-order convention
    and is stable for any step size — unlike Heun which can blow up on large steps.

    Uses nb_step // 2 effective steps (2 UNet calls each) to match IADB convention.
    """
    nb_step   = nb_step // 2
    timesteps = _linear_timesteps(wrapper.T, nb_step)
    B = x0.shape[0]
    x0_states = torch.zeros((nb_step + 1, B, *x0.shape[1:]))
    alphas    = torch.zeros(nb_step + 1, B)
    x0_states[0] = x0.cpu()

    x = x0.clone().to(wrapper.device)
    for i, (t, t_prev) in enumerate(zip(timesteps, timesteps[1:] + [-1])):
        t_mid = max((t + max(t_prev, 0)) // 2, 0)

        t_tensor     = torch.tensor(t,     device=wrapper.device)
        t_mid_tensor = torch.tensor(t_mid, device=wrapper.device)

        # 1st call: eps at t, half-step to t_mid
        eps_t = wrapper.predict_eps(x, t_tensor)
        x_mid = _ddim_step(wrapper, x, t, t_mid, eps_t)

        # 2nd call: eps at t_mid, full step from t to t_prev
        eps_mid = wrapper.predict_eps(x_mid, t_mid_tensor)
        x       = _ddim_step(wrapper, x, t, t_prev, eps_mid)

        alpha = 1.0 - t / (wrapper.T - 1)
        x0_states[i + 1] = x.cpu()
        alphas[i + 1]     = alpha

    if return_trajectory:
        return x, {'states': x0_states, 'alphas': alphas}
    return x


@torch.no_grad()
def sample_ddim_cosine_second_order(wrapper: DDIMWrapper, x0: torch.Tensor,
                                    nb_step: int, return_trajectory: bool = False):
    """Cosine-spaced variant of the midpoint second-order sampler."""
    nb_step   = nb_step // 2
    timesteps = _cosine_timesteps(wrapper.T, nb_step)
    B = x0.shape[0]
    x0_states = torch.zeros((nb_step + 1, B, *x0.shape[1:]))
    alphas    = torch.zeros(nb_step + 1, B)
    x0_states[0] = x0.cpu()

    x = x0.clone().to(wrapper.device)
    for i, (t, t_prev) in enumerate(zip(timesteps, timesteps[1:] + [-1])):
        t_mid = max((t + max(t_prev, 0)) // 2, 0)

        t_tensor     = torch.tensor(t,     device=wrapper.device)
        t_mid_tensor = torch.tensor(t_mid, device=wrapper.device)

        eps_t   = wrapper.predict_eps(x, t_tensor)
        x_mid   = _ddim_step(wrapper, x, t, t_mid, eps_t)
        eps_mid = wrapper.predict_eps(x_mid, t_mid_tensor)
        x       = _ddim_step(wrapper, x, t, t_prev, eps_mid)

        alpha = 1.0 - t / (wrapper.T - 1)
        x0_states[i + 1] = x.cpu()
        alphas[i + 1]     = alpha

    if return_trajectory:
        return x, {'states': x0_states, 'alphas': alphas}
    return x


SAMPLERS = {
    'linear_first':  sample_ddim_linear_first_order,
    'cosine_first':  sample_ddim_cosine_first_order,
    'linear_second': sample_ddim_linear_second_order,
    'cosine_second': sample_ddim_cosine_second_order,
}


# ---------------------------------------------------------------------------
# Loader + save helper
# ---------------------------------------------------------------------------

def load_ddim_wrapper(model_id: str, device: torch.device) -> DDIMWrapper:
    pipeline = DDIMPipeline.from_pretrained(model_id)
    wrapper  = DDIMWrapper(pipeline)
    wrapper.to(device)
    return wrapper


def save_output(images: torch.Tensor, save_path: str, nrow: int = 4):
    os.makedirs(save_path, exist_ok=True)
    images = images.cpu().clamp(0.0, 1.0)
    save_image(images, os.path.join(save_path, 'grid.png'), nrow=nrow, normalize=False)
    for i, img in enumerate(images):
        save_image(img, os.path.join(save_path, f'sample_{i:03d}.png'), normalize=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {device}')

    parser = argparse.ArgumentParser(description='DDIM Inference with IADB-compatible samplers')
    parser.add_argument('--model_id',    type=str,   default='google/ddpm-cifar10-32')
    parser.add_argument('--num_samples', type=int,   default=16)
    parser.add_argument('--nb_step',     type=int,   default=128)
    parser.add_argument('--sampler',     type=str,   default='linear_first',
                        choices=list(SAMPLERS.keys()))
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--save_path',   type=str,   default='./outputs/ddim/')
    args = parser.parse_args()

    print(f'Loading: {args.model_id}')
    wrapper = load_ddim_wrapper(args.model_id, device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    x0 = torch.randn(args.num_samples, 3, 32, 32, device=device, generator=generator)

    sampler_fn = SAMPLERS[args.sampler]
    print(f'Sampling with {args.sampler}, nb_step={args.nb_step} ...')
    x1 = sampler_fn(wrapper, x0, nb_step=args.nb_step)

    print(f'Output shape: {x1.shape},  range: [{x1.min():.3f}, {x1.max():.3f}]')
    save_output(x1, args.save_path)
    print(f'Saved to {args.save_path}')