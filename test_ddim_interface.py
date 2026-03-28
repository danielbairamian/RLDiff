import torch
import os
from torchvision.utils import save_image
from utils.dataloaders import CIFAR_dataloader, CelebAHQ_dataloader, MNIST_dataloader

from src.diffusion.DDIM_inference import (
    load_ddim_wrapper,
    sample_ddim_linear_first_order,
    sample_ddim_cosine_first_order,
    sample_ddim_linear_second_order,
    sample_ddim_cosine_second_order,
)

SAMPLERS = {
    'linear_first':  sample_ddim_linear_first_order,
    'cosine_first':  sample_ddim_cosine_first_order,
    'linear_second': sample_ddim_linear_second_order,
    'cosine_second': sample_ddim_cosine_second_order,
}

if __name__ == '__main__':
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {device}')

    MODEL_ID    = 'google/ddpm-cifar10-32'
    NUM_SAMPLES = 16
    NB_STEP     = 128
    SEED        = 42
    SAVE_PATH   = './outputs/ddim_poc/'

    print(f'Loading: {MODEL_ID}')
    wrapper = load_ddim_wrapper(MODEL_ID, device)

    dataset = "CIFAR10"
    dataset_path = "/Users/danielbairamian/Desktop/RLDiffusion_data/datasets/"

    dataset_path = dataset_path + dataset


    if dataset == "CIFAR10":
        load_fn = CIFAR_dataloader
    elif dataset == "MNIST":
        load_fn = MNIST_dataloader
    elif dataset == "CelebAHQ":
        load_fn = CelebAHQ_dataloader
    else:
        raise ValueError("Unsupported dataset. Choose from: CIFAR10, MNIST, CelebAHQ")

    dataloader, info_dict, denorm_fn = load_fn(dataset_path, batch_size=2, num_workers=1)

    generator = torch.Generator(device=device).manual_seed(SEED)
    x0 = torch.randn(NUM_SAMPLES, 3, 32, 32, device=device, generator=generator)

    os.makedirs(SAVE_PATH, exist_ok=True)

    for name, sampler_fn in SAMPLERS.items():
        print(f'\n[{name}]  nb_step={NB_STEP} ...')
        x1 = sampler_fn(wrapper, x0, nb_step=NB_STEP)
        x1 = (x1 + 1.0) / 2.0  # Rescale from [-1, 1] to [0, 1]
        x1_clamp = x1.cpu().clamp(0.0, 1.0)
        out_path = os.path.join(SAVE_PATH, f'{name}.png')
        save_image(x1_clamp, out_path, nrow=4, normalize=False)
        print(f'  shape : {x1.shape}')
        print(f'  range : [{x1.min():.3f}, {x1.max():.3f}]')
        print(f'  saved : {out_path}')

    print('\nDone. All 4 samplers completed.')