import torch
import torchvision.transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance

import argparse

from utils.dataloaders import CIFAR_dataloader, CelebAHQ_dataloader, MNIST_dataloader, FID_dataloader
from tqdm import tqdm

def calculate_fid(real_dataloader, fake_dataloader, denorm_fn=None, device='cuda', feature=2048):
    
    inception_preprocess = T.Compose([
        T.Resize((299, 299), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    fid = FrechetInceptionDistance(feature=feature, normalize=True).to(device)
    fid.set_dtype(torch.float64)

    def prepare(batch):
        imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        # InceptionV3 expects 3-channel input, so if we have grayscale images, we repeat the channel 3 times
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        return inception_preprocess(imgs)

    for batch in tqdm(real_dataloader, desc="Processing real images"):
        batch[0] = denorm_fn(batch[0])  # Denormalize the batch before feeding to FID 
        batch[0] = torch.clamp(batch[0], 0, 1)  # Ensure the pixel values are in the valid range [0, 1] 
        fid.update(prepare(batch), real=True)

    for batch in tqdm(fake_dataloader, desc="Processing fake images"):
        fid.update(prepare(batch), real=False)

    return fid.compute().item()




if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    parser = argparse.ArgumentParser(description="Generate IADB dataset")
    
    parser.add_argument('--dataset',                    type=str,   default='CIFAR10',       help='Dataset to use: CIFAR10, MNIST, CelebAHQ')
    parser.add_argument('--batch_size',                 type=int,   default=4,               help='Batch size for training')
    parser.add_argument('--budget',                     type=int,   default=10,             help='Maximum number of steps per episode')
    parser.add_argument('--base_dataset_path',          type=str,   default='/Users/danielbairamian/Desktop/RLDiffusion_data/datasets/',        help='Base path for datasets')
    parser.add_argument('--base_FID_dataset_path',      type=str,   default='/Users/danielbairamian/Desktop/RLDiffusion_data/datasets_FID/',        help='Base path for datasets')
    parser.add_argument('--order',                      type=int,   default=2,               help='Order of the method (1=linear, 2=cosine)')
    parser.add_argument('--schedule',                   type=str,   default='RL',        help='Schedule for noise levels: linear or cosine or RL')
    parser.add_argument('--feature_extractor',          type=str,   default="IV3",          help='Feature extractor to use: IV3, DINO')
    parser.add_argument('--diffusion_model',            type=str,   default="IADB",          help='Diffusion model to use: IADB, DDIM')

    args = parser.parse_args()


    if args.dataset == "CIFAR10":
        load_fn = CIFAR_dataloader
    elif args.dataset == "MNIST":
        load_fn = MNIST_dataloader
    elif args.dataset == "CelebAHQ":
        load_fn = CelebAHQ_dataloader
    else:
        raise ValueError("Unsupported dataset. Choose from: CIFAR10, MNIST, CelebAHQ")

    dataset_path     = args.base_dataset_path + args.dataset
    data_log_suffix  = f"{args.dataset}_NFE_{args.budget}_order_{args.order}"
    if args.schedule == 'RL':
        data_log_suffix += f"_{args.feature_extractor}"
    data_log_suffix += f"_schedule_{args.schedule}"

    data_save_path   = args.base_FID_dataset_path + args.diffusion_model + f"/FID_Images/{data_log_suffix}/"

    real_dataloader, info_dict, denorm_fn= load_fn(dataset_path, batch_size=args.batch_size, train=True, drop_last=False)
    fake_dataloader = FID_dataloader(data_save_path, batch_size=args.batch_size)

    print("Calculating FID...")

    fid_score = calculate_fid(real_dataloader, fake_dataloader, denorm_fn=denorm_fn, device=device)
    print(f"FID Score: {fid_score}")