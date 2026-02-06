import os
import torch 
import torch.nn as nn


from src.latent_encoder.AE import AutoEncoder
from utils.dataloaders import CIFAR_dataloader, CelebAHQ_dataloader, MNIST_dataloader
from utils.image_helpers import tensorboard_image_process

import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim

import argparse

def train_ae(dataloader, ae, device, save_path, logs_path, num_epochs, denorm_fn, lr, weight_decay, lr_min):

    ae.to(device)   
    optimizer = optim.AdamW(ae.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_min)

    logger = SummaryWriter(logs_path)


    for epoch in tqdm(range(num_epochs+1)):
        avg_batch_loss = 0.0
        avg_L1_loss = 0.0
        avg_ssim_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            
            data = data.to(device)
            encoded = ae.encode(data)
            decoded = ae.decode(encoded)


            denorm_data = denorm_fn(data)
            denorm_decoded = denorm_fn(decoded)

            # loss = F.mse_loss(decoded, data)



            L1_loss = F.l1_loss(decoded, data)
            ssim_loss = 1 - ssim(denorm_data, denorm_decoded, data_range=1.0, size_average=True)
            # 3DGS style loss
            lambda_ssim = 0.2
            loss = (1 - lambda_ssim) * L1_loss + lambda_ssim * ssim_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_batch_loss += (loss.item() - avg_batch_loss) / (batch_idx + 1)
            avg_L1_loss += (L1_loss.item() - avg_L1_loss) / (batch_idx + 1)
            avg_ssim_loss += (ssim_loss.item() - avg_ssim_loss) / (batch_idx + 1)

        torch.save(ae.state_dict(), os.path.join(save_path, f'ae.pth'))
        lr_scheduler.step()
        logger.add_scalar('Loss/train', avg_batch_loss, epoch)
        logger.add_scalar('Loss/L1', avg_L1_loss, epoch)
        logger.add_scalar('Loss/SSIM', avg_ssim_loss, epoch)
        logger.add_scalar('Learning Rate', lr_scheduler.get_last_lr()[0], epoch)
        
        if epoch % 50 == 0: # Log images every 50 epochs:
            logger.add_image('Ground Truth', tensorboard_image_process(denorm_data), epoch)
            logger.add_image('Reconstruction', tensorboard_image_process(denorm_decoded), epoch)




if __name__ == "__main__":


    # device

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    parser = argparse.ArgumentParser(description='Train AutoEncoder')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to use: CIFAR10, MNIST, CelebAHQ')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    # parser.add_argument('--base_dataset_path', type=str, default='/home/mila/d/daniel.bairamian/scratch/RLDiff_data/datasets/', help='Base path for datasets')
    # parser.add_argument('--base_logs_path', type=str, default='/home/mila/d/daniel.bairamian/scratch/RLDiff_data/logs/AE/', help='Base path for logs and checkpoints')
    parser.add_argument('--base_dataset_path', type=str, default='/Users/danielbairamian/Desktop/RLDiffusion_data/datasets/', help='Base path for datasets')
    parser.add_argument('--base_logs_path', type=str, default='/Users/danielbairamian/Desktop/RLDiffusion_data/logs/AE/', help='Base path for logs and checkpoints')
    parser.add_argument('--latent_dim', type=int, default=512, help='Dimensionality of the latent space')
    parser.add_argument('--latent_channels', type=int, nargs='+', default=[32, 64, 128, 256], help='List of latent channels for the encoder/decoder')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the optimizer')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='Minimum learning rate for the scheduler')

    args = parser.parse_args()


    if args.dataset == "CIFAR10":
        load_fn = CIFAR_dataloader
    elif args.dataset == "MNIST":
        load_fn = MNIST_dataloader
    elif args.dataset == "CelebAHQ":
        load_fn = CelebAHQ_dataloader
    else:
        raise ValueError("Unsupported dataset. Choose from: CIFAR10, MNIST, CelebAHQ")
    
    dataset_path = args.base_dataset_path + args.dataset
    save_path = args.base_logs_path + f"checkpoints/{args.dataset}/"
    logs_path = args.base_logs_path + f"tensorboard/{args.dataset}/"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)

    dataloader, info_dict, denorm_fn = load_fn(dataset_path, batch_size=args.batch_size)
    autoencoder = AutoEncoder(input_W=info_dict['W'], input_H=info_dict['H'], input_channels=info_dict['C'], latent_channels=args.latent_channels, latent_dim=args.latent_dim)
    train_ae(dataloader, autoencoder, device, save_path, logs_path, args.num_epochs, denorm_fn, args.lr, args.weight_decay, args.lr_min)

