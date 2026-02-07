import torch
from diffusers import UNet2DModel
import argparse

from utils.dataloaders import CIFAR_dataloader, CelebAHQ_dataloader, MNIST_dataloader
from utils.image_helpers import tensorboard_image_process

import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.diffusion.iadb_samplers import sample_iadb_linear_first_order, sample_iadb_cosine_first_order, sample_iadb_linear_second_order, sample_iadb_cosine_second_order


def train_iadb(dataloader, iadb_model, device, save_path, logs_path, num_epochs, denorm_fn, lr, weight_decay, lr_min):
    iadb_model.to(device)   
    optimizer = torch.optim.AdamW(iadb_model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_min)
    logger = SummaryWriter(logs_path)

    for epoch in tqdm(range(num_epochs+1)):
        avg_batch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            
            x1 = data.to(device)
            x0 = torch.randn_like(x1).to(device)
            bs = x0.shape[0]

            alpha = torch.rand(bs).to(device)
            x_alpha = alpha.view(-1, 1, 1, 1) * x1 + (1 - alpha).view(-1, 1, 1, 1) * x0

            d = iadb_model(x_alpha, alpha)['sample']
            loss = torch.mean((d - (x1 - x0)) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_batch_loss += (loss.item() - avg_batch_loss) / (batch_idx + 1)
            break

        
        # torch.save(iadb_model.state_dict(), os.path.join(save_path, f'iadb_model.pth'))
        # save the class instance instead of just the state dict to avoid issues with loading the model later without having the exact same code structure
        torch.save(iadb_model, os.path.join(save_path, f'iadb_model.pth'))

        logger.add_scalar('Loss/train', avg_batch_loss, epoch)
        logger.add_scalar('Learning Rate', lr_scheduler.get_last_lr()[0], epoch)
        lr_scheduler.step()

        if epoch % (num_epochs//10) == 0: # Log images every 10% of training:
            with torch.no_grad():
                x1_linear_first_order = sample_iadb_linear_first_order(iadb_model, x0, nb_step=128)
                x_1_cosine_second_order = sample_iadb_cosine_second_order(iadb_model, x0, nb_step=128)
                denorm_x1 = denorm_fn(x1)
                denorm_x1_linear_first_order = denorm_fn(x1_linear_first_order)
                denorm_x1_cosine_second_order = denorm_fn(x_1_cosine_second_order)


                logger.add_image('Ground Truth', tensorboard_image_process(denorm_x1), epoch)
                logger.add_image('Sampled Linear First Order', tensorboard_image_process(denorm_x1_linear_first_order), epoch)
                logger.add_image('Sampled Cosine Second Order', tensorboard_image_process(denorm_x1_cosine_second_order), epoch)




if __name__ == "__main__":

    # device

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    parser = argparse.ArgumentParser(description='Train IADB')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to use: CIFAR10, MNIST, CelebAHQ')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    # parser.add_argument('--base_dataset_path', type=str, default='/home/mila/d/daniel.bairamian/scratch/RLDiff_data/datasets/', help='Base path for datasets')
    # parser.add_argument('--base_logs_path', type=str, default='/home/mila/d/daniel.bairamian/scratch/RLDiff_data/logs/diffusion/IADB/', help='Base path for logs and checkpoints')
    parser.add_argument('--base_dataset_path', type=str, default='/Users/danielbairamian/Desktop/RLDiffusion_data/datasets/', help='Base path for datasets')
    parser.add_argument('--base_logs_path', type=str, default='/Users/danielbairamian/Desktop/RLDiffusion_data/logs/diffusion/IADB/', help='Base path for logs and checkpoints')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='Minimum learning rate for scheduler')
    parser.add_argument('--block_out_channels', type=int, nargs='+', default=[64, 128, 256, 512, 512, 512], help='List of block output channels for the UNet model')
    parser.add_argument('--up_block_types', type=str, nargs='+', default=["UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D",  "UpBlock2D", "UpBlock2D"], help='List of up block types for the UNet model')
    parser.add_argument('--down_block_types', type=str, nargs='+', default=["DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D",  "DownBlock2D"], help='List of down block types for the UNet model')
    parser.add_argument('--layers_per_block', type=int, default=2, help='Number of layers per block in the UNet model')
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

    iadb_model = UNet2DModel(
        block_out_channels=args.block_out_channels,
        out_channels=info_dict["C"],
        in_channels=info_dict["C"],
        up_block_types=tuple(args.up_block_types),
        down_block_types=tuple(args.down_block_types),
        layers_per_block=args.layers_per_block,
        add_attention=True
    )


    train_iadb(dataloader, iadb_model, device, save_path, logs_path, args.num_epochs, denorm_fn, args.lr, args.weight_decay, args.lr_min)
    
