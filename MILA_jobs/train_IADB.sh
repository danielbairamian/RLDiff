#!/bin/bash

#SBATCH --partition=main                                                                # Ask for unkillable job
#SBATCH --cpus-per-task=4                                                               # Ask for 4 CPUs
#SBATCH --gres=gpu:l40s:1                                                                           # Ask for 1 GPU (any GPU)
#SBATCH --mem=48G                                                                        # Ask for 48 GB of RAM
#SBATCH --time=120:00:00                                                                  # The job will run for 120 hours
#SBATCH -o /network/scratch/d/daniel.bairamian/RLDiff_data/SLURM_DUMP/slurm-%j.out  # Write the log on scratch

# 1. Load the required modules
module --quiet load anaconda/3

# 2. Load your environment
conda activate RLDiff

# 3. Launch your job

python /home/mila/d/daniel.bairamian/RLDiff/train_IADB.py --dataset CIFAR10    --batch_size 64  --num_epochs 1000  --base_dataset_path /network/scratch/d/daniel.bairamian/RLDiff_data/datasets/  --base_logs_path /network/scratch/d/daniel.bairamian/RLDiff_data/logs/diffusion/IADB/  --block_out_channels 64 64 128 128 256 512   --up_block_types UpBlock2D AttnUpBlock2D UpBlock2D UpBlock2D UpBlock2D UpBlock2D    --down_block_types DownBlock2D DownBlock2D DownBlock2D DownBlock2D AttnDownBlock2D DownBlock2D --lr 1e-4 --weight_decay 1e-4 --lr_min 1e-5
python /home/mila/d/daniel.bairamian/RLDiff/train_IADB.py --dataset MNIST      --batch_size 256 --num_epochs 1000  --base_dataset_path /network/scratch/d/daniel.bairamian/RLDiff_data/datasets/  --base_logs_path /network/scratch/d/daniel.bairamian/RLDiff_data/logs/diffusion/IADB/  --block_out_channels 32 32 64 64 128         --up_block_types UpBlock2D AttnUpBlock2D UpBlock2D UpBlock2D UpBlock2D              --down_block_types DownBlock2D DownBlock2D DownBlock2D AttnDownBlock2D DownBlock2D             --lr 1e-4 --weight_decay 1e-4 --lr_min 1e-5
python /home/mila/d/daniel.bairamian/RLDiff/train_IADB.py --dataset CelebAHQ   --batch_size 16  --num_epochs 100   --base_dataset_path /network/scratch/d/daniel.bairamian/RLDiff_data/datasets/  --base_logs_path /network/scratch/d/daniel.bairamian/RLDiff_data/logs/diffusion/IADB/  --block_out_channels 128 128 256 256 512 512 --up_block_types UpBlock2D AttnUpBlock2D UpBlock2D UpBlock2D UpBlock2D UpBlock2D    --down_block_types DownBlock2D DownBlock2D DownBlock2D DownBlock2D AttnDownBlock2D DownBlock2D --lr 1e-4 --weight_decay 1e-4 --lr_min 1e-5