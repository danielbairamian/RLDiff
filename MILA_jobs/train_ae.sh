#!/bin/bash

#SBATCH --partition=unkillable                                                                # Ask for unkillable job
#SBATCH --cpus-per-task=4                                                               # Ask for 4 CPUs
#SBATCH --gres=gpu:l40s:1                                                                           # Ask for 1 GPU (any GPU)
#SBATCH --mem=32G                                                                        # Ask for 48 GB of RAM
#SBATCH --time=48:00:00                                                                  # The job will run for 48 hours
#SBATCH -o /network/scratch/d/daniel.bairamian/RLDiff_data/SLURM_DUMP/slurm-%j.out  # Write the log on scratch

# 1. Load the required modules
module --quiet load anaconda/3

# 2. Load your environment
conda activate RLDiff

# 3. Launch your job

python /home/mila/d/daniel.bairamian/RLDiff/train_ae.py --dataset CIFAR10    --batch_size 256 --num_epochs 5000  --base_dataset_path /network/scratch/d/daniel.bairamian/RLDiff_data/datasets/ --base_logs_path /network/scratch/d/daniel.bairamian/RLDiff_data/logs/AE/ --latent_dim 512  --latent_channels 32 64 128 256         --lr 1e-4 --weight_decay 1e-4 --lr_min 1e-5
python /home/mila/d/daniel.bairamian/RLDiff/train_ae.py --dataset MNIST      --batch_size 256 --num_epochs 5000  --base_dataset_path /network/scratch/d/daniel.bairamian/RLDiff_data/datasets/ --base_logs_path /network/scratch/d/daniel.bairamian/RLDiff_data/logs/AE/ --latent_dim 128  --latent_channels 32 64 128             --lr 1e-4 --weight_decay 1e-4 --lr_min 1e-5
python /home/mila/d/daniel.bairamian/RLDiff/train_ae.py --dataset CelebAHQ   --batch_size 64  --num_epochs 500   --base_dataset_path /network/scratch/d/daniel.bairamian/RLDiff_data/datasets/ --base_logs_path /network/scratch/d/daniel.bairamian/RLDiff_data/logs/AE/ --latent_dim 2048 --latent_channels 32 64 128 256 512     --lr 1e-4 --weight_decay 1e-4 --lr_min 1e-5