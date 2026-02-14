#!/bin/bash

#SBATCH --partition=unkillable                                                                # Ask for unkillable job
#SBATCH --cpus-per-task=4                                                               # Ask for 4 CPUs
#SBATCH --gres=gpu:l40s:1                                                                           # Ask for 1 GPU (any GPU)
#SBATCH --mem=32G                                                                        # Ask for 48 GB of RAM
#SBATCH --time=48:00:00                                                                  # The job will run for 120 hours
#SBATCH -o /network/scratch/d/daniel.bairamian/RLDiff_data/SLURM_DUMP/slurm-%j.out  # Write the log on scratch

# 1. Load the required modules
module --quiet load anaconda/3

# 2. Load your environment
conda activate RLDiff

# 3. Launch your job

python /home/mila/d/daniel.bairamian/RLDiff/train_PPO.py --dataset CIFAR10    --batch_size 128  --budget 10 --order 2 --target_steps 4096 --minibatch_size 256 --num_epochs 2000  --fused_dims 64    --time_encoder_dims 64 128      --projection_dims 512 256 128      --base_dataset_path /network/scratch/d/daniel.bairamian/RLDiff_data/datasets/  --base_logs_path /network/scratch/d/daniel.bairamian/RLDiff_data/logs/PPO/IADB/ --base_AE /network/scratch/d/daniel.bairamian/RLDiff_data/logs/AE/ --base_path_diffusion /network/scratch/d/daniel.bairamian/RLDiff_data/logs/diffusion/IADB/ 
python /home/mila/d/daniel.bairamian/RLDiff/train_PPO.py --dataset CIFAR10    --batch_size 128  --budget 10 --order 1 --target_steps 4096 --minibatch_size 256 --num_epochs 2000  --fused_dims 64    --time_encoder_dims 64 128      --projection_dims 512 256 128      --base_dataset_path /network/scratch/d/daniel.bairamian/RLDiff_data/datasets/  --base_logs_path /network/scratch/d/daniel.bairamian/RLDiff_data/logs/PPO/IADB/ --base_AE /network/scratch/d/daniel.bairamian/RLDiff_data/logs/AE/ --base_path_diffusion /network/scratch/d/daniel.bairamian/RLDiff_data/logs/diffusion/IADB/ 
python /home/mila/d/daniel.bairamian/RLDiff/train_PPO.py --dataset MNIST      --batch_size 256  --budget 10 --order 1 --target_steps 4096 --minibatch_size 256 --num_epochs 2000  --fused_dims 64    --time_encoder_dims 64 128      --projection_dims 128 64 32        --base_dataset_path /network/scratch/d/daniel.bairamian/RLDiff_data/datasets/  --base_logs_path /network/scratch/d/daniel.bairamian/RLDiff_data/logs/PPO/IADB/ --base_AE /network/scratch/d/daniel.bairamian/RLDiff_data/logs/AE/ --base_path_diffusion /network/scratch/d/daniel.bairamian/RLDiff_data/logs/diffusion/IADB/  
python /home/mila/d/daniel.bairamian/RLDiff/train_PPO.py --dataset CelebAHQ   --batch_size 32   --budget 10 --order 1 --target_steps 4096 --minibatch_size 64  --num_epochs 2000  --fused_dims 256   --time_encoder_dims 64 128 512  --projection_dims 1024 512 256 128 --base_dataset_path /network/scratch/d/daniel.bairamian/RLDiff_data/datasets/  --base_logs_path /network/scratch/d/daniel.bairamian/RLDiff_data/logs/PPO/IADB/ --base_AE /network/scratch/d/daniel.bairamian/RLDiff_data/logs/AE/ --base_path_diffusion /network/scratch/d/daniel.bairamian/RLDiff_data/logs/diffusion/IADB/  