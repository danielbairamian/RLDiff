#!/bin/bash

# Define the grids for the loops
# DATASETS=("CIFAR10" "MNIST" "CelebAHQ")
DATASETS=("CIFAR10" "MNIST")
ORDERS=(1 2)
BUDGETS=(10 20 30 50 100)
SCHEDULES=("RL" "linear" "cosine")

for DS in "${DATASETS[@]}"; do
    # Define dataset-specific hyperparameters
    case $DS in
        "MNIST")
            B_SIZE=128
            F_DIMS=256
            TIME_ENC="64 256 512"
            PROJ_DIMS="512 256 128"
            LAT_DIM=512
            LAT_CHAN="32 64 128 256"
            ;;

        "CIFAR10")
            B_SIZE=128
            F_DIMS=256
            TIME_ENC="64 256 512"
            PROJ_DIMS="512 256 128"
            LAT_DIM=512
            LAT_CHAN="32 64 128 256"
            ;;

        "CelebAHQ")
            B_SIZE=32
            F_DIMS=256
            TIME_ENC="64 256 512"
            PROJ_DIMS="512 256 128"
            LAT_DIM=512
            LAT_CHAN="16 32 64 128 256 512"
            ;;
    esac

    for ORD in "${ORDERS[@]}"; do
        for BUD in "${BUDGETS[@]}"; do
            for SCHED in "${SCHEDULES[@]}"; do
                
                # Updated Job Name to include Schedule (e.g., GEN_MNIST_O1_B10_RL)
                JOB_NAME="STATS_${DS}_O${ORD}_B${BUD}_${SCHED}"
                
                sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=100:00:00
#SBATCH --requeue
#SBATCH -o /network/scratch/d/daniel.bairamian/RLDiff_data/SLURM_DUMP/${JOB_NAME}-%j.out

# Load Environment
module --quiet load anaconda/3
conda activate RLDiff 

# Launch data generation script
python /home/mila/d/daniel.bairamian/RLDiff/IADB_statsgen.py \\
    --dataset "$DS" \\
    --batch_size $B_SIZE \\
    --budget "$BUD" \\
    --order "$ORD" \\
    --schedule "$SCHED" \\
    --fused_dims $F_DIMS \\
    --time_encoder_dims $TIME_ENC \\
    --projection_dims $PROJ_DIMS \\
    --latent_dim $LAT_DIM \\
    --latent_channels $LAT_CHAN \\
    --base_dataset_path /network/scratch/d/daniel.bairamian/RLDiff_data/datasets/ \\
    --base_FID_dataset_path /network/scratch/d/daniel.bairamian/RLDiff_data/datasets_FID_trajectories/ \\
    --base_logs_path /network/scratch/d/daniel.bairamian/RLDiff_data/logs/PPO/IADB/ \\
    --base_path_diffusion /network/scratch/d/daniel.bairamian/RLDiff_data/logs/diffusion/IADB/
EOF

            done
        done
    done
done