#!/bin/bash

# Define the grids for the loops
DATASETS=("CIFAR10" "MNIST") # "CelebAHQ")
ORDERS=(1 2)
BUDGETS=(10 20 30 50 100)
FEATURE_EXTRACTORS=("DINO" "IV3")

for DS in "${DATASETS[@]}"; do
    # Define dataset-specific hyperparameters
    case $DS in
        "MNIST")
            S_MULT=16
            B_SIZE=64
            MB_SIZE=256
            T_STEPS=4096    
            F_DIMS=64
            N_EPOCHS=2000
            TIME_ENC="32 64 256"
            PROJ_DIMS="512 256 128"
            LAT_DIM=512
            LAT_CHAN="8 16 32 64"
            ;;

        "CIFAR10")
            S_MULT=16
            B_SIZE=64
            MB_SIZE=256
            T_STEPS=4096
            F_DIMS=64
            N_EPOCHS=2000
            TIME_ENC="32 64 256"
            PROJ_DIMS="512 256 128"
            LAT_DIM=512
            LAT_CHAN="8 16 32 64"
            ;;

        "CelebAHQ")
            S_MULT=8
            B_SIZE=64
            MB_SIZE=256
            T_STEPS=4096
            F_DIMS=64
            N_EPOCHS=2000
            TIME_ENC="32 64 256"
            PROJ_DIMS="512 256 128"
            LAT_DIM=512
            LAT_CHAN="8 16 32 64 128 256"
            ;;
        
    esac

    for FE in "${FEATURE_EXTRACTORS[@]}"; do
        for ORD in "${ORDERS[@]}"; do
            for BUD in "${BUDGETS[@]}"; do
                
                JOB_NAME="${DS}_${FE}_O${ORD}_B${BUD}"
                
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

# Launch job with dataset-specific parameters
python /home/mila/d/daniel.bairamian/RLDiff/train_PPOBeta.py \\
    --dataset "$DS" \\
    --batch_size $B_SIZE \\
    --sample_multiplier $S_MULT \\
    --budget "$BUD" \\
    --order "$ORD" \\
    --target_steps $T_STEPS \\
    --minibatch_size $MB_SIZE \\
    --num_epochs $N_EPOCHS \\
    --fused_dims $F_DIMS \\
    --time_encoder_dims $TIME_ENC \\
    --projection_dims $PROJ_DIMS \\
    --latent_dim $LAT_DIM \\
    --latent_channels $LAT_CHAN \\
    --feature_extractor "$FE" \\
    --base_dataset_path /network/scratch/d/daniel.bairamian/RLDiff_data/datasets/ \\
    --base_logs_path /network/scratch/d/daniel.bairamian/RLDiff_data/logs/PPO/IADB/ \\
    --base_path_diffusion /network/scratch/d/daniel.bairamian/RLDiff_data/logs/diffusion/IADB/
EOF

            done
        done
    done
done