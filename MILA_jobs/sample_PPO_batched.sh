#!/bin/bash

DATASETS=("CIFAR10") # "MNIST")
ORDERS=(1 2)
BUDGETS=(10 20 30 50 100)
SCHEDULES=("RL" "linear" "cosine")
RL_FES=("IV3" "DINO")
DIFFUSION_MODELS=("IADB" "DDIM")

for DS in "${DATASETS[@]}"; do
    case $DS in
        "MNIST")
            B_SIZE=128
            F_DIMS=64
            TIME_ENC="64 64" 
            PROJ_DIMS="256 128" 
            LAT_DIM=128
            LAT_CHAN="8 16 32 64" 
            ;;

        "CIFAR10")
            B_SIZE=128 
            F_DIMS=64 
            TIME_ENC="64 64"
            PROJ_DIMS="256 128"
            LAT_DIM=128
            LAT_CHAN="8 16 32 64"
            ;;

        "CelebAHQ")
            B_SIZE=32
            F_DIMS=64 
            TIME_ENC="64 64"
            PROJ_DIMS="256 128"
            LAT_DIM=128
            LAT_CHAN="8 16 32 32 64 64"
            ;;
    esac

    for ORD in "${ORDERS[@]}"; do
        for BUD in "${BUDGETS[@]}"; do
            for DM in "${DIFFUSION_MODELS[@]}"; do
                for SCHED in "${SCHEDULES[@]}"; do

                    if [ "$SCHED" == "RL" ]; then
                        FE_LIST=("${RL_FES[@]}")
                    else
                        FE_LIST=("none")
                    fi

                    for FE in "${FE_LIST[@]}"; do

                        if [ "$SCHED" == "RL" ]; then
                            JOB_NAME="GEN_${DS}_O${ORD}_B${BUD}_${SCHED}_${FE}_DM_${DM}"
                            FE_ARG="--feature_extractor $FE"
                        else
                            JOB_NAME="GEN_${DS}_O${ORD}_B${BUD}_${SCHED}_DM_${DM}"
                            FE_ARG=""
                        fi

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

module --quiet load anaconda/3
conda activate RLDiff 

python /home/mila/d/daniel.bairamian/RLDiff/PPO_datagen.py \\
    --dataset "$DS" \\
    --batch_size $B_SIZE \\
    --budget "$BUD" \\
    --order "$ORD" \\
    --schedule "$SCHED" \\
    $FE_ARG \\
    --fused_dims $F_DIMS \\
    --time_encoder_dims $TIME_ENC \\
    --projection_dims $PROJ_DIMS \\
    --latent_dim $LAT_DIM \\
    --latent_channels $LAT_CHAN \\
    --diffusion_model "$DM" \\
    --base_dataset_path /network/scratch/d/daniel.bairamian/RLDiff_data/datasets/ \\
    --base_FID_dataset_path /network/scratch/d/daniel.bairamian/RLDiff_data/datasets_FID/ \\
    --base_logs_path /network/scratch/d/daniel.bairamian/RLDiff_data/logs/PPO/ \\
    --base_path_diffusion /network/scratch/d/daniel.bairamian/RLDiff_data/logs/diffusion/
EOF
                    done
                done
            done
        done
    done
done