#!/bin/bash

# Define the grids for the loops
# DATASETS=("CIFAR10" "MNIST" "CelebAHQ")
DATASETS=("CIFAR10") # "MNIST")
ORDERS=(1 2)
BUDGETS=(10 20 30 50 100)
SCHEDULES=("RL" "linear" "cosine")
RL_FES=("IV3" "DINO")
DIFFUSION_MODELS=("IADB" "DDIM")


for DS in "${DATASETS[@]}"; do
    # Define dataset-specific batch size (FID calculation can usually handle larger batches)
    case $DS in
        "MNIST")
            B_SIZE=128
            ;;
        "CIFAR10")
            B_SIZE=128
            ;;
        "CelebAHQ")
            B_SIZE=64
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
                            JOB_NAME="FID_${DS}_O${ORD}_B${BUD}_${SCHED}_${FE}"
                            FE_ARG="--feature_extractor $FE"
                        else
                            JOB_NAME="FID_${DS}_O${ORD}_B${BUD}_${SCHED}"
                            FE_ARG=""
                        fi
                    
                        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH -o /network/scratch/d/daniel.bairamian/RLDiff_data/SLURM_DUMP/${JOB_NAME}-%j.out

# Load Environment
module --quiet load anaconda/3
conda activate RLDiff 

# Launch FID calculation script
# Note: Ensure the file name matches your actual python filename (e.g., calculate_fid.py)
python /home/mila/d/daniel.bairamian/RLDiff/FID_calculation.py \\
    --dataset "$DS" \\
    --batch_size $B_SIZE \\
    --budget "$BUD" \\
    --order "$ORD" \\
    --schedule "$SCHED" \\
    $FE_ARG \\
    --diffusion_model "$DM" \\
    --base_dataset_path /network/scratch/d/daniel.bairamian/RLDiff_data/datasets/ \\
    --base_FID_dataset_path /network/scratch/d/daniel.bairamian/RLDiff_data/datasets_FID/
EOF
                    done
                done
            done
        done
    done
done