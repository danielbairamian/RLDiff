#!/bin/bash

# Define the datasets to loop through
DATASETS=("CelebAHQ" "CIFAR10" "MNIST")

for DS in "${DATASETS[@]}"; do
    # Define dataset-specific hyperparameters, architectures, and partitions
    case $DS in
        "CelebAHQ")
            PARTITION="main"
            B_SIZE=32
            N_EPOCHS=200
            BLOCK_OUT="32 64 128 256 512 512"
            UP_BLOCKS="UpBlock2D AttnUpBlock2D UpBlock2D UpBlock2D UpBlock2D UpBlock2D"
            DOWN_BLOCKS="DownBlock2D DownBlock2D DownBlock2D DownBlock2D AttnDownBlock2D DownBlock2D"
            ;;

        "CIFAR10")
            PARTITION="long"
            B_SIZE=128
            N_EPOCHS=2000
            BLOCK_OUT="32 64 128 128 256 512"
            UP_BLOCKS="UpBlock2D AttnUpBlock2D UpBlock2D UpBlock2D UpBlock2D UpBlock2D"
            DOWN_BLOCKS="DownBlock2D DownBlock2D DownBlock2D DownBlock2D AttnDownBlock2D DownBlock2D"
            ;;

        "MNIST")
            PARTITION="long"
            B_SIZE=256
            N_EPOCHS=2000
            BLOCK_OUT="32 32 64 64 128"
            UP_BLOCKS="UpBlock2D AttnUpBlock2D UpBlock2D UpBlock2D UpBlock2D"
            DOWN_BLOCKS="DownBlock2D DownBlock2D DownBlock2D AttnDownBlock2D DownBlock2D"
            ;;
    esac

    JOB_NAME="IADB_${DS}"

    # Submit a dedicated job for this specific dataset
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -o /network/scratch/d/daniel.bairamian/RLDiff_data/SLURM_DUMP/${JOB_NAME}-%j.out

# Load Environment
module --quiet load anaconda/3
conda activate RLDiff 

# Launch job with dataset-specific parameters
python /home/mila/d/daniel.bairamian/RLDiff/train_IADB.py \\
    --dataset "$DS" \\
    --batch_size $B_SIZE \\
    --num_epochs $N_EPOCHS \\
    --base_dataset_path /network/scratch/d/daniel.bairamian/RLDiff_data/datasets/ \\
    --base_logs_path /network/scratch/d/daniel.bairamian/RLDiff_data/logs/diffusion/IADB/ \\
    --block_out_channels $BLOCK_OUT \\
    --up_block_types $UP_BLOCKS \\
    --down_block_types $DOWN_BLOCKS \\
    --lr 1e-4 \\
    --weight_decay 1e-4 \\
    --lr_min 1e-5
EOF
done