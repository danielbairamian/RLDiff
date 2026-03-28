import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as vutils


import argparse

from tqdm import tqdm

from utils.dataloaders import CIFAR_dataloader, CelebAHQ_dataloader, MNIST_dataloader
import os

from src.rl.DiffusionEnv import DiffusionEnv
from src.rl.PPOAgentBeta import PPOAgent, VisionEncoder
from src.diffusion.iadb_samplers import sample_iadb_linear_first_order, sample_iadb_cosine_first_order, sample_iadb_linear_second_order, sample_iadb_cosine_second_order

import glob

from train_PPOBeta import generate_rollout


def get_last_index(directory):
    """Finds the highest index existing in the folder to resume."""
    files = glob.glob(os.path.join(directory, f"image_*{IMAGE_EXT}"))
    if not files:
        return 0
    # Extract numbers from filenames like 'image_00042.png'
    indices = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in files]
    return max(indices)

def sampling_strategy_IADB(iadb_model, nb_step, order, schedule="linear", ppo_agent=None, env=None):
    if schedule == "linear":
        env.reset()
        if order == 1:
            x1, trajectory_dict = sample_iadb_linear_first_order(iadb_model, env.x0, nb_step, return_trajectory=True)
            return x1, trajectory_dict
        elif order == 2:
            x1, trajectory_dict = sample_iadb_linear_second_order(iadb_model, env.x0, nb_step, return_trajectory=True)
            return x1, trajectory_dict
    elif schedule == "cosine":
        env.reset()
        if order == 1:
            x1, trajectory_dict = sample_iadb_cosine_first_order(iadb_model, env.x0, nb_step, return_trajectory=True)
            return x1, trajectory_dict
        elif order == 2:
            x1, trajectory_dict = sample_iadb_cosine_second_order(iadb_model, env.x0, nb_step, return_trajectory=True)
            return x1, trajectory_dict
    elif schedule == "RL":
        if ppo_agent is None or env is None:
            raise ValueError("PPO agent and environment must be provided for RL-based sampling")
        # no need to call env.reset(), generate_rollout calls it
        _, debug_dict = generate_rollout(env, ppo_agent, deterministic=False, return_trajectory=True)
        states = debug_dict['states']  # shape (T, B, C, H, W)
        final_states = debug_dict['final_x0s']  # shape (B, C, H, W)
        states = torch.cat((states, final_states.unsqueeze(0)), dim=0)  # shape (T+1, B, C, H, W)

        alphas = debug_dict['alphas']  # shape (T, B)
        final_alphas = debug_dict['final_alphas']  # shape (B,)
        alphas = torch.cat((alphas, final_alphas.unsqueeze(0)), dim=0)  # shape (T+1, B)

        x1 = debug_dict['final_x0s']
        trajectory_dict = {"states": states.cpu(), "alphas": alphas.cpu()}

        return x1, trajectory_dict
    else:
        raise ValueError("Unsupported schedule type. Choose from: linear, cosine, RL")


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    parser = argparse.ArgumentParser(description="Generate IADB dataset")
    
    parser.add_argument('--dataset',                    type=str,   default='CIFAR10',       help='Dataset to use: CIFAR10, MNIST, CelebAHQ')
    parser.add_argument('--batch_size',                 type=int,   default=16,               help='Batch size for training')
    parser.add_argument('--budget',                     type=int,   default=10,             help='Maximum number of steps per episode')
    parser.add_argument('--base_dataset_path',          type=str,   default='/Users/danielbairamian/Desktop/RLDiffusion_data/datasets/',        help='Base path for datasets')
    parser.add_argument('--base_FID_dataset_path',      type=str,   default='/Users/danielbairamian/Desktop/RLDiffusion_data/datasets_FID/',        help='Base path for datasets')
    parser.add_argument('--base_logs_path',             type=str,   default='/Users/danielbairamian/Desktop/RLDiffusion_data/logs/PPO/',   help='Base path for logs and checkpoints')
    parser.add_argument('--base_path_diffusion',        type=str,   default='/Users/danielbairamian/Desktop/RLDiffusion_data/logs/diffusion/', help='Base path for diffusion checkpoints')
    parser.add_argument('--diffusion_model',      type=str,   default='IADB',          help='Diffusion model to use: DDIM, IADB')
    parser.add_argument('--fused_dims',                 type=int,   default=64,              help='Dimension of the fused state-time representation')
    parser.add_argument('--time_encoder_dims',          type=int,   nargs='+', default=[32, 64],       help='Output dims for each layer in the time encoder')
    parser.add_argument('--projection_dims',            type=int,   nargs='+', default=[256, 128],     help='Output dims for each layer in the projection encoder')
    parser.add_argument('--order',                      type=int,   default=2,               help='Order of the method (1=linear, 2=cosine)')
    parser.add_argument('--latent_dim',                 type=int,   default=512,             help='Dimensionality of the image state latent space')
    parser.add_argument('--latent_channels',            type=int,   nargs='+', default=[64, 128, 256], help='Latent channels for the encoder')
    parser.add_argument('--schedule',                   type=str,   default='RL',        help='Schedule for noise levels: linear or cosine or RL')
    parser.add_argument('--start_idx_offset',           type=int,   default=1,               help='Starting index offset for resuming generation to avoid corrupted images' )
    parser.add_argument('--seed',                       type=int,   default=42,              help='Random seed for reproducibility' )
    parser.add_argument('--feature_extractor',          type=str,   default="DINO",          help='Feature extractor to use: IV3, DINO')


    args = parser.parse_args()


    if args.dataset == "CIFAR10":
        load_fn = CIFAR_dataloader
    elif args.dataset == "MNIST":
        load_fn = MNIST_dataloader
    elif args.dataset == "CelebAHQ":
        load_fn = CelebAHQ_dataloader
    else:
        raise ValueError("Unsupported dataset. Choose from: CIFAR10, MNIST, CelebAHQ")

    dataset_path     = args.base_dataset_path + args.dataset
    ppo_exp_suffix   = f"{args.dataset}_NFE_{args.budget}_order_{args.order}"
    # data_log_suffix= f"{args.dataset}_NFE_{args.budget}_order_{args.order}_{args.feature_extractor}_schedule_{args.schedule}"
    data_log_suffix  = ppo_exp_suffix 
    if args.schedule == "RL":
        data_log_suffix += f"_{args.feature_extractor}"
    data_log_suffix += f"_schedule_{args.schedule}"

    data_save_path   = args.base_FID_dataset_path + f"FID_Images/{data_log_suffix}/"
    traj_save_path   = args.base_FID_dataset_path + f"FID_Trajectories/{data_log_suffix}/"
    diffusion_path   = args.base_path_diffusion + f"checkpoints/{args.dataset}/"
    
    ppo_save_path    = args.base_logs_path + f"checkpoints/{ppo_exp_suffix}_{args.feature_extractor}/"

    os.makedirs(data_save_path, exist_ok=True)
    os.makedirs(traj_save_path, exist_ok=True)

    iadb_model = torch.load(os.path.join(diffusion_path, 'iadb_model.pth'), map_location=device).eval()


    # Load dataloader with batch size 2 just to get info dict for initializing the vision encoder, the actual dataloader with the correct batch size will be loaded inside the environment
    dataloader, info_dict, denorm_fn = load_fn(dataset_path, batch_size=args.batch_size, num_workers=1)
    vision_encoder = VisionEncoder(
        input_W=info_dict['W'], input_H=info_dict['H'], input_channels=info_dict['C'],
        latent_channels=args.latent_channels, latent_dim=args.latent_dim
    )

    env = DiffusionEnv(
        dataloader=dataloader, iadb_model=iadb_model, device=device,
        order=args.order, budget=args.budget,
        sample_multiplier=1, denorm_fn=denorm_fn, eval_mode=True, feature_extractor=args.feature_extractor
    )
    
    torch.manual_seed(args.seed)
    env.reset()


    ppo_agent = PPOAgent(
        vision_encoder=vision_encoder,
        state_dim=args.latent_dim,
        fused_dims=args.fused_dims,
        time_encoder_dims=args.time_encoder_dims,
        projection_dims=args.projection_dims,
        action_dim=1,
        mean_action_init=(1.0 / env.budget),
        concentration_init=4.0
    ).to(device)

    checkpoint_file = os.path.join(ppo_save_path, 'ppo_checkpoint.pth')
    
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device)
        ppo_agent.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded PPO checkpoint from {checkpoint_file}")
    else:
        print(f"No PPO checkpoint found at {checkpoint_file}")
    
    print(ppo_agent)


    # TOTAL_SAMPLES = 64 # 50_000
    TOTAL_SAMPLES = 50_000
    IMAGE_EXT = '.png'


    start_idx = get_last_index(data_save_path)
    start_idx = max(0, start_idx - args.start_idx_offset)  # Apply offset to avoid corrupted images
    
    FIRST_TRAJ = start_idx == 0  # Only save the first trajectory if we're starting from the beginning

    print(f"Starting data generation from index {start_idx} (with offset {args.start_idx_offset}) to {TOTAL_SAMPLES}")

    pbar = tqdm(total=TOTAL_SAMPLES, initial=start_idx, desc="Generating IADB dataset")
    current_idx = start_idx
   
    # --- Initialize before the loop ---
    step_counts = torch.zeros(args.budget + 1, dtype=torch.long)  # index = step value
    welford_n   = 0
    welford_mean = 0.0
    welford_M2   = 0.0  # sum of squared deviations

    with torch.no_grad():
        while current_idx < TOTAL_SAMPLES:
            # x1 = sample_iadb_linear_first_order(iadb_model, x0, nb_step=args.budget)
            x1, trajectory_dict = sampling_strategy_IADB(iadb_model, nb_step=args.budget, order=args.order, schedule=args.schedule, ppo_agent=ppo_agent, env=env)

            mask = (trajectory_dict['alphas'] == 1.0)  # [trajectory, batch_size]
            steps = mask.float().argmax(dim=0)          # [batch_size] — first True along trajectory dim

            # 1. Update frequency table (exact median)
            for s in steps:
                step_counts[s.item()] += 1

            # 2. Welford's online mean + variance
            for s in steps:
                x = float(s.item())
                welford_n    += 1
                delta         = x - welford_mean
                welford_mean += delta / welford_n
                delta2        = x - welford_mean
                welford_M2   += delta * delta2


            x1 = denorm_fn(x1)
            x1 = torch.clamp(x1, 0, 1)  # Ensure pixel values are in [0, 1]
            for i in range(x1.shape[0]):
                if current_idx >= TOTAL_SAMPLES:
                    break
                current_idx += 1
                img_filename = f"image_{current_idx:05d}{IMAGE_EXT}"
                img_path = os.path.join(data_save_path, img_filename)
                vutils.save_image(x1[i], img_path)
                pbar.update(1)
            
            if FIRST_TRAJ:
                states = trajectory_dict['states']  # shape (T+1, B, C, H, W)
                alphas = trajectory_dict['alphas']    # shape (T+1, B)

                # denormalize states for saving
                T = states.shape[0]
                for t in range(T):
                    states[t] = denorm_fn(states[t])
                    states[t] = torch.clamp(states[t], 0, 1)  # Ensure pixel values are in [0, 1]   

                FIRST_TRAJ = False
                torch.save({'states': states, 'alphas': alphas}, os.path.join(traj_save_path, 'iadb_trajectories.pth'))
                print(f"Saved first trajectory with {states.shape[1]} states to {os.path.join(traj_save_path, 'iadb_trajectories.pth')}")
                print(f"States shape: {states.shape}, Alphas shape: {alphas.shape}")
                print(f"Example state pixel range: {states.min().item()} to {states.max().item()}")


    # --- After the loop ---
    global_mean = welford_mean
    global_std  = math.sqrt(welford_M2 / welford_n)  # population std
    # global_std = math.sqrt(welford_M2 / (welford_n - 1))  # sample std

    # Exact median from frequency table
    half = welford_n / 2
    cumsum = 0
    global_median = None
    for val, count in enumerate(step_counts):
        cumsum += count.item()
        if cumsum >= half:
            global_median = val
            break

    # Global max and min
    global_max = torch.where(step_counts > 0)[0].max().item()
    global_min = torch.where(step_counts > 0)[0].min().item()


    print(f"Global mean of steps: {global_mean:.4f}")
    print(f"Global std of steps: {global_std:.4f}")
    print(f"Global median of steps: {global_median}")
    print(f"Global max of steps: {global_max}")
    print(f"Global min of steps: {global_min}")
    print(f"Total samples: {welford_n}")