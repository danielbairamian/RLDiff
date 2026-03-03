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


def sampling_strategy(iadb_model, nb_step, order, schedule="linear", ppo_agent=None, env=None):
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
    parser.add_argument('--batch_size',                 type=int,   default=4,               help='Batch size for training')
    parser.add_argument('--budget',                     type=int,   default=10,             help='Maximum number of steps per episode')
    parser.add_argument('--base_dataset_path',          type=str,   default='/Users/danielbairamian/Desktop/RLDiffusion_data/datasets/',        help='Base path for datasets')
    parser.add_argument('--base_FID_dataset_path',      type=str,   default='/Users/danielbairamian/Desktop/RLDiffusion_data/datasets_FID_trajectories/',        help='Base path for datasets')
    parser.add_argument('--base_logs_path',             type=str,   default='/Users/danielbairamian/Desktop/RLDiffusion_data/logs/PPO/IADB/',   help='Base path for logs and checkpoints')
    parser.add_argument('--base_path_diffusion',        type=str,   default='/Users/danielbairamian/Desktop/RLDiffusion_data/logs/diffusion/IADB/', help='Base path for diffusion checkpoints')
    parser.add_argument('--fused_dims',                 type=int,   default=64,              help='Dimension of the fused state-time representation')
    parser.add_argument('--time_encoder_dims',          type=int,   nargs='+', default=[32, 64],       help='Output dims for each layer in the time encoder')
    parser.add_argument('--projection_dims',            type=int,   nargs='+', default=[256, 128],     help='Output dims for each layer in the projection encoder')
    parser.add_argument('--order',                      type=int,   default=2,               help='Order of the method (1=linear, 2=cosine)')
    parser.add_argument('--latent_dim',                 type=int,   default=512,             help='Dimensionality of the image state latent space')
    parser.add_argument('--latent_channels',            type=int,   nargs='+', default=[32, 64, 128, 256], help='Latent channels for the encoder')
    parser.add_argument('--schedule',                   type=str,   default='RL',        help='Schedule for noise levels: linear or cosine or RL')
    parser.add_argument('--start_idx_offset',           type=int,   default=1,               help='Starting index offset for resuming generation to avoid corrupted images' )
    parser.add_argument('--seed',                       type=int,   default=42,              help='Random seed for reproducibility' )

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
    data_log_suffix  = f"{args.dataset}_NFE_{args.budget}_order{args.order}_schedule_{args.schedule}"
    
    data_save_path   = args.base_FID_dataset_path + f"{data_log_suffix}/"
    diffusion_path   = args.base_path_diffusion + f"checkpoints/{args.dataset}/"
    
    ppo_save_path    = args.base_logs_path + f"checkpoints/{ppo_exp_suffix}/"

    os.makedirs(data_save_path, exist_ok=True)

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
        sample_multiplier=1, denorm_fn=denorm_fn, eval_mode=True
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


    num_trajectories = 1024
    num_saved_image_trajectories = 64

    pbar = tqdm(total=num_trajectories, desc="Generating IADB statistics")
    current_idx = 1

    states = None
    alphas = None

    with torch.no_grad():
        while current_idx <= num_trajectories:
            # x1 = sample_iadb_linear_first_order(iadb_model, x0, nb_step=args.budget)
            x1, trajectory_dict = sampling_strategy(iadb_model, nb_step=args.budget, order=args.order, schedule=args.schedule, ppo_agent=ppo_agent, env=env)

            # Extract new data
            new_states = trajectory_dict['states']  # (T_new, B_new, C, H, W)
            new_alphas = trajectory_dict['alphas']  # (T_new, B_new)

            if states is None:
                states = new_states
                alphas = new_alphas
            else:
                # Determine current and new T
                T_existing = states.shape[0]
                T_new = new_states.shape[0]
                T_max = max(T_existing, T_new)

                # Helper to pad the T dimension by repeating the last element
                def pad_trajectory(tensor, target_T):
                    current_T = tensor.shape[0]
                    if current_T < target_T:
                        # Get the last time step: shape (1, B, ...)
                        last_step = tensor[-1:].clone()
                        # Calculate how many repetitions are needed
                        diff = target_T - current_T
                        # Repeat last step and concatenate
                        padding = last_step.repeat(diff, *([1] * (len(tensor.shape) - 1)))
                        return torch.cat([tensor, padding], dim=0)
                    return tensor

                # Pad both to the new T_max
                states = pad_trajectory(states, T_max)
                new_states = pad_trajectory(new_states, T_max)
                
                alphas = pad_trajectory(alphas, T_max)
                new_alphas = pad_trajectory(new_alphas, T_max)

                # Now safe to concatenate along Batch dimension (dim=1)
                states = torch.cat((states, new_states), dim=1)
                alphas = torch.cat((alphas, new_alphas), dim=1)

            current_idx += new_states.shape[1]  # Increment by the batch size of new states
            pbar.update(new_states.shape[1])  # Update progress bar by the batch size of new states
    
    # Save the final concatenated trajectories
    states = states[:, :num_saved_image_trajectories, ...]  # Keep only the first N trajectories for images
    torch.save({'states': states, 'alphas': alphas}, os.path.join(data_save_path, 'iadb_trajectories.pth'))
    print(f"Saved {num_trajectories} trajectories to {os.path.join(data_save_path, 'iadb_trajectories.pth')}")
    print(f"Final states shape: {states.shape}, Final alphas shape: {alphas.shape}")