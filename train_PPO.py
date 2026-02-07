import torch
import argparse

from utils.dataloaders import CIFAR_dataloader, CelebAHQ_dataloader, MNIST_dataloader
from utils.image_helpers import tensorboard_image_process

import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.rl.DiffusionEnv import DiffusionEnv
from src.rl.PPOAgent import PPOAgent


GAMMA = 1.0
GAE_LAMBDA = 1.0

def generate_rollout(env, ppo_agent):

    obs = env.reset()
    T = env.budget
    B = env.x1.shape[0]

    # pre-allocate tensors for the rollout
    b_states = torch.zeros((T, B, obs['x0_encoded'].shape[1]), device=env.device)
    b_alphas = torch.zeros((T, B), device=env.device)
    b_steps = torch.zeros((T, B), device=env.device)
    b_actions = torch.zeros((T, B), device=env.device)
    b_logprobs = torch.zeros((T, B), device=env.device)
    b_rewards = torch.zeros((T, B), device=env.device)
    b_dones = torch.zeros((T, B), device=env.device)
    b_values = torch.zeros((T, B), device=env.device)

    last_t = T

    # rollout loop
    for t in range(T):
        # Here you would typically use your policy to get the action and log probability
        # For demonstration, we'll use random actions and dummy log probabilities
        action, value, logprob = ppo_agent(obs['x0_encoded'], obs['alpha'], obs['steps'] / env.budget)
        next_obs, rewards, dones = env.step(action.squeeze(-1)) # Assuming action shape is (B, 1), squeeze to (B,)

        # Store the transition in the rollout buffer
        b_states[t] = obs['x0_encoded']
        b_alphas[t] = obs['alpha']
        b_steps[t] = obs['steps'] / env.budget  # Normalize steps to [0, 1]
        b_actions[t] = action.squeeze(-1)  # Store action without the last dimension
        b_logprobs[t] = logprob
        b_rewards[t] = rewards
        b_dones[t] = dones
        b_values[t] = value

        obs = next_obs

        if dones.all():
            last_t = t + 1 # +1 to include the final step where all episodes ended
            break
    
    # Truncate the rollout tensors to the actual length of the episode
    b_states = b_states[:last_t]
    b_alphas = b_alphas[:last_t]
    b_steps = b_steps[:last_t]
    b_actions = b_actions[:last_t]
    b_logprobs = b_logprobs[:last_t]
    b_rewards = b_rewards[:last_t]
    b_dones = b_dones[:last_t] 
    b_values = b_values[:last_t]
   
    shifted_dones = torch.cat([torch.zeros((1, B), device=b_dones.device), b_dones[:-1]], dim=0)
    actual_finish_mask = b_dones.bool() & (~shifted_dones.bool())
    active_mask = ~shifted_dones.bool()


    b_advantages = torch.zeros_like(b_rewards)
    last_gae_lambda = 0.0

    for t in reversed(range(last_t)):
        is_terminal = b_dones[t].float()
        next_non_terminal = 1.0 - is_terminal
        if t == last_t - 1:
            next_values = 0.0
        else:
            next_values = b_values[t + 1]
        
        step_reward = b_rewards[t] * actual_finish_mask[t].float()
        # delta = r + gamma * V(s') - V(s)
        delta = step_reward + GAMMA * next_values * next_non_terminal - b_values[t]
        # advantages[t] = delta + gamma * lambda * next_non_terminal * advantages[t + 1]
        b_advantages[t] = last_gae_lambda = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lambda
    
    b_advantages = b_advantages * active_mask.float()  # Zero out advantages for already done episodes
    b_returns = b_advantages + b_values

    print("Rollout generated with shape:")
    print("Rewards shape: ", b_rewards.shape)
    print("Advantages shape: ", b_advantages.shape)
    print(b_rewards)
    print(b_advantages)
    print(b_returns)
    print(b_actions)
    print(b_alphas)

    rollout = {
        'states': b_states[active_mask], # Only include states from active steps
        'alphas': b_alphas[active_mask],
        'steps': b_steps[active_mask],
        'actions': b_actions[active_mask],
        'logprobs': b_logprobs[active_mask],
        'advantages': b_advantages[active_mask],
        'returns': b_returns[active_mask],
        'dones': b_dones[active_mask],
    }

    return rollout


def train_PPO(env, ppo_agent):
    
    rollout = generate_rollout(env, ppo_agent)

if __name__ == "__main__":

    # device

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    parser = argparse.ArgumentParser(description='Train IADB')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to use: CIFAR10, MNIST, CelebAHQ')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--budget', type=int, default=10, help='Maximum number of steps per episode')
    # parser.add_argument('--base_dataset_path', type=str, default='/home/mila/d/daniel.bairamian/scratch/RLDiff_data/datasets/', help='Base path for datasets')
    # parser.add_argument('--base_logs_path', type=str, default='/home/mila/d/daniel.bairamian/scratch/RLDiff_data/logs/diffusion/IADB/', help='Base path for logs and checkpoints')
    parser.add_argument('--base_dataset_path', type=str, default='/Users/danielbairamian/Desktop/RLDiffusion_data/datasets/', help='Base path for datasets')
    parser.add_argument('--base_logs_path', type=str, default='/Users/danielbairamian/Desktop/RLDiffusion_data/logs/PPO/IADB/', help='Base path for logs and checkpoints')
    parser.add_argument('--base_path_diffusion', type=str, default='/Users/danielbairamian/Desktop/RLDiffusion_data/logs/diffusion/IADB/', help='Base path for logs and checkpoints')
    parser.add_argument('--base_AE', type=str, default='/Users/danielbairamian/Desktop/RLDiffusion_data/logs/AE/', help='Base path for logs and checkpoints')
    parser.add_argument('--fused_dims', type=int, default=256, help='Dimension of the fused state-time representation')
    parser.add_argument('--time_encoder_dims', type=int, nargs='+', default=[32, 64, 128], help='List of output dimensions for each layer in the time encoder')
    parser.add_argument('--projection_dims', type=int, nargs='+', default=[512, 256, 64], help='List of output dimensions for each layer in the projection encoder')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train')
    args = parser.parse_args()


    if args.dataset == "CIFAR10":
        load_fn = CIFAR_dataloader
    elif args.dataset == "MNIST":
        load_fn = MNIST_dataloader
    elif args.dataset == "CelebAHQ":
        load_fn = CelebAHQ_dataloader
    else:
        raise ValueError("Unsupported dataset. Choose from: CIFAR10, MNIST, CelebAHQ")
    
    dataset_path = args.base_dataset_path + args.dataset
    save_path = args.base_logs_path + f"checkpoints/{args.dataset}/"
    logs_path_AE = args.base_logs_path + f"tensorboard/{args.dataset}/"
    diffusion_path = args.base_path_diffusion + f"checkpoints/{args.dataset}/"
    ae_path = args.base_AE + f"checkpoints/{args.dataset}/"


    dataloader, info_dict, denorm_fn = load_fn(dataset_path, batch_size=args.batch_size)

    autoencoder = torch.load(os.path.join(ae_path, f'ae.pth'), map_location=device).eval() # Load the entire AE class instance and set to eval mode
    iadb_model = torch.load(os.path.join(diffusion_path, f'iadb_model.pth'), map_location=device).eval() # Load the entire IADB model class instance and set to eval mode

    dataloader, info_dict, denorm_fn = load_fn(dataset_path, batch_size=args.batch_size)
    env = DiffusionEnv(dataloader, iadb_model, autoencoder, device, budget=args.budget)

    ppo_agent = PPOAgent(state_dim=autoencoder.latent_dim, 
                         fused_dims=args.fused_dims, 
                         time_encoder_dims=args.time_encoder_dims, 
                         projection_dims=args.projection_dims, 
                         action_dim=1).to(device)

    train_PPO(env, ppo_agent)
