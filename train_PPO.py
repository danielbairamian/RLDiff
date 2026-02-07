import torch
import torch.nn as nn
import torch.nn.functional as F

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
PPO_EPSILON = 0.2

@torch.no_grad()
def generate_rollout(env, ppo_agent, deterministic=False):

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
        action, value, logprob = ppo_agent(obs['x0_encoded'], obs['alpha'], obs['steps'] / env.budget, deterministic=deterministic) # Normalize steps to [0, 1] for the agent
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


def ppo_update(env, ppo_agent, target_steps=256, minibatch_size=64):
    collected_steps = 0
    rollout_chunks = []

    # 1. Collection Phase: Loop until target_steps is reached
    while collected_steps < target_steps:
        # Generate a rollout from the environment
        # Each rollout dictionary already has the 'active_mask' applied
        rollout = generate_rollout(env, ppo_agent)
        
        chunk_size = rollout['states'].shape[0]
        rollout_chunks.append(rollout)
        collected_steps += chunk_size
        
    # 2. Concatenation Phase
    # Combine all chunks into one large batch for the PPO update
    full_rollout = {
        key: torch.cat([c[key] for c in rollout_chunks], dim=0) 
        for key in rollout_chunks[0].keys()
    }

    # normalize advantages
    full_rollout['advantages'] = (full_rollout['advantages'] - full_rollout['advantages'].mean()) / (full_rollout['advantages'].std() + 1e-8)

    for i in range(0, full_rollout['states'].shape[0], minibatch_size):

        # drop last
        if i + minibatch_size > full_rollout['states'].shape[0]:
            break

        minibatch = {key: full_rollout[key][i:i+minibatch_size] for key in full_rollout.keys()}

    new_logprobs, new_values, entropy = ppo_agent.evaluate_actions(minibatch['states'], minibatch['alphas'], minibatch['steps'], minibatch['actions'])
    kl = new_logprobs - minibatch['logprobs']
    ratio = torch.exp(kl.clip(-10, 10)) # clip for numerical stability
    surrogate1 = ratio * minibatch['advantages']
    surrogate2 = torch.clamp(ratio, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * minibatch['advantages'] 
    
    policy_loss = -torch.min(surrogate1, surrogate2).mean()
    value_loss = F.mse_loss(new_values, minibatch['returns'])

    return policy_loss, value_loss, entropy.mean()

def train_PPO(env, ppo_agent, num_epochs=1000, target_steps=256, minibatch_size=64, lr=1e-4, weight_decay=1e-5, entropy_coef=0.01):

    optimizer = torch.optim.AdamW(ppo_agent.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in tqdm(range(num_epochs)):
        policy_loss, value_loss, entropy = ppo_update(env, ppo_agent, target_steps, minibatch_size)

        optimizer.zero_grad()
        loss = policy_loss + value_loss - entropy_coef * entropy
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}: Policy Loss={policy_loss.item():.4f}, Value Loss={value_loss.item():.4f}, Entropy={entropy.item():.4f}")


if __name__ == "__main__":

    # device

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    parser = argparse.ArgumentParser(description='Train IADB')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to use: CIFAR10, MNIST, CelebAHQ')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--budget', type=int, default=100, help='Maximum number of steps per episode')
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
