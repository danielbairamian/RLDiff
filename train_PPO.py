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
GAE_LAMBDA = 0.97
PPO_EPSILON = 0.1

@torch.no_grad()
def generate_rollout(env, ppo_agent, deterministic=False):

    obs = env.reset()
    T = env.budget
    B = env.x0.shape[0]

    # pre-allocate tensors for the rollout
    b_states = torch.zeros((T, B, obs['x0_encoded'].shape[1]), device=env.device)
    b_alphas = torch.zeros((T, B), device=env.device)
    b_steps = torch.zeros((T, B), device=env.device)
    b_actions = torch.zeros((T, B, 1), device=env.device)
    b_logprobs = torch.zeros((T, B), device=env.device)
    b_rewards = torch.zeros((T, B), device=env.device)
    b_dones = torch.zeros((T, B), device=env.device)
    b_values = torch.zeros((T, B), device=env.device)
    b_action_means = torch.zeros((T, B, 1), device=env.device) # for debugging and analysis purposes
    b_action_log_stds = torch.zeros((T, B, 1), device=env.device) # for debugging and analysis purposes
    final_x0s = None
    terminal_rewards = None
    final_alphas = None

    last_t = T

    # rollout loop
    for t in range(T):
        # Here you would typically use your policy to get the action and log probability
        # For demonstration, we'll use random actions and dummy log probabilities
        action, value, logprob, action_mean, action_log_std = ppo_agent(obs['x0_encoded'], obs['alpha'], obs['steps'] / env.budget, deterministic=deterministic) # Normalize steps to [0, 1] for the agent
        next_obs, rewards, dones = env.step(action.squeeze(-1)) # Assuming action shape is (B, 1), squeeze to (B,)

        # Store the transition in the rollout buffer
        b_states[t] = obs['x0_encoded']
        b_alphas[t] = obs['alpha']
        b_steps[t] = obs['steps'] / env.budget  # Normalize steps to [0, 1]
        b_actions[t] = action  # Store action without the last dimension
        b_logprobs[t] = logprob
        b_rewards[t] = rewards
        b_dones[t] = dones
        b_values[t] = value
        b_action_means[t] = action_mean # Store action mean without the last dimension
        b_action_log_stds[t] = action_log_std # Store action log std without the last dimension

        obs = next_obs
        if dones.all():
            last_t = t + 1 # +1 to include the final step where all episodes ended
            final_x0s = obs['x0'] # Store the final x0s for all episodes at the end of the rollout
            terminal_rewards = rewards
            final_alphas = obs['alpha']
            break
    
    # Truncate tensors to actual episode length
    b_states = b_states[:last_t]
    b_alphas = b_alphas[:last_t]
    b_steps = b_steps[:last_t]
    b_actions = b_actions[:last_t]
    b_logprobs = b_logprobs[:last_t]
    b_rewards = b_rewards[:last_t]
    b_dones = b_dones[:last_t] 
    b_values = b_values[:last_t]
    b_action_means = b_action_means[:last_t]
    b_action_log_stds = b_action_log_stds[:last_t]

    active_mask = ~torch.cat([torch.zeros((1, B), device=b_dones.device, dtype=torch.bool), b_dones[:-1].bool()], dim=0)
    
    # Zero out rewards and values for dead transitions (post-terminal steps)
    # The env returns terminal reward for dead steps, but we only want reward at the actual terminal step
    b_rewards = b_rewards * active_mask.float()
    b_values = b_values * active_mask.float()

    
    # Compute GAE advantages via reverse iteration (only over active transitions)
    # Formula: A_t = δ_t + (γλ) * A_{t+1} * (1 - done_t)
    # where δ_t = r_t + γ * V_{t+1} * (1 - done_t) - V_t
    b_advantages = torch.zeros_like(b_rewards)
    next_advantage = 0.0
    next_value = 0.0
    
    for t in reversed(range(last_t)):
        is_active = active_mask[t].float()
        not_terminal = 1.0 - b_dones[t].float()
        delta = b_rewards[t] + GAMMA * next_value * not_terminal - b_values[t]
        b_advantages[t] = (delta + GAMMA * GAE_LAMBDA * not_terminal * next_advantage) * is_active
        # Only propagate from active steps
        next_advantage = torch.where(active_mask[t], b_advantages[t], next_advantage)
        next_value = torch.where(active_mask[t], b_values[t], next_value)
    
    b_returns = b_advantages + b_values

    # Extract only active (non-post-terminal) transitions
    rollout = {key: val[active_mask] for key, val in {
        'states': b_states, 'alphas': b_alphas, 'steps': b_steps,
        'actions': b_actions, 'logprobs': b_logprobs, 'advantages': b_advantages,
        'returns': b_returns, 'dones': b_dones, 'values': b_values,
        'action_means': b_action_means, 'action_stds': torch.exp(b_action_log_stds),
    }.items()}
    
    # Episode length = number of steps taken (count active steps per trajectory)
    episode_lengths = active_mask.float().sum(dim=0)
    

    debug_dict = {'final_x0s': final_x0s, 
                  'terminal_rewards': terminal_rewards, 
                  'episode_lengths': episode_lengths, 
                  'final_alphas': final_alphas}

    return rollout, debug_dict

def ppo_buffer_generator(env, ppo_agent, target_steps=256):
    collected_steps = 0
    rollout_chunks = []

    # 1. Collection Phase: Loop until target_steps is reached
    while collected_steps < target_steps:
        # Generate a rollout from the environment
        # Each rollout dictionary already has the 'active_mask' applied
        rollout, debug_dict = generate_rollout(env, ppo_agent)
        
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
    return full_rollout, debug_dict # return the final x0s from the last rollout for logging or analysis purposes after the PPO update

def ppo_update(ppo_agent, minibatch_size=64, full_rollout=None):
    for i in range(0, full_rollout['states'].shape[0], minibatch_size):

        # drop last
        if i + minibatch_size > full_rollout['states'].shape[0]:
            break

        minibatch = {key: full_rollout[key][i:i+minibatch_size] for key in full_rollout.keys()}

        new_logprobs, new_values, entropy, new_action_mean, new_action_log_std = ppo_agent.evaluate_actions(minibatch['states'], minibatch['alphas'], minibatch['steps'], minibatch['actions'])
        kl = new_logprobs - minibatch['logprobs']
        ratio = torch.exp(kl.clip(-10, 10)) # clip for numerical stability
        surrogate1 = ratio * minibatch['advantages']
        surrogate2 = torch.clamp(ratio, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * minibatch['advantages'] 
        
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # value_loss = F.mse_loss(new_values, minibatch['returns'])
        v_pred = new_values
        v_target = minibatch['returns']
        v_old = minibatch['values']

        # 1 . Unclipped loss: MSE
        value_loss_unclipped = (v_pred - v_target)**2

        # 2. Clipped loss
        v_clipped = v_old + (v_pred - v_old).clamp(-PPO_EPSILON, PPO_EPSILON)
        value_loss_clipped = (v_clipped - v_target)**2

        # 3. Maximum of the two losses
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

        # out of bound loss
        act_max_lim = torch.ones_like(new_action_mean)*ppo_agent.act_max    
        act_min_lim = torch.ones_like(new_action_mean)*ppo_agent.act_min

        upper_bound_violation = F.relu(new_action_mean - act_max_lim)
        lower_bound_violation = F.relu(act_min_lim - new_action_mean)
        out_of_bound_loss = (upper_bound_violation**2 + lower_bound_violation**2).mean()


        yield policy_loss, value_loss, entropy.mean(), out_of_bound_loss

def train_PPO(env, ppo_agent, num_epochs=1000, target_steps=256, minibatch_size=64, num_ppo_epochs=4, lr=1e-4, weight_decay=1e-5, entropy_coef=0.01, oob_coef=1.0, denorm_fn=None, logs_path=None, save_path=None):

    print(f"Training PPO for {num_epochs} epochs with target_steps={target_steps}, minibatch_size={minibatch_size}, num_ppo_epochs={num_ppo_epochs}, lr={lr}, weight_decay={weight_decay}, entropy_coef={entropy_coef}, oob_coef={oob_coef}")

    std_params = [ppo_agent.action_log_std]
    base_params = [param for name, param in ppo_agent.named_parameters() if 'action_log_std' not in name]

    std_boost_farcor = 1.0

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': std_params, 'lr': lr * std_boost_farcor, 'weight_decay': 0.0}  # Higher learning rate for log_std
    ])
    logger = SummaryWriter(logs_path)

    for epoch in tqdm(range(num_epochs)):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        epoch_total_loss = 0.0
        epoch_oob_loss = 0.0
        update_count = 0  # Track actual number of minibatch updates

        rollout_buffer, debug_dict = ppo_buffer_generator(env, ppo_agent, target_steps=target_steps)

        for k in range(num_ppo_epochs):
            # randomize the order of the rollout for each epoch to ensure better training stability and data efficiency
            perm = torch.randperm(rollout_buffer['states'].shape[0])
            for key in rollout_buffer.keys():
                rollout_buffer[key] = rollout_buffer[key][perm]

            for policy_loss, value_loss, entropy, out_of_bound_loss in ppo_update(ppo_agent, minibatch_size, full_rollout=rollout_buffer):
                optimizer.zero_grad()
                loss = policy_loss + value_loss - entropy_coef * entropy + oob_coef * out_of_bound_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ppo_agent.parameters(), max_norm=0.5) # gradient clipping for stability
                optimizer.step()

                update_count += 1
                epoch_policy_loss += (policy_loss.item() - epoch_policy_loss) / update_count
                epoch_value_loss += (value_loss.item() - epoch_value_loss) / update_count
                epoch_entropy += (entropy.item() - epoch_entropy) / update_count
                epoch_total_loss += (loss.item() - epoch_total_loss) / update_count
                epoch_oob_loss += (out_of_bound_loss.item() - epoch_oob_loss) / update_count

        test_rollout, debug_dict_test = generate_rollout(env, ppo_agent, deterministic=True)

        if logs_path is not None:
            logger.add_scalar('Loss/Policy', epoch_policy_loss, epoch)
            logger.add_scalar('Loss/Value', epoch_value_loss, epoch)
            logger.add_scalar('Loss/Entropy', epoch_entropy, epoch)
            logger.add_scalar('Loss/Total', epoch_total_loss, epoch)
            logger.add_scalar('Loss/OutOfBound', epoch_oob_loss, epoch)
            final_x0s = denorm_fn(debug_dict['final_x0s']) if denorm_fn is not None else debug_dict['final_x0s']
            final_x0s = tensorboard_image_process(final_x0s)
            logger.add_image('Diffusion Samples', final_x0s, epoch)

            final_x0s_test = denorm_fn(debug_dict_test['final_x0s']) if denorm_fn is not None else debug_dict_test['final_x0s']
            final_x0s_test = tensorboard_image_process(final_x0s_test)
            logger.add_image('Diffusion Samples Test', final_x0s_test, epoch)
            
            for key, value in debug_dict.items():
                if value is None:
                    continue
                else:
                    logger.add_scalar(f'Episode Stats / {key}_mean', value.mean().item(), epoch)
                    logger.add_scalar(f'Episode Stats / {key}_std', value.std().item(), epoch)
                    logger.add_scalar(f'Episode Stats / {key}_max', value.max().item(), epoch)
                    logger.add_scalar(f'Episode Stats / {key}_min', value.min().item(), epoch)
                    logger.add_scalar(f'Episode Stats / {key}_median', value.median().item(), epoch)
            
            for key, value in debug_dict_test.items():
                if value is None:
                    continue
                else:
                    logger.add_scalar(f'Test Episode Stats / {key}_mean', value.mean().item(), epoch)
                    logger.add_scalar(f'Test Episode Stats / {key}_std', value.std().item(), epoch)
                    logger.add_scalar(f'Test Episode Stats / {key}_max', value.max().item(), epoch)
                    logger.add_scalar(f'Test Episode Stats / {key}_min', value.min().item(), epoch)
                    logger.add_scalar(f'Test Episode Stats / {key}_median', value.median().item(), epoch)
            
            for key, value in rollout_buffer.items():
                logger.add_scalar(f'Rollout/{key}_mean', value.mean().item(), epoch)
                logger.add_scalar(f'Rollout/{key}_std', value.std().item(), epoch)
                logger.add_scalar(f'Rollout/{key}_min', value.min().item(), epoch)
                logger.add_scalar(f'Rollout/{key}_max', value.max().item(), epoch)
                logger.add_scalar(f'Rollout/{key}_median', value.median().item(), epoch)
            
            for key, value in test_rollout.items():
                logger.add_scalar(f'Test Rollout/{key}_mean', value.mean().item(), epoch)
                logger.add_scalar(f'Test Rollout/{key}_std', value.std().item(), epoch)
                logger.add_scalar(f'Test Rollout/{key}_min', value.min().item(), epoch)
                logger.add_scalar(f'Test Rollout/{key}_max', value.max().item(), epoch)
                logger.add_scalar(f'Test Rollout/{key}_median', value.median().item(), epoch)



        if save_path is not None:
            torch.save(ppo_agent.state_dict(), os.path.join(save_path, 'ppo_agent.pth'))


if __name__ == "__main__":

    # device

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    parser = argparse.ArgumentParser(description='Train IADB')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to use: CIFAR10, MNIST, CelebAHQ')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--budget', type=int, default=100, help='Maximum number of steps per episode')
    # parser.add_argument('--base_dataset_path', type=str, default='/home/mila/d/daniel.bairamian/scratch/RLDiff_data/datasets/', help='Base path for datasets')
    # parser.add_argument('--base_logs_path', type=str, default='/home/mila/d/daniel.bairamian/scratch/RLDiff_data/logs/diffusion/IADB/', help='Base path for logs and checkpoints')
    parser.add_argument('--base_dataset_path', type=str, default='/Users/danielbairamian/Desktop/RLDiffusion_data/datasets/', help='Base path for datasets')
    parser.add_argument('--base_logs_path', type=str, default='/Users/danielbairamian/Desktop/RLDiffusion_data/logs/PPO/IADB/', help='Base path for logs and checkpoints')
    parser.add_argument('--base_path_diffusion', type=str, default='/Users/danielbairamian/Desktop/RLDiffusion_data/logs/diffusion/IADB/', help='Base path for logs and checkpoints')
    parser.add_argument('--base_AE', type=str, default='/Users/danielbairamian/Desktop/RLDiffusion_data/logs/AE/', help='Base path for logs and checkpoints')
    parser.add_argument('--fused_dims', type=int, default=64, help='Dimension of the fused state-time representation')
    parser.add_argument('--time_encoder_dims', type=int, nargs='+', default=[32, 64], help='List of output dimensions for each layer in the time encoder')
    parser.add_argument('--projection_dims', type=int, nargs='+', default=[256, 128], help='List of output dimensions for each layer in the projection encoder')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--entropy_coef', type=float, default=0.0, help='Entropy coefficient for PPO')
    parser.add_argument('--oob_coef', type=float, default=1.0, help='Coefficient for out-of-bounds action penalty'  )
    parser.add_argument('--target_steps', type=int, default=512, help='Number of steps to collect for each PPO update')
    parser.add_argument('--minibatch_size', type=int, default=256, help='Minibatch size for PPO updates')
    parser.add_argument('--num_ppo_epochs', type=int, default=4, help='Number of PPO epochs to perform for each update')
    parser.add_argument('--sample_multiplier', type=int, default=32, help='How many x1 samples to generate per x0 sample in the environment, to increase batch size for RL training')
    parser.add_argument('--order', type=int, default=1, help='Order of the method (1 for linear first order, 2 for cosine second order)')
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
    logs_path = args.base_logs_path + f"tensorboard/{args.dataset}/"
    save_path = args.base_logs_path + f"checkpoints/{args.dataset}/"
    logs_path_AE = args.base_logs_path + f"tensorboard/{args.dataset}/"
    diffusion_path = args.base_path_diffusion + f"checkpoints/{args.dataset}/"
    ae_path = args.base_AE + f"checkpoints/{args.dataset}/"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)


    autoencoder = torch.load(os.path.join(ae_path, f'ae.pth'), map_location=device).eval() # Load the entire AE class instance and set to eval mode
    iadb_model = torch.load(os.path.join(diffusion_path, f'iadb_model.pth'), map_location=device).eval() # Load the entire IADB model class instance and set to eval mode

    dataloader, info_dict, denorm_fn = load_fn(dataset_path, batch_size=args.batch_size*args.sample_multiplier) # Multiply batch size by sample multiplier to generate more samples for RL training
    env = DiffusionEnv(dataloader, iadb_model, autoencoder, device, order=args.order, budget=args.budget, sample_multiplier=args.sample_multiplier, denorm_fn=denorm_fn) # Pass the denormalization function to the environment so it can log denormalized images to TensorBoard during training

    ppo_agent = PPOAgent(state_dim=autoencoder.latent_dim, 
                         fused_dims=args.fused_dims, 
                         time_encoder_dims=args.time_encoder_dims, 
                         projection_dims=args.projection_dims, 
                         action_dim=1,
                         mean_action_init=(1.0/env.budget)).to(device)

    train_PPO(env, ppo_agent, 
            num_epochs=args.num_epochs, 
            target_steps=args.target_steps, 
            minibatch_size=args.minibatch_size, 
            num_ppo_epochs=args.num_ppo_epochs,
            lr=args.lr, weight_decay=args.weight_decay,
            entropy_coef=args.entropy_coef,
            oob_coef=args.oob_coef,
            denorm_fn=denorm_fn, 
            logs_path=logs_path, save_path=save_path)
