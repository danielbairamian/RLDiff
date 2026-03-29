import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

from utils.dataloaders import CIFAR_dataloader, CelebAHQ_dataloader, MNIST_dataloader
from utils.image_helpers import tensorboard_image_process

import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.rl.DiffusionRLEnv import IADBDiffusionEnv, DDIMDiffusionEnv
from src.rl.PPOAgentBeta import PPOAgent, VisionEncoder

from src.diffusion.DDIM_inference import load_ddim_wrapper

@torch.no_grad()
def generate_rollout(env, ppo_agent, deterministic=False, return_trajectory=False, gamma=1.0, gae_lambda=1.0):

    obs = env.reset()
    T = env.budget
    B = env.x0.shape[0]

    # Pre-allocate rollout buffers on CPU to avoid accumulating GPU memory
    b_states         = torch.zeros((T, B, *obs['x0'].shape[1:]))  # CPU
    b_alphas         = torch.zeros((T, B))
    b_steps          = torch.zeros((T, B))
    b_actions        = torch.zeros((T, B, 1))
    b_logprobs       = torch.zeros((T, B))
    b_rewards        = torch.zeros((T, B))
    b_dones          = torch.zeros((T, B))
    b_values         = torch.zeros((T, B))
    b_action_means   = torch.zeros((T, B, 1))   # α / (α+β) — interpretable mean
    b_concentrations_a = torch.zeros((T, B, 1))   # α     — interpretable confidence
    b_concentrations_b = torch.zeros((T, B, 1))   # β     — interpretable confidence

    final_x0s        = None
    terminal_rewards = None
    final_alphas     = None
    last_t           = T

    # Rollout loop — agent forward on GPU, store results on CPU immediately
    for t in range(T):
        action, value, logprob, conc_alpha, conc_beta = ppo_agent(
            obs['x0'], obs['alpha'], obs['steps'] / env.budget, deterministic=deterministic
        )
        next_obs, rewards, dones = env.step(action.squeeze(-1))

        b_states[t]          = obs['x0'].cpu()
        b_alphas[t]          = obs['alpha'].cpu()
        b_steps[t]           = (obs['steps'] / env.budget).cpu()
        b_actions[t]         = action.cpu()
        b_logprobs[t]        = logprob.cpu()
        b_rewards[t]         = rewards.cpu()
        b_dones[t]           = dones.cpu()
        b_values[t]          = value.cpu()
        b_action_means[t]    = (conc_alpha / (conc_alpha + conc_beta)).cpu()
        b_concentrations_a[t]  = conc_alpha.cpu()
        b_concentrations_b[t]  = conc_beta.cpu()

        obs = next_obs
        if dones.all():
            last_t           = t + 1
            final_x0s        = obs['x0'].cpu()
            terminal_rewards = rewards.cpu()
            final_alphas     = obs['alpha'].cpu()
            break

    # Truncate to actual episode length
    b_states         = b_states[:last_t]
    b_alphas         = b_alphas[:last_t]
    b_steps          = b_steps[:last_t]
    b_actions        = b_actions[:last_t]
    b_logprobs       = b_logprobs[:last_t]
    b_rewards        = b_rewards[:last_t]
    b_dones          = b_dones[:last_t]
    b_values         = b_values[:last_t]
    b_action_means   = b_action_means[:last_t]
    b_concentrations_a = b_concentrations_a[:last_t]
    b_concentrations_b = b_concentrations_b[:last_t]

    # All mask / GAE computation on CPU
    active_mask = ~torch.cat([
        torch.zeros((1, B), dtype=torch.bool),
        b_dones[:-1].bool()
    ], dim=0)

    b_rewards = b_rewards * active_mask.float()
    b_values  = b_values  * active_mask.float()

    b_advantages  = torch.zeros_like(b_rewards)
    next_advantage = 0.0
    next_value     = 0.0

    for t in reversed(range(last_t)):
        is_active    = active_mask[t].float()
        not_terminal = 1.0 - b_dones[t].float()
        delta        = b_rewards[t] + gamma * next_value * not_terminal - b_values[t]
        b_advantages[t] = (delta + gamma * gae_lambda * not_terminal * next_advantage) * is_active
        next_advantage  = torch.where(active_mask[t], b_advantages[t], next_advantage)
        next_value      = torch.where(active_mask[t], b_values[t],      next_value)

    b_returns = b_advantages + b_values

    rollout = {key: val[active_mask] for key, val in {
        'states':         b_states,
        'alphas':         b_alphas,
        'steps':          b_steps,
        'actions':        b_actions,
        'logprobs':       b_logprobs,
        'advantages':     b_advantages,
        'returns':        b_returns,
        'dones':          b_dones,
        'values':         b_values,
        'action_means':   b_action_means,
        'concentrations_a': b_concentrations_a,
        'concentrations_b': b_concentrations_b,
        'concentrations' : b_concentrations_a + b_concentrations_b,
        'beta_mean': b_concentrations_a  / (b_concentrations_a + b_concentrations_b + 1e-8),
    }.items()}

    episode_lengths = active_mask.float().sum(dim=0)

    debug_dict = {
        'final_x0s':        final_x0s,             # CPU    
        'terminal_rewards': terminal_rewards,      # CPU
        'episode_lengths':  episode_lengths,       # CPU
        'final_alphas':     final_alphas,          # CPU
    }

    if return_trajectory:
        debug_dict['states'] = b_states
        debug_dict['alphas'] = b_alphas

    return rollout, debug_dict


def ppo_buffer_generator(env, ppo_agent, target_steps=256, gamma=1.0, gae_lambda=1.0):
    collected_steps = 0
    rollout_chunks  = []

    while collected_steps < target_steps:
        rollout, debug_dict = generate_rollout(env, ppo_agent, gamma=gamma, gae_lambda=gae_lambda)

        chunk_size = rollout['states'].shape[0]
        rollout_chunks.append(rollout)
        collected_steps += chunk_size

    full_rollout = {
        key: torch.cat([c[key] for c in rollout_chunks], dim=0)
        for key in rollout_chunks[0].keys()
    }

    # full_rollout['advantages'] = (
    #     (full_rollout['advantages'] - full_rollout['advantages'].mean())
    #     / (full_rollout['advantages'].std() + 1e-4)
    # )

    full_rollout['advantages'] = (
        (full_rollout['advantages'] - full_rollout['advantages'].mean())
        / (full_rollout['advantages'].std() + 1.0)
    )


    return full_rollout, debug_dict


def ppo_update(ppo_agent, device, minibatch_size=64, full_rollout=None, ppo_clip_epsilon=0.1):
    for i in range(0, full_rollout['states'].shape[0], minibatch_size):

        if i + minibatch_size > full_rollout['states'].shape[0]:
            break

        # Slice on CPU, then transfer the minibatch to GPU in one go
        minibatch = {
            key: full_rollout[key][i:i+minibatch_size].to(device)
            for key in full_rollout.keys()
        }

        new_logprobs, new_values, entropy, conc_alpha, conc_beta, net_dict = ppo_agent.evaluate_actions(
            minibatch['states'], minibatch['alphas'], minibatch['steps'], minibatch['actions']
        )
        kl        = new_logprobs - minibatch['logprobs']
        approx_kl = minibatch['logprobs'] - new_logprobs
        ratio     = torch.exp(kl.clip(-10, 10))
        surrogate1 = ratio * minibatch['advantages']
        surrogate2 = torch.clamp(ratio, 1 - ppo_clip_epsilon, 1 + ppo_clip_epsilon) * minibatch['advantages']

        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        v_pred   = new_values
        v_target = minibatch['returns']
        v_old    = minibatch['values']

        value_loss_unclipped = (v_pred - v_target) ** 2
        v_clipped            = v_old + (v_pred - v_old).clamp(-ppo_clip_epsilon, ppo_clip_epsilon)
        value_loss_clipped   = (v_clipped - v_target) ** 2
        value_loss           = torch.max(value_loss_unclipped, value_loss_clipped).mean()


        # entropy surrogate loss directly on net_dict's kappa
        # - entropy_coef * entropy
        concentration_kappa = net_dict['kappa']  # higher kappa = more confident = lower entropy


        yield policy_loss, value_loss, entropy.mean(), approx_kl.mean(), concentration_kappa.mean()


def train_PPO(env, ppo_agent, device, ppo_args, denorm_fn=None, logs_path=None, save_path=None):

    optimizer = torch.optim.AdamW(ppo_agent.parameters(), lr=ppo_args['lr'], weight_decay=ppo_args['weight_decay'])

    logger = SummaryWriter(logs_path)

    start_epoch    = 0
    checkpoint_file = os.path.join(save_path, 'ppo_checkpoint.pth')
    if os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint: {checkpoint_file}")
        try:
            checkpoint = torch.load(checkpoint_file, map_location=device)
            ppo_agent.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Successfully resumed at epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
    
    best_avg_reward = -float('inf')
    save_best = False

    for epoch in tqdm(range(start_epoch, ppo_args['num_epochs'] + 1)):
        epoch_policy_loss = 0.0
        epoch_value_loss  = 0.0
        epoch_entropy     = 0.0
        epoch_total_loss  = 0.0
        epoch_kl          = 0.0
        epoch_kappa       = 0.0
        update_count      = 0

        KL_terminated = False

        rollout_buffer, debug_dict = ppo_buffer_generator(env, ppo_agent, target_steps=ppo_args['target_steps'], gamma=ppo_args['gamma'], gae_lambda=ppo_args['gae_lambda'])

        for k in range(ppo_args['num_ppo_epochs']):
            perm = torch.randperm(rollout_buffer['states'].shape[0])
            for key in rollout_buffer.keys():
                rollout_buffer[key] = rollout_buffer[key][perm]

            for policy_loss, value_loss, entropy, kl, concentration_kappa in ppo_update(
                ppo_agent, device, ppo_args['minibatch_size'], full_rollout=rollout_buffer, ppo_clip_epsilon=ppo_args['ppo_clip_epsilon']
            ):
                # concentration_kappa = torch.log(concentration_kappa)  # log-space penalty for stability
                concentration_kappa = torch.log(1.0 + concentration_kappa) # softplus penalty
                optimizer.zero_grad()
                loss = policy_loss + value_loss + (ppo_args['entropy_coef'] * concentration_kappa) # - entropy_coef * ppo_args['entropy_coef']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ppo_agent.parameters(), max_norm=1.0)
                optimizer.step()

                update_count      += 1
                epoch_policy_loss += (policy_loss.item() - epoch_policy_loss) / update_count
                epoch_value_loss  += (value_loss.item()  - epoch_value_loss)  / update_count
                epoch_entropy     += (entropy.item()     - epoch_entropy)     / update_count
                epoch_total_loss  += (loss.item()        - epoch_total_loss)  / update_count
                epoch_kl          += (kl.item()          - epoch_kl)          / update_count
                epoch_kappa       += (concentration_kappa.item() - epoch_kappa) / update_count  

            #     # early stopping if KL divergence is too high, to prevent destructive updates
            #     if kl.item() > KL_TERMINATION_THRESHOLD:
            #         KL_terminated = True    
            #         break
            # if KL_terminated:
            #     break

        test_rollout, debug_dict_test = None, None
        if epoch % 100 == 0:
            test_rollout, debug_dict_test = generate_rollout(env, ppo_agent, deterministic=True)

        if logs_path is not None:
            logger.add_scalar('Loss/Policy',  epoch_policy_loss, epoch)
            logger.add_scalar('Loss/Value',   epoch_value_loss,  epoch)
            logger.add_scalar('Loss/Entropy', epoch_entropy,     epoch)
            logger.add_scalar('Loss/KL',      epoch_kl,          epoch)
            logger.add_scalar('Loss/Kappa',   epoch_kappa,       epoch)
            logger.add_scalar('Loss/Total',   epoch_total_loss,  epoch)
            logger.add_scalar('Stats/Update Count', update_count, epoch)

            if epoch % 100 == 0:
                final_x0s = denorm_fn(debug_dict['final_x0s']) if denorm_fn is not None else debug_dict['final_x0s']
                final_x0s = tensorboard_image_process(final_x0s)
                logger.add_image('Diffusion Samples', final_x0s, epoch)

                if test_rollout is not None:
                    final_x0s_test = denorm_fn(debug_dict_test['final_x0s']) if denorm_fn is not None else debug_dict_test['final_x0s']
                    final_x0s_test = tensorboard_image_process(final_x0s_test)
                    logger.add_image('Diffusion Samples Test', final_x0s_test, epoch)

            for key, value in debug_dict.items():
                if value is None:
                    continue
                logger.add_scalar(f'Episode Stats / {key}_mean',   value.mean().item(),   epoch)
                logger.add_scalar(f'Episode Stats / {key}_std',    value.std().item(),    epoch)
                logger.add_scalar(f'Episode Stats / {key}_max',    value.max().item(),    epoch)
                logger.add_scalar(f'Episode Stats / {key}_min',    value.min().item(),    epoch)
                logger.add_scalar(f'Episode Stats / {key}_median', value.median().item(), epoch)

            if test_rollout is not None:
                for key, value in debug_dict_test.items():
                    if value is None:
                        continue
                    logger.add_scalar(f'Test Episode Stats / {key}_mean',   value.mean().item(),   epoch)
                    logger.add_scalar(f'Test Episode Stats / {key}_std',    value.std().item(),    epoch)
                    logger.add_scalar(f'Test Episode Stats / {key}_max',    value.max().item(),    epoch)
                    logger.add_scalar(f'Test Episode Stats / {key}_min',    value.min().item(),    epoch)
                    logger.add_scalar(f'Test Episode Stats / {key}_median', value.median().item(), epoch)

            for key, value in rollout_buffer.items():
                logger.add_scalar(f'Rollout/{key}_mean',   value.mean().item(),   epoch)
                logger.add_scalar(f'Rollout/{key}_std',    value.std().item(),    epoch)
                logger.add_scalar(f'Rollout/{key}_min',    value.min().item(),    epoch)
                logger.add_scalar(f'Rollout/{key}_max',    value.max().item(),    epoch)
                logger.add_scalar(f'Rollout/{key}_median', value.median().item(), epoch)

            if test_rollout is not None:
                for key, value in test_rollout.items():
                    logger.add_scalar(f'Test Rollout/{key}_mean',   value.mean().item(),   epoch)
                    logger.add_scalar(f'Test Rollout/{key}_std',    value.std().item(),    epoch)
                    logger.add_scalar(f'Test Rollout/{key}_min',    value.min().item(),    epoch)
                    logger.add_scalar(f'Test Rollout/{key}_max',    value.max().item(),    epoch)
                    logger.add_scalar(f'Test Rollout/{key}_median', value.median().item(), epoch)
        
        current_avg_reward = debug_dict['terminal_rewards'].mean().item()
        if current_avg_reward > best_avg_reward:
            best_avg_reward = current_avg_reward
            save_best = True

        if save_path is not None:
            checkpoint_file  = os.path.join(save_path, 'ppo_checkpoint.pth')
            temp_checkpoint  = checkpoint_file + ".tmp"

            try:
                checkpoint_data = {
                    'epoch':                epoch,
                    'model_state_dict':     ppo_agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint_data, temp_checkpoint)
                os.replace(temp_checkpoint, checkpoint_file)

            except Exception as e:
                print(f"Warning: Failed to save checkpoint at epoch {epoch}: {e}")
                if os.path.exists(temp_checkpoint):
                    os.remove(temp_checkpoint)
            
            if save_best:
                best_checkpoint_file = os.path.join(save_path, 'ppo_checkpoint_best.pth')
                try:
                    torch.save(checkpoint_data, best_checkpoint_file)
                    # print(f"Saved new best checkpoint with avg reward {best_avg_reward:.4f} at epoch {epoch}")
                except Exception as e:
                    print(f"Warning: Failed to save best checkpoint at epoch {epoch}: {e}")
                
                logger.add_scalar('Best Avg Reward', best_avg_reward, epoch)
                save_best = False


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    parser = argparse.ArgumentParser(description='Train Diffusion Policy with PPO')

    parser.add_argument('--dataset',              type=str,   default='CelebAHQ',       help='Dataset to use: CIFAR10, MNIST, CelebAHQ')
    parser.add_argument('--batch_size',           type=int,   default=8,               help='Batch size for training')
    parser.add_argument('--budget',               type=int,   default=10,             help='Maximum number of steps per episode')
    parser.add_argument('--base_dataset_path',    type=str,   default='/Users/danielbairamian/Desktop/RLDiffusion_data/datasets/',        help='Base path for datasets')
    parser.add_argument('--base_logs_path',       type=str,   default='/Users/danielbairamian/Desktop/RLDiffusion_data/logs/PPO/',   help='Base path for logs and checkpoints')
    parser.add_argument('--base_path_diffusion',  type=str,   default='/Users/danielbairamian/Desktop/RLDiffusion_data/logs/diffusion/', help='Base path for diffusion checkpoints')
    parser.add_argument('--diffusion_model',      type=str,   default='IADB',          help='Diffusion model to use: DDIM, IADB')
    parser.add_argument('--fused_dims',           type=int,   default=64,              help='Dimension of the fused state-time representation')
    parser.add_argument('--time_encoder_dims',    type=int,   nargs='+', default=[32, 64],       help='Output dims for each layer in the time encoder')
    parser.add_argument('--projection_dims',      type=int,   nargs='+', default=[256, 128],     help='Output dims for each layer in the projection encoder')
    parser.add_argument('--num_epochs',           type=int,   default=200,             help='Number of epochs to train')
    parser.add_argument('--lr',                   type=float, default=1e-4,            help='Learning rate for optimizer')
    parser.add_argument('--weight_decay',         type=float, default=1e-3,            help='Weight decay for optimizer')
    parser.add_argument('--entropy_coef',         type=float, default=0.0,             help='Entropy coefficient for PPO')
    parser.add_argument('--target_steps',         type=int,   default=64,             help='Steps to collect per PPO update')
    parser.add_argument('--minibatch_size',       type=int,   default=256,             help='Minibatch size for PPO updates')
    parser.add_argument('--num_ppo_epochs',       type=int,   default=1,               help='PPO epochs per update')
    parser.add_argument('--sample_multiplier',    type=int,   default=4,               help='x1 samples generated per x0 sample in the environment')
    parser.add_argument('--order',                type=int,   default=2,               help='Order of the method')
    parser.add_argument('--latent_dim',           type=int,   default=512,             help='Dimensionality of the image state latent space')
    parser.add_argument('--latent_channels',      type=int,   nargs='+', default=[32, 64, 128], help='Latent channels for the encoder')
    parser.add_argument('--feature_extractor',    type=str,   default="IV3",          help='Feature extractor to use: IV3, DINO')
    parser.add_argument('--ppo_clip_epsilon',     type=float, default=0.1,             help='Clipping epsilon for PPO updates')
    parser.add_argument('--gae_lambda',           type=float, default=1.0,            help='GAE lambda for advantage estimation')
    parser.add_argument('--gamma',                type=float, default=1.0,             help='Discount factor for rewards')
    parser.add_argument('--kl_termination_value', type=float, default=0.1,             help='KL divergence threshold for early stopping of PPO updates')
    parser.add_argument('--seed',                 type=int,   default=42,              help='Random seed for reproducibility')
    args = parser.parse_args()


    torch.manual_seed(args.seed)

    if args.dataset == "CIFAR10":
        load_fn = CIFAR_dataloader
    elif args.dataset == "MNIST":
        load_fn = MNIST_dataloader
    elif args.dataset == "CelebAHQ":
        load_fn = CelebAHQ_dataloader
    else:
        raise ValueError("Unsupported dataset. Choose from: CIFAR10, MNIST, CelebAHQ")


    dataset_path   = args.base_dataset_path + args.dataset
    ppo_exp_suffix = f"{args.dataset}_NFE_{args.budget}_order_{args.order}_{args.feature_extractor}"
    
    logs_path      = args.base_logs_path + args.diffusion_model + f"/tensorboard/{ppo_exp_suffix}/"
    save_path      = args.base_logs_path + args.diffusion_model + f"/checkpoints/{ppo_exp_suffix}/"
    diffusion_path = args.base_path_diffusion + args.diffusion_model + f"/checkpoints/{args.dataset}/"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)

    if args.diffusion_model == "IADB":
        model = torch.load(os.path.join(diffusion_path, 'iadb_model.pth'), map_location=device).eval()
    elif args.diffusion_model == "DDIM":
        if args.dataset == "CIFAR10":
            model = load_ddim_wrapper('google/ddpm-cifar10-32', device)
        elif args.dataset == "MNIST":
            raise ValueError("No publicly available DDIM model for MNIST. Please train your own or choose a different dataset.")
        elif args.dataset == "CelebAHQ":
            model = load_ddim_wrapper('google/ddpm-celebahq-256', device)
        else:
            raise ValueError("Unsupported dataset for DDIM. Choose from: CIFAR10, MNIST, CelebAHQ")
        
    dataloader, info_dict, denorm_fn_data = load_fn(dataset_path, batch_size=args.batch_size * args.sample_multiplier)

    if args.diffusion_model == "DDIM":
        denorm_fn = lambda x: (x + 1.0) / 2.0  # Rescale from [-1, 1] to [0, 1]
    else:
        denorm_fn = denorm_fn_data

    vision_encoder = VisionEncoder(
        input_W=info_dict['W'], input_H=info_dict['H'], input_channels=info_dict['C'],
        latent_channels=args.latent_channels, latent_dim=args.latent_dim
    )

    if args.diffusion_model == "IADB":
        env_cls = IADBDiffusionEnv
    elif args.diffusion_model == "DDIM":
        env_cls = DDIMDiffusionEnv

    env = env_cls(
        dataloader=dataloader, model=model, device=device,
        order=args.order, budget=args.budget,
        sample_multiplier=args.sample_multiplier, denorm_fn=denorm_fn, denorm_fn_data=denorm_fn_data, feature_extractor=args.feature_extractor
    )
    env.reset()

    ppo_agent = PPOAgent(
        vision_encoder=vision_encoder,
        state_dim=args.latent_dim,
        fused_dims=args.fused_dims,
        time_encoder_dims=args.time_encoder_dims,
        projection_dims=args.projection_dims,
        action_dim=1,
        mean_action_init=(1.0 / env.budget),
        concentration_init=8.0
    ).to(device)

    ppo_args = {
        'num_epochs': args.num_epochs,
        'target_steps': args.target_steps,
        'minibatch_size': args.minibatch_size,
        'num_ppo_epochs': args.num_ppo_epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'entropy_coef': args.entropy_coef,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'ppo_clip_epsilon': args.ppo_clip_epsilon,
        'kl_termination_value': args.kl_termination_value,
    }

    train_PPO(
        env, ppo_agent, device, ppo_args,
        denorm_fn=denorm_fn,
        logs_path=logs_path,
        save_path=save_path,
    )