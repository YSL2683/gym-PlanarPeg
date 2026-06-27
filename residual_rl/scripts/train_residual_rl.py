import os
import sys
import json
import argparse
import random
import zarr
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from copy import deepcopy
from tqdm import tqdm

# Add project root to sys path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import base policy
sys.path.append(os.path.join(project_root, 'IL_policy'))
from diffusion_policy import DiffusionPolicy

# Import residual components
from residual_rl.models.actor import ResidualTD3Actor
from residual_rl.models.critic import REDQCriticEnsemble
from residual_rl.utils.normalization import ActionScaler, StateStandardizer
from residual_rl.utils.replay_buffer import SymmetricReplayBuffer
from residual_rl.utils.reward import LaNEDenseReward
from residual_rl.wrappers.residual_env_wrapper import PlanarPegResidualWrapper

import gymnasium as gym
import planar_peg # Register env

try:
    import wandb
except ImportError:
    wandb = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='residual_rl/configs/residual_sac.yaml')
    parser.add_argument('--base_policy_run_dir', type=str, required=True, help='Path to IL_policy output dir')
    parser.add_argument('--base_policy_checkpoint', type=str, default='best.pth')
    parser.add_argument('--base_policy_config', type=str, default='IL_policy/configs/base.yaml')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--wandb_mode', type=str, default=None)
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_feat_and_prop(obs, device, dense_reward_calc, is_all_cams):
    state = torch.from_numpy(obs['observation.state']).float().unsqueeze(0).to(device)
    base_action = torch.from_numpy(obs['observation.base_action']).float().unsqueeze(0).to(device)
    prop = torch.cat([state, base_action], dim=-1) # (1, 6)
    
    front_img = torch.from_numpy(obs['observation.images.front']).unsqueeze(0).to(device)
    with torch.no_grad():
        emb_front = dense_reward_calc.dino_encoder(front_img)
        if is_all_cams:
            top_img = torch.from_numpy(obs['observation.images.top']).unsqueeze(0).to(device)
            emb_top = dense_reward_calc.dino_encoder(top_img)
            feat = torch.cat([emb_front, emb_top], dim=1)
        else:
            feat = emb_front
            
    return feat, prop

def evaluate(eval_env, base_policy, actor, action_scaler, state_standardizer, cfg, device, dense_reward_calc, is_all_cams, step=None, wandb_run=None):
    # Create wrapper
    eval_wrapper = PlanarPegResidualWrapper(eval_env, base_policy, action_scaler, state_standardizer, device)
    successes = 0
    
    video_frames = []
    
    for i in range(cfg.eval.episodes):
        obs, _ = eval_wrapper.reset()
        done = False
        record_video = (i == 0 and wandb_run is not None)
        
        if record_video:
            top_img = eval_wrapper.env.unwrapped._get_obs()['observation.images.top']
            video_frames.append(top_img)
            
        while not done:
            feat, prop = extract_feat_and_prop(obs, device, dense_reward_calc, is_all_cams)
            with torch.no_grad():
                action = actor.select_action(feat, prop, deterministic=True)
                action = action.cpu().numpy().flatten()
            obs, reward, terminated, truncated, info = eval_wrapper.step(action)
            done = terminated or truncated
            
            if record_video:
                top_img = eval_wrapper.env.unwrapped._get_obs()['observation.images.top']
                video_frames.append(top_img)
                
            if info.get('success', False): # Only trust the explicit success flag
                successes += 1
                break
                
    if len(video_frames) > 0 and wandb_run is not None:
        # wandb.Video expects shape (time, channel, height, width) for RGB
        # video_frames is list of (H, W, C) -> array of (T, H, W, C)
        vid_array = np.array(video_frames)
        vid_array = np.transpose(vid_array, (0, 3, 1, 2))
        wandb_run.log({"eval/video": wandb.Video(vid_array, fps=10, format="mp4")}, step=step)
        
    return successes / cfg.eval.episodes

def main():
    args = parse_args()
    
    # 1. Setup
    cfg = OmegaConf.load(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    set_seed(seed)
    
    # Determine prefix from cache
    try:
        cache = torch.load(cfg.latent_cache_path, map_location='cpu', weights_only=False)
        is_all_cams = cache.get('use_all_cameras', False)
    except:
        is_all_cams = False

    import datetime
    prefix = "residual_sac_all_cams" if is_all_cams else "residual_sac_front_only"
    run_name = f"{prefix}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(cfg.output_dir, run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)
    
    wandb_mode = args.wandb_mode if args.wandb_mode is not None else cfg.wandb.mode
    if wandb is not None:
        try:
            wandb.init(
                project=cfg.wandb.project, 
                entity=cfg.wandb.entity, 
                mode=wandb_mode, 
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=run_dir,
                name=run_name
            )
        except Exception as e:
            print(f"Failed to init wandb: {e}")
            wandb_mode = "disabled"
    
    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA is requested but not available on this system. Falling back to CPU.")
        device = 'cpu'

    # 2. Load Base Policy
    print("Loading Base Policy...")
    cfg_base = OmegaConf.load(args.base_policy_config)
    with open(os.path.join(args.base_policy_run_dir, 'stats.json'), 'r') as f:
        stats = json.load(f)
    
    base_policy = DiffusionPolicy(cfg_base, stats=stats).to(device)
    ckpt_path = os.path.join(args.base_policy_run_dir, 'checkpoints', args.base_policy_checkpoint)
    base_policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    base_policy.eval()
    for p in base_policy.parameters():
        p.requires_grad = False

    # 3. Compute Normalization Stats
    print(f"Computing normalization stats from {cfg.dataset_root}...")
    root = zarr.open(cfg.dataset_root, 'r')
    actions = root['data/action'][:]
    states = root['data/observation.state'][:]
    
    action_scaler = ActionScaler.from_dataset_stats(
        action_min=actions.min(axis=0), action_max=actions.max(axis=0), device=device)
    state_standardizer = StateStandardizer.from_dataset_stats(
        state_mean=states.mean(axis=0), state_std=states.std(axis=0), device=device)

    # 4. Environment & Wrapper
    print("Creating environments...")
    env = gym.make(cfg.env.id, grid=cfg.env.grid)
    eval_env = gym.make(cfg.env.id, grid=cfg.eval.grid)
    wrapped_env = PlanarPegResidualWrapper(env, base_policy, action_scaler, state_standardizer, device)

    # 5. Dense Reward Module
    print("Loading Dense Reward Module...")
    dense_reward_calc = LaNEDenseReward(cfg.latent_cache_path, device, cfg.dense_reward.discount_gamma, cfg.dense_reward.p_reward)

    from residual_rl.models.actor import ResidualTD3Actor
    
    # 6. RL Networks
    prop_dim = 6 # state(3) + base_action(3)
    feat_dim = 768 if is_all_cams else 384
    action_dim = 3
    actor = ResidualTD3Actor(feat_dim=feat_dim, prop_dim=prop_dim, action_dim=action_dim, hidden_dim=cfg.actor.hidden_dim, feature_dim=50, action_scale=cfg.actor.action_scale, last_layer_init_scale=cfg.actor.last_layer_init_scale).to(device)
    actor_target = deepcopy(actor).to(device)
    critic = REDQCriticEnsemble(feat_dim=feat_dim, prop_dim=prop_dim, action_dim=action_dim, hidden_dim=cfg.critic.hidden_dim, feature_dim=50, num_q=cfg.critic.num_q, num_q_subsample=cfg.critic.num_q_subsample).to(device)
    critic_target = deepcopy(critic).to(device)
    
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic.lr)

    # 7. Replay Buffer & Offline Data
    print("Building Offline Transitions...")
    replay_buffer = SymmetricReplayBuffer(capacity=cfg.train.buffer_size, offline_fraction=cfg.train.offline_fraction)
    
    # Extract DINO embeddings for all offline data
    images_front = root['data/observation.images.front'][:]
    images_top = root['data/observation.images.top'][:] if is_all_cams else None
    
    dino_embs_offline = []
    print("Extracting DINO embeddings for offline data...")
    for i in range(0, len(images_front), 64):
        batch_front = images_front[i:i+64]
        batch_front = torch.from_numpy(batch_front).permute(0, 3, 1, 2).float() / 255.0
        batch_front = batch_front.to(device)
        with torch.no_grad():
            emb_front = dense_reward_calc.dino_encoder(batch_front)
            if is_all_cams:
                batch_top = images_top[i:i+64]
                batch_top = torch.from_numpy(batch_top).permute(0, 3, 1, 2).float() / 255.0
                batch_top = batch_top.to(device)
                emb_top = dense_reward_calc.dino_encoder(batch_top)
                emb = torch.cat([emb_front, emb_top], dim=1)
            else:
                emb = emb_front
        dino_embs_offline.append(emb.cpu().numpy())
    dino_embs_offline = np.concatenate(dino_embs_offline, axis=0)

    offline_transitions = []
    episode_ends = root['meta/episode_ends'][:]
    start_idx = 0
    for end_idx in episode_ends:
        ep_states = states[start_idx:end_idx]
        ep_actions = actions[start_idx:end_idx]
        seq_len = end_idx - start_idx
        
        for t in range(seq_len - 1):
            obs_state = state_standardizer.standardize(ep_states[t])
            obs_base_action = action_scaler.scale(ep_actions[t])
            obs_feat = dino_embs_offline[start_idx + t]
            
            res_action = np.zeros(action_dim, dtype=np.float32)
            discount_power = seq_len - 1 - t
            dense_r = (cfg.dense_reward.discount_gamma ** discount_power) * cfg.dense_reward.p_reward
            reward = -1.0 + dense_r
            next_obs_state = state_standardizer.standardize(ep_states[t+1])
            next_obs_base_action = action_scaler.scale(ep_actions[t+1])
            next_obs_feat = dino_embs_offline[start_idx + t + 1]
            done = False
            
            offline_transitions.append({
                'obs_feat': obs_feat,
                'obs_state': obs_state,
                'obs_base_action': obs_base_action,
                'action': res_action,
                'reward': float(reward),
                'next_obs_feat': next_obs_feat,
                'next_obs_state': next_obs_state,
                'next_obs_base_action': next_obs_base_action,
                'done': float(done)
            })
        
        # Last step
        obs_state = state_standardizer.standardize(ep_states[-1])
        obs_base_action = action_scaler.scale(ep_actions[-1])
        obs_feat = dino_embs_offline[end_idx - 1]
        res_action = np.zeros(action_dim, dtype=np.float32)
        discount_power = 0
        dense_r = (cfg.dense_reward.discount_gamma ** discount_power) * cfg.dense_reward.p_reward
        reward = 100.0 + dense_r # Success
        next_obs_state = obs_state
        next_obs_base_action = obs_base_action
        next_obs_feat = obs_feat
        done = True
        
        offline_transitions.append({
            'obs_feat': obs_feat,
            'obs_state': obs_state,
            'obs_base_action': obs_base_action,
            'action': res_action,
            'reward': float(reward),
            'next_obs_feat': next_obs_feat,
            'next_obs_state': next_obs_state,
            'next_obs_base_action': next_obs_base_action,
            'done': float(done)
        })
        
        start_idx = end_idx
        
    replay_buffer.load_offline(offline_transitions)

    # 8. Initial Evaluation (Base Policy Zero-Shot Performance)
    print("Evaluating Base Policy (step 0)...")
    base_success = evaluate(eval_env, base_policy, actor, action_scaler, state_standardizer, cfg, device, dense_reward_calc, is_all_cams, step=0, wandb_run=wandb.run if wandb else None)
    if wandb and wandb.run:
        wandb.log({'eval/success_rate': base_success}, step=0)
    print(f"Step 0 | Base Policy Eval Success: {base_success:.2%}")

    # 9. Training Loop
    print("Starting Training...")
    obs, info = wrapped_env.reset()
    episode_reward = 0
    episode_count = 0
    episode_steps = 0
    episode_dense_active_steps = 0
    episode_res_magnitude = 0
    
    pbar = tqdm(range(cfg.train.total_timesteps))
    for step in pbar:
        # Exploration noise schedule
        stddev = cfg.actor.action_scale * max(0.01, 1.0 - step / (cfg.train.total_timesteps * 0.8))

        # Sample action
        if step < cfg.train.critic_warmup_steps:
            residual_action = np.random.uniform(-cfg.actor.action_scale, cfg.actor.action_scale, size=action_dim).astype(np.float32)
            feat, prop = extract_feat_and_prop(obs, device, dense_reward_calc, is_all_cams)
        else:
            feat, prop = extract_feat_and_prop(obs, device, dense_reward_calc, is_all_cams)
            with torch.no_grad():
                residual_action = actor.select_action(feat, prop, deterministic=False, stddev=stddev)
                residual_action = residual_action.cpu().numpy().flatten()
                
        # Step env
        next_obs, env_reward, terminated, truncated, info = wrapped_env.step(residual_action)
        done = terminated or truncated
        
        # Compute dense reward
        front_img = torch.from_numpy(next_obs['observation.images.front']).unsqueeze(0).to(device)
        top_img = torch.from_numpy(next_obs['observation.images.top']).unsqueeze(0).to(device) if is_all_cams else None
        done_tensor = torch.tensor([done], device=device)
        dense_r = dense_reward_calc.compute(front_img, done_tensor, top_images=top_img).item()
        total_reward = env_reward + dense_r
        
        # Extract next feat for buffer
        next_feat, _ = extract_feat_and_prop(next_obs, device, dense_reward_calc, is_all_cams)
        
        # Store transition
        replay_buffer.add_online({
            'obs_feat': feat.cpu().numpy().flatten(),
            'obs_state': obs['observation.state'],
            'obs_base_action': obs['observation.base_action'],
            'action': residual_action,
            'reward': float(total_reward),
            'next_obs_feat': next_feat.cpu().numpy().flatten(),
            'next_obs_state': next_obs['observation.state'],
            'next_obs_base_action': next_obs['observation.base_action'],
            'done': float(done)
        })
        
        episode_reward += total_reward
        episode_steps += 1
        if dense_r > 0:
            episode_dense_active_steps += 1
        episode_res_magnitude += float(np.abs(residual_action).mean())
        
        # RL Update
        if step >= cfg.train.learning_starts and len(replay_buffer) >= cfg.train.batch_size:
            batch = replay_buffer.sample(cfg.train.batch_size, device)
            
            # -- Critic Update --
            with torch.no_grad():
                next_prop = torch.cat([batch['next_obs_state'], batch['next_obs_base_action']], dim=-1)
                next_action = actor_target.select_action(batch['next_obs_feat'], next_prop, deterministic=True, stddev=0.0)
                # Add clipped noise to target action
                noise = torch.randn_like(next_action) * 0.2
                noise = torch.clamp(noise, -0.5, 0.5)
                next_action = torch.clamp(next_action + noise, -cfg.actor.action_scale, cfg.actor.action_scale)
                
                target_q = critic_target.q_target(batch['next_obs_feat'], next_prop, next_action)
                td_target = batch['reward'].unsqueeze(-1) + cfg.sac.discount * (1 - batch['done'].unsqueeze(-1)) * target_q
                
            prop = torch.cat([batch['obs_state'], batch['obs_base_action']], dim=-1)
            all_q = critic(batch['obs_feat'], prop, batch['action']) # (num_q, B, 1)
            
            # Compute loss for all Q networks
            critic_loss = sum(F.mse_loss(all_q[i], td_target) for i in range(cfg.critic.num_q)) / cfg.critic.num_q
            
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            
            # -- Actor Update --
            if step >= cfg.train.critic_warmup_steps and step % cfg.sac.actor_update_freq == 0:
                feat_actor = batch['obs_feat'].detach()
                prop_actor = prop.detach()
                action_pred, _ = actor(feat_actor, prop_actor)
                q_for_actor = critic.q_for_policy(feat_actor, prop_actor, action_pred)
                
                # TD3 Actor loss (maximize Q) + Action L2 Penalty
                action_l2_reg_weight = 0.01 # L2 penalty weight
                action_l2_penalty = action_l2_reg_weight * torch.mean(torch.sum(action_pred**2, dim=-1))
                actor_loss = -q_for_actor.mean() + action_l2_penalty
                
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                
            # Update targets using polyak averaging
            if step >= cfg.train.critic_warmup_steps and step % cfg.sac.target_update_freq == 0:
                with torch.no_grad():
                    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                        target_param.data.copy_(cfg.sac.tau * param.data + (1 - cfg.sac.tau) * target_param.data)
                    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                        target_param.data.copy_(cfg.sac.tau * param.data + (1 - cfg.sac.tau) * target_param.data)
                            
        # Logging step
        if wandb and wandb.run and step >= cfg.train.critic_warmup_steps:
            log_data = {'train/q_value': all_q.mean().item(), 'train/stddev': stddev}
            if 'critic_loss' in locals(): log_data['train/critic_loss'] = critic_loss.item()
            if 'actor_loss' in locals(): 
                log_data['train/actor_loss'] = actor_loss.item()
                log_data['train/action_l2_penalty'] = action_l2_penalty.item()
            wandb.log(log_data, step=step)
            
        # Episode boundary
        if done:
            train_success = float(info.get('success', False))
            if wandb and wandb.run:
                wandb.log({
                    'train/episode_reward': episode_reward,
                    'train/episode_length': episode_steps,
                    'train/episode': episode_count,
                    'train/success_rate': train_success,
                    'train/residual_magnitude': episode_res_magnitude / episode_steps if episode_steps > 0 else 0.0,
                    'train/dense_reward_activation_rate': episode_dense_active_steps / episode_steps if episode_steps > 0 else 0.0,
                }, step=step)
            episode_reward = 0
            episode_steps = 0
            episode_dense_active_steps = 0
            episode_res_magnitude = 0
            episode_count += 1
            obs, info = wrapped_env.reset()
        else:
            obs = next_obs
            
        # Evaluation
        if step > 0 and step % cfg.eval.freq == 0:
            eval_success = evaluate(eval_env, base_policy, actor, action_scaler, state_standardizer, cfg, device, dense_reward_calc, is_all_cams, step=step, wandb_run=wandb.run if wandb else None)
            if wandb and wandb.run:
                wandb.log({'eval/success_rate': eval_success}, step=step)
            pbar.write(f"Step {step} | Eval Success: {eval_success:.2%}")
            
            torch.save({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'log_alpha': log_alpha.detach().cpu(),
                'step': step,
            }, os.path.join(ckpt_dir, f'checkpoint_{step}.pt'))

    if wandb and wandb.run:
        wandb.finish()

if __name__ == '__main__':
    main()
