import sys
import os
import torch
import hydra
import json
import logging
from omegaconf import DictConfig
from pathlib import Path
import numpy as np
import gymnasium as gym
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import planar_peg
from diffusion_policy import DiffusionPolicy
from utils.checkpoints import get_best_checkpoint, get_latest_checkpoint

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger.info("Starting evaluation...")
    logger.info(f"Using policy: {cfg.policy.name}")
    
    run_dir = None
    
    if cfg.get("checkpoint_path") and os.path.exists(cfg.checkpoint_path):
        logger.info(f"Searching for the specified checkpoint path: {cfg.checkpoint_path}")
        ckpt_path = cfg.checkpoint_path
        run_dir = str(Path(ckpt_path).parent.parent)
    elif cfg.get("checkpoint_dir"):
        ckpt_dir_path = Path(cfg.checkpoint_dir)
        if (ckpt_dir_path / "checkpoints").exists():
            ckpt_dir_path = ckpt_dir_path / "checkpoints"
            
        logger.info(f"Searching for best checkpoint in {ckpt_dir_path}")
        ckpt_path = get_best_checkpoint(ckpt_dir_path)
        if ckpt_path is None:
            logger.info(f"Searching for latest checkpoint in {ckpt_dir_path}")
            ckpt_path = get_latest_checkpoint(ckpt_dir_path)
        
        if ckpt_path:
            run_dir = str(Path(ckpt_path).parent.parent)
    else:
        logger.warning("No valid checkpoint_dir or checkpoint_path provided")
        return
        
    if ckpt_path is None:
        logger.warning("No checkpoint found")
        return
        
    if run_dir:
        eval_log_dir = os.path.join(run_dir, "eval")
        os.makedirs(eval_log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(eval_log_dir, "eval.log"))
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(file_handler)

    num_episodes = cfg.val.get("eval_n_envs", 10)
    device = cfg.device

    logger.info("Loading environment...")
    grid_name = cfg.task.name
    env = gym.make("PlanarPegInsertion-v0", render_mode=cfg.val.get("render_mode", None), grid=grid_name)
    
    if run_dir is None:
        logger.warning("Could not determine run directory to load stats.json")
        return
        
    stats_path = os.path.join(run_dir, "stats.json")
    with open(stats_path, "r") as f:
        stats = json.load(f)
        
    logger.info(f"Loading model from {ckpt_path}...")
    model = DiffusionPolicy(cfg, stats=stats).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    obs_horizon = cfg.policy.obs_horizon
    action_horizon = cfg.policy.action_horizon

    success_count = 0

    for ep_idx in range(num_episodes):
        model._action_queue.clear()
        
        obs, _ = env.reset()
        
        obs_history_top = deque(maxlen=obs_horizon)
        obs_history_front = deque(maxlen=obs_horizon)
        obs_history_prop = deque(maxlen=obs_horizon)
        
        for _ in range(obs_horizon):
            obs_history_top.append(obs["observation.images.top"])
            obs_history_front.append(obs["observation.images.front"])
            obs_history_prop.append(obs["observation.state"])
            
        done = False
        step_idx = 0
        
        while not done:
            top_tensor = torch.from_numpy(np.stack(obs_history_top)).float().permute(0, 3, 1, 2) / 255.0
            front_tensor = torch.from_numpy(np.stack(obs_history_front)).float().permute(0, 3, 1, 2) / 255.0
            prop_tensor = torch.from_numpy(np.stack(obs_history_prop)).float()
            
            obs_dict = {
                "observation.images.top": top_tensor.unsqueeze(0).to(device),
                "observation.images.front": front_tensor.unsqueeze(0).to(device),
                "observation.state": prop_tensor.unsqueeze(0).to(device)
            }
            
            with torch.no_grad():
                action = model.select_action(obs_dict)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            obs_history_top.append(next_obs["observation.images.top"])
            obs_history_front.append(next_obs["observation.images.front"])
            obs_history_prop.append(next_obs["observation.state"])
            
            if terminated or truncated:
                if info.get("success", False):
                    success_count += 1
                    logger.info(f"Episode {ep_idx + 1}: SUCCESS")
                else:
                    logger.info(f"Episode {ep_idx + 1}: FAILED")
                done = True
                
            step_idx += 1
            
    logger.info(f"Evaluation Complete for Task [{cfg.task.name}] Grid [{grid_name}]! Success Rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    env.close()

if __name__ == "__main__":
    main()
