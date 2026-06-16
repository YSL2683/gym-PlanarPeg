import sys
import os
import torch
import hydra
import json
from omegaconf import DictConfig
import numpy as np
import gymnasium as gym
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import planar_peg
from diffusion_policy import DiffusionPolicy

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):
    if cfg.eval_dir is None:
        raise ValueError("Please provide eval_dir, e.g. python IL_policy/eval.py eval_dir=outputs/diffusion-20260522_195047")
        
    ckpt_name = cfg.get("checkpoint_name", "best.pth")
    ckpt_path = os.path.join(cfg.eval_dir, "checkpoints", ckpt_name)
    num_episodes = 10
    device = cfg.device

    print(f"Loading environment...")
    env = gym.make("PlanarPegInsertion-v0", render_mode="human")
    
    stats_path = os.path.join(cfg.eval_dir, "stats.json")
    with open(stats_path, "r") as f:
        stats = json.load(f)
        
    print(f"Loading model from {ckpt_path}...")
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
                    print(f"Episode {ep_idx + 1}: SUCCESS")
                else:
                    print(f"Episode {ep_idx + 1}: FAILED")
                done = True
                
            step_idx += 1
            
    print(f"\nEvaluation Complete! Success Rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    env.close()

if __name__ == "__main__":
    main()
