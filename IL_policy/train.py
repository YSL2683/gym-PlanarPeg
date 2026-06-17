import os
import torch
import wandb
import hydra
import json
import datetime
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader

import random
import numpy as np
import zarr

from diffusers.training_utils import EMAModel
from utils.dataset import PlanarPegDataset
from diffusion_policy import DiffusionPolicy

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"diffusion-{now}")
    
    # Set seeds
    if "seed" in cfg:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
    
    OmegaConf.set_struct(cfg, False)
    cfg.train.output_dir = output_dir
    cfg.train.log_dir = os.path.join(output_dir, "logs")
    cfg.train.save_dir = os.path.join(output_dir, "checkpoints")
    OmegaConf.set_struct(cfg, True)

    print(OmegaConf.to_yaml(cfg))
    
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    os.makedirs(cfg.train.log_dir, exist_ok=True)
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    
    wandb.init(
        project=cfg.wandb.project, 
        name=cfg.wandb.name, 
        dir=cfg.train.output_dir,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    zarr_root = zarr.open(cfg.task.dataset_root, mode='r')
    total_episodes = len(zarr_root['meta/episode_ends'])
    
    val_episodes = cfg.val.get("num_episodes", 0)
    if val_episodes > 0:
        val_indices = list(range(total_episodes - val_episodes, total_episodes))
        train_indices = list(range(total_episodes - val_episodes))
    else:
        val_indices = None
        train_indices = None

    train_dataset = PlanarPegDataset(cfg, episode_indices=train_indices)
    
    stats = train_dataset.get_stats()
    
    stats_path = os.path.join(cfg.train.output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_dataloader = None
    if val_episodes > 0:
        val_dataset = PlanarPegDataset(cfg, episode_indices=val_indices)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.val.get("batch_size", 64),
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    device = cfg.device
    model = DiffusionPolicy(cfg, stats=stats).to(device)
    
    total_steps = cfg.train.steps
    optimizer = model.get_optimizer()
    scheduler = model.get_scheduler(optimizer, total_steps)

    use_ema = cfg.train.get("use_ema", False)
    if use_ema:
        ema = EMAModel(model.parameters(), power=cfg.train.get("ema_power", 0.75))
    else:
        ema = None

    best_loss = float('inf')

    print("Starting training...")
    model.train()
    
    progress_bar = tqdm(total=total_steps, desc="Training")
    step = 0
    step_loss = 0.0
    
    log_freq = cfg.train.get("log_freq", 100)
    save_freq = cfg.train.get("save_freq", 10000)
    save_dir = cfg.train.save_dir
    
    while step < total_steps:
        for obs_dict, action in train_dataloader:
            if step >= total_steps:
                break
                
            obs_dict = {k: v.to(device) for k, v in obs_dict.items()}
            action = action.to(device)

            optimizer.zero_grad()
            
            loss = model.compute_loss(obs_dict, action)
            
            loss.backward()
            
            max_grad_norm = cfg.train.get("max_grad_norm", None)
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
            optimizer.step()
            scheduler.step()
            
            if ema is not None:
                ema.step(model.parameters())
            
            step_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            step += 1
            
            if step % log_freq == 0:
                avg_loss = step_loss / log_freq
                wandb.log({"train/step": step, "train/loss": avg_loss, "train/lr": scheduler.get_last_lr()[0]})
                
                # If no validation set, save best based on training loss
                if val_dataloader is None and avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
                
                step_loss = 0.0
                
            val_freq = cfg.val.get("val_offline_freq", 0)
            if val_freq > 0 and val_dataloader is not None and step % val_freq == 0:
                if ema is not None:
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    
                model.eval()
                val_loss = 0.0
                num_samples = 0
                
                with torch.no_grad():
                    for val_obs_dict, val_action in val_dataloader:
                        val_obs_dict = {k: v.to(device) for k, v in val_obs_dict.items()}
                        val_action = val_action.to(device)
                        
                        v_loss = model.compute_loss(val_obs_dict, val_action)
                        batch_size = val_action.shape[0]
                        val_loss += v_loss.item() * batch_size
                        num_samples += batch_size
                
                avg_val_loss = val_loss / num_samples
                wandb.log({"val/step": step, "val/loss": avg_val_loss})
                print(f"\\nStep {step} | Val Loss: {avg_val_loss:.4f}")
                
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
                    
                model.train()
                if ema is not None:
                    ema.restore(model.parameters())
                
            if step % save_freq == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f"step_{step}.pth"))
                torch.save(model.state_dict(), os.path.join(save_dir, "latest.pth"))

    wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    main()
