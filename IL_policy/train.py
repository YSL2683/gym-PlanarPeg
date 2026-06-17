import os
import logging
import torch
import wandb
import hydra
import json
import datetime
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import random
import numpy as np
import zarr
from tqdm import tqdm

from diffusers.training_utils import EMAModel
from utils.dataset import PlanarPegDataset
from diffusion_policy import DiffusionPolicy

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    torch.backends.cudnn.benchmark = True
    
    # Set seeds
    if "seed" in cfg:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        
    print(OmegaConf.to_yaml(cfg))
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project, 
        name=cfg.wandb.name, 
        dir=cfg.output_dir,
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
    
    stats_path = os.path.join(cfg.output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=True, 
        num_workers=cfg.train.get("num_workers", 4), 
        pin_memory=True
    )
    
    val_dataloader = None
    if val_episodes > 0:
        val_dataset = PlanarPegDataset(cfg, episode_indices=val_indices)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.val.get("batch_size", 64),
            shuffle=False,
            num_workers=cfg.val.get("num_workers", 4),
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
        
    use_amp = cfg.train.get("use_amp", False)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    grad_scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    best_loss = float('inf')

    logger.info("Starting training...")
    model.train()
    
    progress_bar = tqdm(total=total_steps, desc="Training")
    step = 0
    step_loss = 0.0
    
    log_freq = cfg.train.get("log_freq", 100)
    save_freq = cfg.train.get("save_freq", 10000)
    save_dir = cfg.checkpoint_dir
    
    while step < total_steps:
        for obs_dict, action in train_dataloader:
            if step >= total_steps:
                break
                
            obs_dict = {k: v.to(device) for k, v in obs_dict.items()}
            action = action.to(device)

            optimizer.zero_grad()
            
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss = model.compute_loss(obs_dict, action)
            
            grad_scaler.scale(loss).backward()
            
            max_grad_norm = cfg.train.get("max_grad_norm", None)
            if max_grad_norm is not None:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            
            if ema is not None:
                ema.step(model.parameters())
            
            step_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            step += 1
            
            if step % log_freq == 0:
                avg_loss = step_loss / log_freq
                lr = scheduler.get_last_lr()[0]
                wandb.log({"train/step": step, "train/loss": avg_loss, "train/lr": lr})
                
                tqdm.write(f"Step {step}/{total_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                
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
                logger.info(f"Validation at Step {step} | Val Loss: {avg_val_loss:.4f}")
                
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
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
