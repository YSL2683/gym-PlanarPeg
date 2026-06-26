import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import zarr
from tqdm import tqdm

from residual_rl.models.latent_encoder import DINOFrontEncoder, MLPE2C

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='data/ID_base/ID_base_v2.zarr')
    parser.add_argument('--output_dir', type=str, default='residual_rl/outputs')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--obs_dim', type=int, default=384)
    parser.add_argument('--action_dim', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dino_batch_size', type=int, default=32)
    parser.add_argument('--wandb_project', type=str, default='PlanarPeg_ResidualRL')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str, default='online')
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA is requested but not available on this system. Falling back to CPU.")
        device = 'cpu'
        
    import datetime
    run_name = f"latent_encoder-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, 'latent_cache.pt')
    
    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            name=run_name,
            dir=run_dir,
            config=vars(args)
        )
    except Exception as e:
        print(f"Failed to init wandb: {e}")
        wandb = None

    # 1. Load zarr dataset
    print(f"Loading dataset from {args.dataset_root}")
    root = zarr.open(args.dataset_root, 'r')
    
    images = root['data/observation.images.front'][:] # (N, H, W, C) uint8
    states = root['data/observation.state'][:] # (N, 3) float32
    episode_ends = root['meta/episode_ends'][:] # (num_episodes,) int64
    
    num_frames = images.shape[0]
    num_episodes = episode_ends.shape[0]
    print(f"Loaded {num_episodes} episodes, {num_frames} frames.")

    # 2. Extract DINO embeddings
    print("Extracting DINO embeddings...")
    dino_encoder = DINOFrontEncoder(device=device)
    
    all_embeddings = []
    
    for i in tqdm(range(0, num_frames, args.dino_batch_size)):
        batch_imgs = images[i:i+args.dino_batch_size]
        # Preprocess: (B, H, W, C) -> (B, C, H, W)
        batch_imgs = torch.from_numpy(batch_imgs).permute(0, 3, 1, 2).float() / 255.0
        batch_imgs = batch_imgs.to(device)
        
        with torch.no_grad():
            emb = dino_encoder(batch_imgs)
        
        all_embeddings.append(emb.cpu())
        
    all_embeddings = torch.cat(all_embeddings, dim=0) # (N, 384)
    
    # 3. Build transitions
    print("Building transitions...")
    obs_emb_list = []
    action_list = []
    next_obs_emb_list = []
    
    start_idx = 0
    for end_idx in episode_ends:
        # We need pairs of (t, t+1) within the episode
        ep_emb = all_embeddings[start_idx:end_idx]
        ep_states = states[start_idx:end_idx]
        
        seq_len = end_idx - start_idx
        if seq_len > 1:
            obs_emb_list.append(ep_emb[:-1])
            next_obs_emb_list.append(ep_emb[1:])
            
            # Action as delta state
            ep_actions = ep_states[1:] - ep_states[:-1]
            action_list.append(torch.from_numpy(ep_actions))
            
        start_idx = end_idx
        
    obs_emb_tensor = torch.cat(obs_emb_list, dim=0)
    action_tensor = torch.cat(action_list, dim=0)
    next_obs_emb_tensor = torch.cat(next_obs_emb_list, dim=0)
    
    dataset = TensorDataset(obs_emb_tensor, action_tensor, next_obs_emb_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 4. Train MLPE2C
    print("Training MLPE2C...")
    e2c = MLPE2C(obs_dim=args.obs_dim, action_dim=args.action_dim, z_dim=args.z_dim).to(device)
    optimizer = torch.optim.Adam(e2c.parameters(), lr=args.lr)
    
    for epoch in range(args.num_epochs):
        e2c.train()
        total_loss = 0
        total_kl = 0
        total_recon = 0
        total_trans = 0
        
        for obs_emb, action, next_obs_emb in dataloader:
            obs_emb = obs_emb.to(device)
            action = action.to(device)
            next_obs_emb = next_obs_emb.to(device)
            
            kl, recon, trans = e2c(obs_emb, action, next_obs_emb)
            
            # Weight MSE by obs_dim (768 or 384) as in LaNE
            loss = kl + recon * args.obs_dim + trans
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_kl += kl.item()
            total_recon += recon.item()
            total_trans += trans.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs} | Loss: {total_loss/len(dataloader):.4f} "
                  f"(KL: {total_kl/len(dataloader):.4f}, Recon: {total_recon/len(dataloader):.4f}, Trans: {total_trans/len(dataloader):.4f})")
            
        if 'wandb' in locals() and wandb is not None and wandb.run is not None:
            wandb.log({
                'train/loss': total_loss / len(dataloader),
                'train/kl_loss': total_kl / len(dataloader),
                'train/recon_loss': total_recon / len(dataloader),
                'train/trans_loss': total_trans / len(dataloader),
                'epoch': epoch + 1
            }, step=epoch + 1)
            
    # 5. Cache Demo Latents and compute ref_one_step_dist
    print("Caching demo latents...")
    e2c.eval()
    
    z_demo_list = []
    discount_list = []
    one_step_dists = []
    
    start_idx = 0
    with torch.no_grad():
        for end_idx in episode_ends:
            ep_emb = all_embeddings[start_idx:end_idx].to(device)
            seq_len = end_idx - start_idx
            
            # Encode entire episode (mean only)
            z_mean, _ = e2c.enc(ep_emb)
            z_demo_list.append(z_mean.cpu())
            
            # Compute discounts: steps remaining to end
            discounts = torch.arange(seq_len - 1, -1, -1)
            discount_list.append(discounts)
            
            # Compute one-step squared distances
            if seq_len > 1:
                dists = ((z_mean[1:] - z_mean[:-1]) ** 2).sum(dim=1)
                one_step_dists.append(dists.mean().item())
                
            start_idx = end_idx
            
    z_demo_all = torch.cat(z_demo_list, dim=0)
    discount_all = torch.cat(discount_list, dim=0)
    ref_one_step_dist = np.mean(one_step_dists)
    
    print(f"ref_one_step_dist: {ref_one_step_dist:.4f}")
    if 'wandb' in locals() and wandb is not None and wandb.run is not None:
        wandb.log({'ref_one_step_dist': ref_one_step_dist})
    
    # 6. Save
    print(f"Saving to {output_path}")
    torch.save({
        'e2c_state_dict': e2c.state_dict(),
        'z_demo_all': z_demo_all,
        'discount_all': discount_all,
        'ref_one_step_dist': ref_one_step_dist,
        'obs_dim': args.obs_dim,
        'z_dim': args.z_dim,
        'action_dim': args.action_dim,
    }, output_path)
    
    if 'wandb' in locals() and wandb is not None and wandb.run is not None:
        wandb.finish()
        
    print("Done!")

if __name__ == '__main__':
    main()
