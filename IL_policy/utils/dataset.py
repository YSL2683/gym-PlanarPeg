import torch
import zarr
import numpy as np
from torch.utils.data import Dataset
from omegaconf import DictConfig

def create_sample_indices(
        episode_ends: np.ndarray, sequence_length: int, 
        pad_before: int = 0, pad_after: int = 0,
        episode_indices: list = None):
    indices = []
    for i in range(len(episode_ends)):
        if episode_indices is not None and i not in episode_indices:
            continue
            
        start_idx = 0 if i == 0 else episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx,
            ])
    return np.array(indices)

def sample_sequence(train_data, sequence_length, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx):
    result = np.zeros((sequence_length,) + train_data.shape[1:], dtype=train_data.dtype)
    result[sample_start_idx:sample_end_idx] = train_data[buffer_start_idx:buffer_end_idx]
    
    if sample_start_idx > 0:
        result[:sample_start_idx] = result[sample_start_idx]
    if sample_end_idx < sequence_length:
        result[sample_end_idx:] = result[sample_end_idx - 1]
        
    return result

class PlanarPegDataset(Dataset):
    def __init__(self, cfg: DictConfig, episode_indices: list = None):
        super().__init__()
        self.dataset_root = cfg.task.dataset_root
        self.root = zarr.open(self.dataset_root, mode='r')
        
        self.obs_horizon = cfg.policy.obs_horizon
        self.pred_horizon = cfg.policy.pred_horizon
        self.action_horizon = cfg.policy.action_horizon
        self.sequence_length = self.pred_horizon
        
        episode_ends = self.root['meta/episode_ends'][:]
        
        pad_before = self.obs_horizon - 1
        pad_after = self.pred_horizon - 1
        
        self.indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=self.sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_indices=episode_indices
        )
        
        cache_all = cfg.train.get("cache_all", False)
        if cache_all:
            print("Caching dataset into RAM...")
            self.img_top = self.root['data/observation.images.top'][:]
            self.img_front = self.root['data/observation.images.front'][:]
        else:
            self.img_top = self.root['data/observation.images.top']
            self.img_front = self.root['data/observation.images.front']
            
        self.proprio = self.root['data/observation.state'][:]
        self.action = self.root['data/action'][:]

    def get_stats(self):
        """Computes min/max over the dataset for normalization."""
        stats = {}
        
        actions = self.action
        stats["action"] = {
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist()
        }
        
        proprio = self.proprio
        stats["observation.state"] = {
            "min": proprio.min(axis=0).tolist(),
            "max": proprio.max(axis=0).tolist()
        }
        return stats

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        
        img_top = sample_sequence(self.img_top, self.sequence_length, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        img_front = sample_sequence(self.img_front, self.sequence_length, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        proprio = sample_sequence(self.proprio, self.sequence_length, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        action = sample_sequence(self.action, self.sequence_length, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        
        img_top = torch.from_numpy(img_top).float().permute(0, 3, 1, 2) / 255.0
        img_front = torch.from_numpy(img_front).float().permute(0, 3, 1, 2) / 255.0
        
        proprio = torch.from_numpy(proprio).float()
        action = torch.from_numpy(action).float()
        
        obs_dict = {
            "observation.images.top": img_top[:self.obs_horizon],
            "observation.images.front": img_front[:self.obs_horizon],
            "observation.state": proprio[:self.obs_horizon]
        }
        
        return obs_dict, action
