import torch
import numpy as np
from typing import Union

class ActionScaler:
    """Min-Max scales actions to [-1, 1] range."""
    def __init__(self, action_min, action_max, min_range=0.05, device='cuda'):
        self.device = device
        
        # Ensure minimum range to avoid division by zero
        half_range = (action_max - action_min) / 2.0
        center = (action_max + action_min) / 2.0
        
        self.half_range = np.maximum(half_range, min_range / 2.0)
        self.center = center
        
        # Torch versions
        self.center_t = torch.tensor(self.center, dtype=torch.float32, device=device)
        self.half_range_t = torch.tensor(self.half_range, dtype=torch.float32, device=device)

    @classmethod
    def from_dataset_stats(cls, action_min, action_max, min_range=0.05, device='cuda'):
        return cls(action_min, action_max, min_range, device)

    def scale(self, action: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Convert from physical to [-1, 1]"""
        if isinstance(action, torch.Tensor):
            scaled = (action - self.center_t) / self.half_range_t
            return torch.clamp(scaled, -1.0, 1.0)
        else:
            scaled = (action - self.center) / self.half_range
            return np.clip(scaled, -1.0, 1.0).astype(np.float32)

    def unscale(self, scaled: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Convert from [-1, 1] to physical"""
        if isinstance(scaled, torch.Tensor):
            clamped = torch.clamp(scaled, -1.0, 1.0)
            return clamped * self.half_range_t + self.center_t
        else:
            clamped = np.clip(scaled, -1.0, 1.0)
            return (clamped * self.half_range + self.center).astype(np.float32)


class StateStandardizer:
    """Mean-std standardizes state to N(0, 1) approximately."""
    def __init__(self, state_mean, state_std, min_std=0.1, device='cuda'):
        self.device = device
        
        self.mean = state_mean
        self.std = np.maximum(state_std, min_std)
        
        # Torch versions
        self.mean_t = torch.tensor(self.mean, dtype=torch.float32, device=device)
        self.std_t = torch.tensor(self.std, dtype=torch.float32, device=device)

    @classmethod
    def from_dataset_stats(cls, state_mean, state_std, min_std=0.1, device='cuda'):
        return cls(state_mean, state_std, min_std, device)

    def standardize(self, state: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(state, torch.Tensor):
            return (state - self.mean_t) / self.std_t
        else:
            return ((state - self.mean) / self.std).astype(np.float32)
