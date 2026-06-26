import torch
import numpy as np
import random

class SymmetricReplayBuffer:
    """
    Replay buffer that enforces a specific ratio of offline vs online data sampling.
    """
    def __init__(self, capacity=300000, offline_fraction=0.5):
        self.capacity = capacity
        self.offline_fraction = offline_fraction
        
        self.offline_buffer = []
        
        self.online_buffer = []
        self.online_pos = 0

    def add_online(self, transition_dict):
        """Add a single transition dict to the online buffer (circular)."""
        if len(self.online_buffer) < self.capacity:
            self.online_buffer.append(transition_dict)
        else:
            self.online_buffer[self.online_pos] = transition_dict
        self.online_pos = (self.online_pos + 1) % self.capacity

    def load_offline(self, list_of_transition_dicts):
        """Load offline data into the offline buffer."""
        self.offline_buffer.extend(list_of_transition_dicts)
        print(f"Loaded {len(list_of_transition_dicts)} offline transitions.")

    def sample(self, batch_size, device='cuda'):
        """Sample a batch maintaining the offline_fraction."""
        n_offline = int(batch_size * self.offline_fraction)
        n_online = batch_size - n_offline

        # Adjust ratio if online buffer is too small
        if len(self.online_buffer) < n_online:
            n_online = len(self.online_buffer)
            n_offline = batch_size - n_online

        # If offline buffer is too small (should not happen, but safeguard)
        if len(self.offline_buffer) < n_offline:
            n_offline = len(self.offline_buffer)
            n_online = batch_size - n_offline

        batch = []
        if n_offline > 0:
            batch.extend(random.sample(self.offline_buffer, n_offline))
        if n_online > 0:
            batch.extend(random.sample(self.online_buffer, n_online))

        # Convert list of dicts to dict of batched tensors
        batch_dict = {}
        if len(batch) > 0:
            keys = batch[0].keys()
            for k in keys:
                # Stack numpy arrays into torch tensors
                arr = np.array([x[k] for x in batch])
                batch_dict[k] = torch.from_numpy(arr).float().to(device)

        return batch_dict

    def __len__(self):
        return len(self.offline_buffer) + len(self.online_buffer)
