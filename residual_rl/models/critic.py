import torch
import torch.nn as nn
import numpy as np

class REDQCriticEnsemble(nn.Module):
    """
    Randomized Ensemble Double Q-learning (REDQ) Critic.
    Uses LayerNorm for stability during residual finetuning.
    """

    def __init__(self, obs_dim, action_dim=3, hidden_dim=256, num_q=10, num_q_subsample=2):
        super().__init__()
        self.num_q = num_q
        self.num_q_subsample = num_q_subsample

        self.q_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_q)
        ])

    def forward(self, obs_feat, action):
        """
        Args:
            obs_feat: (B, obs_dim)
            action: (B, action_dim)
        Returns:
            q_values: (num_q, B, 1) tensor
        """
        x = torch.cat([obs_feat, action], dim=-1)
        q_values = torch.stack([net(x) for net in self.q_nets], dim=0)
        return q_values

    def q_target(self, obs_feat, action):
        """
        Conservative target using min over random subset of Q-networks.
        Returns: (B, 1) tensor
        """
        with torch.no_grad():
            q_values = self.forward(obs_feat, action) # (num_q, B, 1)
            
            # Random subsample
            indices = np.random.choice(self.num_q, self.num_q_subsample, replace=False)
            subsampled_q_values = q_values[indices] # (num_q_subsample, B, 1)
            
            min_q, _ = torch.min(subsampled_q_values, dim=0) # (B, 1)
            return min_q

    def q_for_policy(self, obs_feat, action):
        """
        Mean over all Q-networks for actor gradient.
        Returns: (B, 1) tensor
        """
        q_values = self.forward(obs_feat, action) # (num_q, B, 1)
        mean_q = torch.mean(q_values, dim=0) # (B, 1)
        return mean_q
