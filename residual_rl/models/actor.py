import torch
import torch.nn as nn
from torch.distributions import Normal

class ResidualTD3Actor(nn.Module):
    """
    TD3/RLPD Actor for residual action prediction.
    Output is deterministic and bounded to [-action_scale, +action_scale].
    """

    def __init__(self, feat_dim, prop_dim=6, action_dim=3, hidden_dim=256, feature_dim=50, action_scale=0.15, last_layer_init_scale=0.0):
        super().__init__()
        self.action_scale = action_scale

        self.compress = nn.Sequential(
            nn.Linear(feat_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        policy_in_dim = feature_dim + prop_dim

        self.trunk = nn.Sequential(
            nn.Linear(policy_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.fc_mean = nn.Linear(hidden_dim, action_dim)

        if last_layer_init_scale == 0.0:
            nn.init.zeros_(self.fc_mean.weight)
            nn.init.zeros_(self.fc_mean.bias)
        else:
            nn.init.orthogonal_(self.fc_mean.weight, gain=last_layer_init_scale)
            nn.init.zeros_(self.fc_mean.bias)

    def forward(self, feat, prop):
        """
        Returns raw action (before scaling) and scaled action
        """
        comp_feat = self.compress(feat)
        policy_input = torch.cat([comp_feat, prop], dim=-1)
        h = self.trunk(policy_input)
        mean = self.fc_mean(h)
        y = torch.tanh(mean)
        action = y * self.action_scale

        return action, mean

    def select_action(self, feat, prop, deterministic=False, stddev=0.0):
        with torch.no_grad():
            action, _ = self.forward(feat, prop)
            if not deterministic and stddev > 0.0:
                # Add TruncatedNormal noise
                noise = torch.randn_like(action) * stddev
                # Clip noise to 2 * stddev for stability (DrQv2 style)
                noise = torch.clamp(noise, -2 * stddev, 2 * stddev)
                action = action + noise
                # Clamp to action bounds
                action = torch.clamp(action, -self.action_scale, self.action_scale)
            return action
