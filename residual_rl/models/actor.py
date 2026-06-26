import torch
import torch.nn as nn
from torch.distributions import Normal

class ResidualSACActor(nn.Module):
    """
    SAC Actor for residual action prediction.
    Output is bounded to [-action_scale, +action_scale].
    """

    def __init__(self, obs_dim, action_dim=3, hidden_dim=256, action_scale=0.15, last_layer_init_scale=0.0):
        super().__init__()
        self.action_scale = action_scale

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

        if last_layer_init_scale == 0.0:
            nn.init.zeros_(self.fc_mean.weight)
            nn.init.zeros_(self.fc_mean.bias)
        else:
            nn.init.orthogonal_(self.fc_mean.weight, gain=last_layer_init_scale)
            nn.init.zeros_(self.fc_mean.bias)

        self.LOG_STD_MIN = -5.0
        self.LOG_STD_MAX = 2.0

    def forward(self, obs_feat):
        """
        Returns (action, log_prob, raw_mean)
        """
        h = self.trunk(obs_feat)
        mean = self.fc_mean(h)
        log_std = self.fc_log_std(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        normal = Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale

        # log_prob with tanh squashing correction
        log_prob = normal.log_prob(x) - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean

    def select_action(self, obs_feat, deterministic=False):
        with torch.no_grad():
            if deterministic:
                h = self.trunk(obs_feat)
                mean = self.fc_mean(h)
                action = torch.tanh(mean) * self.action_scale
                return action
            else:
                action, _, _ = self.forward(obs_feat)
                return action
