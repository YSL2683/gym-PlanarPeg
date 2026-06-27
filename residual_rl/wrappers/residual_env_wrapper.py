import gymnasium as gym
import numpy as np
import torch
from collections import deque
from gymnasium.spaces import Box, Dict

class PlanarPegResidualWrapper(gym.Wrapper):
    """
    Wraps the PlanarPeg environment for residual RL.
    Maintains the state history for the Diffusion Policy (Base Policy)
    and combines its action with the learned residual action.
    """
    def __init__(self, env, base_policy, action_scaler, state_standardizer, device='cuda'):
        super().__init__(env)
        self.base_policy = base_policy
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer
        self.device = device
        
        # Diffusion policy config
        if hasattr(base_policy, 'obs_horizon'):
            self.obs_horizon = base_policy.obs_horizon
        elif hasattr(base_policy, 'config') and hasattr(base_policy.config.policy, 'obs_horizon'):
            self.obs_horizon = base_policy.config.policy.obs_horizon
        else:
            self.obs_horizon = 1 # Fallback
            
        self.action_dim = 3
        
        # Observation history for base policy
        self._obs_history_top = deque(maxlen=self.obs_horizon)
        self._obs_history_front = deque(maxlen=self.obs_horizon)
        self._obs_history_state = deque(maxlen=self.obs_horizon)
        
        self._last_base_naction = np.zeros(self.action_dim, dtype=np.float32)

        # Define spaces for residual RL
        # observation space: standardized state (3) + base_action (3) = (6,)
        low = np.full(self.action_dim * 2, -np.inf, dtype=np.float32)
        high = np.full(self.action_dim * 2, np.inf, dtype=np.float32)
        
        # Action space of residual actor is [-1, 1] mapped to action_scale
        # But gym expects bounds. We'll just define [-1, 1] as the interface.
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        
        # We return a dict observation for the Residual Actor
        self.observation_space = Dict({
            'observation.state': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'observation.base_action': Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            'observation.images.front': Box(low=0, high=1.0, shape=(3, 224, 224), dtype=np.float32)
        })

    def _build_policy_obs_dict(self):
        """Construct observation dict expected by the DiffusionPolicy."""
        # Stack numpy arrays
        top_seq = np.stack(self._obs_history_top) # (T, H, W, C)
        front_seq = np.stack(self._obs_history_front)
        state_seq = np.stack(self._obs_history_state)

        # Convert to torch and correct shape
        # Images: (1, T, C, H, W) normalized to [0, 1]
        top_tensor = torch.from_numpy(top_seq).permute(0, 3, 1, 2).float() / 255.0
        front_tensor = torch.from_numpy(front_seq).permute(0, 3, 1, 2).float() / 255.0
        
        # State: (1, T, dim)
        state_tensor = torch.from_numpy(state_seq).float()
        
        obs_dict = {
            'observation.images.top': top_tensor.unsqueeze(0).to(self.device),
            'observation.images.front': front_tensor.unsqueeze(0).to(self.device),
            'observation.state': state_tensor.unsqueeze(0).to(self.device)
        }
        return obs_dict

    def _get_base_action(self, obs_dict):
        """Query the frozen base policy for the next action."""
        with torch.no_grad():
            action = self.base_policy.select_action(obs_dict)
            # select_action handles the action queue and chunking internally
            # returns a numpy array (3,)
        return action

    def _build_augmented_obs(self, raw_obs):
        """Construct the observation dict for the Residual Actor."""
        std_state = self.state_standardizer.standardize(raw_obs['observation.state'])
        
        front_img = raw_obs['observation.images.front'].astype(np.float32) / 255.0
        front_img = np.transpose(front_img, (2, 0, 1)) # (H,W,C) -> (C,H,W)
        
        top_img = raw_obs['observation.images.top'].astype(np.float32) / 255.0
        top_img = np.transpose(top_img, (2, 0, 1))
        
        return {
            'observation.state': std_state,
            'observation.base_action': self._last_base_naction,
            'observation.images.front': front_img,
            'observation.images.top': top_img
        }

    def reset(self, **kwargs):
        # 1. Reset env
        raw_obs, info = self.env.reset(**kwargs)
        
        # 2. Clear action queue (crucial for diffusion policy chunking)
        if hasattr(self.base_policy, '_action_queue'):
            self.base_policy._action_queue.clear()
        
        # 3. Fill history
        for _ in range(self.obs_horizon):
            self._obs_history_top.append(raw_obs['observation.images.top'])
            self._obs_history_front.append(raw_obs['observation.images.front'])
            self._obs_history_state.append(raw_obs['observation.state'])
            
        # 4. Get first base action
        obs_dict = self._build_policy_obs_dict()
        base_action = self._get_base_action(obs_dict)
        
        # 5. Scale base action
        self._last_base_naction = self.action_scaler.scale(base_action)
        
        # 6. Build augmented obs
        aug_obs = self._build_augmented_obs(raw_obs)
        
        return aug_obs, info

    def step(self, residual_naction):
        # 1. Combine actions (residual is already bounded by actor's tanh * action_scale)
        combined_naction = self._last_base_naction + residual_naction
        
        # 2. Unscale to physical action space
        env_action = self.action_scaler.unscale(combined_naction)
        
        # 3. Step env
        raw_obs, reward, terminated, truncated, info = self.env.step(env_action)
        done = terminated or truncated
        
        # 4. Update history
        self._obs_history_top.append(raw_obs['observation.images.top'])
        self._obs_history_front.append(raw_obs['observation.images.front'])
        self._obs_history_state.append(raw_obs['observation.state'])
        
        # 5. Handle episode end
        if done and hasattr(self.base_policy, '_action_queue'):
            self.base_policy._action_queue.clear()
            
        # 6. Get next base action (for next step's observation)
        if not done:
            obs_dict = self._build_policy_obs_dict()
            base_action = self._get_base_action(obs_dict)
            self._last_base_naction = self.action_scaler.scale(base_action)
        else:
            base_action = self.action_scaler.unscale(self._last_base_naction) # Keep last
            
        # 7. Build augmented obs
        aug_obs = self._build_augmented_obs(raw_obs)
        
        info['combined_naction'] = combined_naction
        info['base_action'] = base_action
        
        return aug_obs, reward, terminated, truncated, info
