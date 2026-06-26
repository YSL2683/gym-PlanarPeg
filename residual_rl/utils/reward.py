import torch
from residual_rl.models.latent_encoder import DINOFrontEncoder, MLPE2C

class LaNEDenseReward:
    """
    Computes dense reward based on distances in the learned latent space
    from cached demonstration trajectories.
    """
    def __init__(self, latent_cache_path, device='cuda', discount_gamma=0.98, p_reward=1.0):
        self.device = device
        self.discount_gamma = discount_gamma
        self.p_reward = p_reward

        # Load cache
        print(f"Loading latent cache from {latent_cache_path}")
        cache = torch.load(latent_cache_path, map_location=device, weights_only=False)
        
        self.z_demo_all = cache['z_demo_all'].to(device) # (Total_frames, z_dim)
        self.discount_all = cache['discount_all'].to(device) # (Total_frames,)
        self.ref_one_step_dist = cache['ref_one_step_dist']
        
        obs_dim = cache['obs_dim']
        z_dim = cache['z_dim']
        action_dim = cache['action_dim']
        self.use_all_cameras = cache.get('use_all_cameras', False)

        # Instantiate models
        self.dino_encoder = DINOFrontEncoder(device=device)
        self.e2c = MLPE2C(obs_dim=obs_dim, action_dim=action_dim, z_dim=z_dim).to(device)
        
        self.e2c.load_state_dict(cache['e2c_state_dict'])
        self.e2c.eval()
        for p in self.e2c.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def compute(self, front_images, done, top_images=None):
        """
        Compute additional reward for a batch of states.
        Args:
            front_images: (B, 3, 224, 224) float32 in [0, 1]
            done: (B,) bool or float
            top_images: (B, 3, 224, 224) float32 in [0, 1], optional
        Returns:
            additional_reward: (B,) float tensor
        """
        if self.p_reward == 0:
            return torch.zeros(front_images.shape[0], device=self.device)

        # 1. Embed and encode current states
        dino_emb_front = self.dino_encoder(front_images)
        if self.use_all_cameras and top_images is not None:
            dino_emb_top = self.dino_encoder(top_images)
            dino_emb = torch.cat([dino_emb_front, dino_emb_top], dim=1)
        else:
            dino_emb = dino_emb_front
            
        z_pred, _ = self.e2c.enc(dino_emb) # Using mean for comparison. Shape: (B, z_dim)

        # 2. Compute distances to all demo states at once
        # cdist computes L2 distance. We square it to get squared L2.
        # z_pred: (B, z_dim), z_demo_all: (Total_frames, z_dim)
        # dists: (B, Total_frames)
        dists = torch.cdist(z_pred, self.z_demo_all) ** 2

        # 3. Find nearest neighbor in latent space
        min_dists, min_indices = torch.min(dists, dim=1) # min_dists: (B,), min_indices: (B,)

        # 4. Compute reward
        discount_power = self.discount_all[min_indices]
        
        # Reward condition: must be close to a demo state AND not terminal
        not_done = ~done.bool()
        reward_mask = (min_dists < self.ref_one_step_dist) & not_done
        
        additional_reward = (self.discount_gamma ** discount_power) * reward_mask.float() * self.p_reward
        
        return additional_reward
