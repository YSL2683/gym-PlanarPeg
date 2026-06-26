import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.optimization import get_scheduler
from omegaconf import DictConfig
from collections import deque
from models.vision_encoder import VisionEncoder
from models.conditional_unet1d import ConditionalUnet1d
from utils.normalize import NormalizeMinMax, UnnormalizeMinMax

NOISE_SCHEDULER_REGISTRY = {
    "DDPM": DDPMScheduler,
    "DDIM": DDIMScheduler,
}

class DiffusionPolicy(nn.Module):
    def __init__(self, cfg: DictConfig, stats: dict = None):
        super().__init__()
        self.config = cfg
        self.pred_horizon = cfg.policy.pred_horizon
        self.obs_horizon = cfg.policy.obs_horizon
        self.action_horizon = cfg.policy.action_horizon
        self.action_dim = cfg.task.action_dim
        self.use_delta_action = cfg.task.get("use_delta_action", False)
        
        if stats is not None:
            self.normalize_inputs = NormalizeMinMax(["observation.state"], stats)
            self.normalize_targets = NormalizeMinMax(["action"], stats)
            self.unnormalize_outputs = UnnormalizeMinMax(["action"], stats)
        else:
            self.normalize_inputs = nn.Identity()
            self.normalize_targets = nn.Identity()
            self.unnormalize_outputs = nn.Identity()
            
        self._action_queue = deque()
        self._latest_full_pred = None
        
        self.vision_encoder = VisionEncoder(cfg)
        
        unet_cfg = cfg.policy.unet
        self.noise_pred_net = ConditionalUnet1d(
            input_dim=self.action_dim,
            global_cond_dim=self.vision_encoder.out_dim * self.obs_horizon,
            down_dims=list(unet_cfg.down_dims),
            kernel_size=unet_cfg.kernel_size,
            n_groups=unet_cfg.n_groups,
            diffusion_step_embed_dim=unet_cfg.diffusion_step_embed_dim,
            use_film_scale_modulation=unet_cfg.use_film_scale_modulation
        )
        
        scheduler_cfg = cfg.policy.noise_scheduler
        scheduler_cls = NOISE_SCHEDULER_REGISTRY[scheduler_cfg.type]
        self.noise_scheduler = scheduler_cls(
            num_train_timesteps=scheduler_cfg.num_train_timesteps,
            beta_start=scheduler_cfg.beta_start,
            beta_end=scheduler_cfg.beta_end,
            beta_schedule=scheduler_cfg.beta_schedule,
            clip_sample=scheduler_cfg.clip_sample,
            clip_sample_range=scheduler_cfg.clip_sample_range,
            prediction_type=scheduler_cfg.prediction_type
        )
        self.num_inference_steps = scheduler_cfg.num_inference_steps

    def get_optimizer(self):
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.config.optimizer_lr,
            betas=list(self.config.optimizer_betas),
            eps=self.config.optimizer_eps,
            weight_decay=self.config.optimizer_weight_decay,
        )

    def get_scheduler(self, optimizer, num_training_steps):
        return get_scheduler(
            name=self.config.scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=self.config.scheduler_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def compute_loss(self, obs_dict, action):
        obs_dict = self.normalize_inputs(obs_dict)
        action = self.normalize_targets({"action": action})["action"]
        
        B = action.shape[0]
        device = action.device
        
        global_cond = self.vision_encoder(obs_dict)
        global_cond = global_cond.view(B, -1)
        
        noise = torch.randn_like(action)
        
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (B,), device=device
        ).long()
        
        noisy_actions = self.noise_scheduler.add_noise(action, noise, timesteps)
        
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=global_cond)
        
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = action
        else:
            raise ValueError(f"Unknown prediction_type: {self.noise_scheduler.config.prediction_type}")
            
        loss = F.mse_loss(noise_pred, target)
        return loss

    def generate_actions(self, obs_dict):
        device = next(self.parameters()).device
        B = obs_dict["observation.images.top"].shape[0]
        
        global_cond = self.vision_encoder(obs_dict)
        global_cond = global_cond.view(B, -1)
        
        sample = torch.randn(
            (B, self.pred_horizon, self.action_dim), 
            device=device
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
        for k in self.noise_scheduler.timesteps:
            k = k.to(device) if torch.is_tensor(k) else torch.tensor(k, device=device)
            noise_pred = self.noise_pred_net(
                sample=sample, 
                timestep=k,
                global_cond=global_cond
            )
            sample = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=sample
            ).prev_sample
            
        return sample

    @torch.no_grad()
    def select_action(self, obs_dict):
        self.eval()
        
        if self.use_delta_action:
            unnormalized_state = obs_dict["observation.state"][:, -1, :].clone()
            
        obs_dict = self.normalize_inputs(obs_dict)
        
        if len(self._action_queue) == 0:
            actions = self.generate_actions(obs_dict)
            
            actions = self.unnormalize_outputs({"action": actions})["action"]
            
            if self.use_delta_action:
                actions = actions + unnormalized_state.unsqueeze(1)
                
            self._latest_full_pred = actions.clone()
            
            start = self.obs_horizon - 1
            end = start + self.action_horizon
            actions = actions[:, start:end]
            
            action_seq = actions.squeeze(0).cpu().numpy()
            self._action_queue.extend(action_seq)
            
        return self._action_queue.popleft()
