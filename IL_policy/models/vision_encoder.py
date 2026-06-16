import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from omegaconf import DictConfig

class SpatialSoftmax(nn.Module):
    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w),
            np.linspace(-1.0, 1.0, self._in_h),
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features):
        if self.nets is not None:
            features = self.nets(features)

        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)
        return feature_keypoints


def _replace_bn_with_gn(module):
    """Replace all BatchNorm2d with GroupNorm (num_groups = num_features // 16)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(
                num_groups=child.num_features // 16,
                num_channels=child.num_features,
            ))
        else:
            _replace_bn_with_gn(child)
    return module


def _get_output_shape(module, input_shape):
    with torch.no_grad():
        dummy = torch.zeros(input_shape)
        output = module(dummy)
    return output.shape


class DiffusionRgbEncoder(nn.Module):
    """ResNet18 + GroupNorm + SpatialSoftmax encoder for Diffusion Policy."""
    def __init__(
        self,
        crop_shape: tuple[int, int] = (216, 216),
        spatial_softmax_num_keypoints: int = 32,
    ):
        super().__init__()
        self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
        self.random_crop = torchvision.transforms.RandomCrop(crop_shape)

        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))
        _replace_bn_with_gn(self.backbone)

        dummy_h, dummy_w = crop_shape
        feature_map_shape = _get_output_shape(self.backbone, (1, 3, dummy_h, dummy_w))[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=spatial_softmax_num_keypoints)
        self.feature_dim = spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(self.feature_dim, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = self.random_crop(x)
        else:
            x = self.center_crop(x)

        x = self.backbone(x)
        x = torch.flatten(self.pool(x), start_dim=1)
        x = self.relu(self.out(x))
        return x


class VisionEncoder(nn.Module):
    """
    Extracts features from top and front images, and concatenates with proprioception.
    Output becomes the global conditioning vector for the 1D U-Net.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        crop_shape = tuple(cfg.crop_shape)
        num_kp = cfg.policy.spatial_softmax_num_keypoints
        state_dim = cfg.task.state_dim
        
        self.encoder_top = DiffusionRgbEncoder(crop_shape=crop_shape, spatial_softmax_num_keypoints=num_kp)
        self.encoder_front = DiffusionRgbEncoder(crop_shape=crop_shape, spatial_softmax_num_keypoints=num_kp)
        
        # 32 keypoints * 2 (x,y) * 2 cameras + 3 proprioception = 131 dimensions
        self.out_dim = self.encoder_top.feature_dim + self.encoder_front.feature_dim + state_dim

    def forward(self, obs_dict):
        """
        obs_dict:
          observation.images.top: (B, T, C, H, W)
          observation.images.front: (B, T, C, H, W)
          observation.state: (B, T, state_dim)
        Returns:
          (B, T, out_dim) feature vector
        """
        B, T, C, H, W = obs_dict["observation.images.top"].shape
        
        img_top = obs_dict["observation.images.top"].view(B*T, C, H, W)
        img_front = obs_dict["observation.images.front"].view(B*T, C, H, W)
        
        feat_top = self.encoder_top(img_top)
        feat_front = self.encoder_front(img_front)
        
        feat_top = feat_top.view(B, T, -1)
        feat_front = feat_front.view(B, T, -1)
        proprio = obs_dict["observation.state"]
        
        features = torch.cat([feat_top, feat_front, proprio], dim=-1)
        return features
