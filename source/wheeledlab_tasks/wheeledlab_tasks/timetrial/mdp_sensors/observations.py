from __future__ import annotations

import torch
import torchvision.transforms as transforms
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera
from isaaclab.envs.mdp import *
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# ImageNet statistics
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean, std)
grayscale = transforms.Grayscale()
gray_normalize = transforms.Normalize([0.5], [0.5])
gray_transform = transforms.Compose([grayscale, gray_normalize])

color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.2, saturation=0.8, hue=0.5)
sharpness = transforms.RandomAdjustSharpness(sharpness_factor=2)
gaussian_blur = transforms.GaussianBlur(5, sigma=(0.1, 5.0))

def lidar_ranges(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The ranges from the given lidar sensor."""
    # extract the used quantities (to enable type-hinting)
    sensor = env.scene.sensors[sensor_cfg.name]
    lidar_ranges = sensor.data.output["linear_depth"]
    return lidar_ranges

def lidar_ranges_normalized(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return normalized lidar ranges with Gaussian noise."""
    # Extract the lidar sensor from the scene
    # sensor: Lidar = env.scene.sensors[sensor_cfg.name] # BUG: BROKEN IMPORT
    sensor = env.scene.sensors[sensor_cfg.name]

    lidar_ranges = sensor.data.output["linear_depth"]

    # Get the min and max range from the sensor configuration
    min_range = sensor.cfg.min_range  # Minimum possible range
    max_range = sensor.cfg.max_range  # Maximum possible range

    # Generate Gaussian noise with the same shape as the lidar data
    mean = 0.0  # Mean of the Gaussian distribution
    std = 0.1  # Standard deviation of the Gaussian distribution
    gaussian_noise = torch.normal(mean=mean, std=std, size=lidar_ranges.shape, device=lidar_ranges.device)

    # Add noise to the lidar data
    lidar_ranges_noisy = lidar_ranges + gaussian_noise

    # Clip the noisy lidar data to the min and max range
    lidar_ranges_noisy = torch.clip(lidar_ranges_noisy, min=min_range, max=max_range)

    # Normalize the noisy lidar data
    lidar_ranges_normalized = (lidar_ranges_noisy - min_range) / (max_range - min_range)

    return lidar_ranges_normalized

def camera_data_rgb(env: ManagerBasedEnv, sensor_cfg:SceneEntityCfg) -> torch.Tensor:
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    return sensor.data.output["rgb"]

def camera_data_rgb_flattened(env: ManagerBasedEnv, sensor_cfg:SceneEntityCfg) -> torch.Tensor:
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    images = sensor.data.output["rgb"]
    B, H, W, C = images.shape
    images = images[:, H//3:, :, :]
    H = H - H // 3
    images = images.permute(0, 3, 1, 2).float()
    normalized_imgs = gray_normalize(grayscale(images)/255.)
    normalized_imgs = normalized_imgs.reshape(B, -1)
    return normalized_imgs

def camera_data_rgb_flattened_aug(env: ManagerBasedEnv, sensor_cfg:SceneEntityCfg) -> torch.Tensor:
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    images = sensor.data.output["rgb"]
    B, H, W, C = images.shape
    images = images[:, H//3:, :, :]
    H -= H // 3
    images = images.permute(0, 3, 1, 2).float() / 255.
    images = color_jitter(images)
    # images = sharpness(images)
    images = gaussian_blur(images)
    normalized_imgs = gray_normalize(grayscale(images))
    normalized_imgs = normalized_imgs.reshape(B, -1)
    return normalized_imgs

def camera_data_depth(env:ManagerBasedEnv, sensor_cfg:SceneEntityCfg) -> torch.Tensor:
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    return sensor.data.output["distance_to_image_plane"]

def raycast_depth(env:ManagerBasedEnv, sensor_cfg:SceneEntityCfg) -> torch.Tensor:
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    return sensor.data.output["distance_to_image_plane"]