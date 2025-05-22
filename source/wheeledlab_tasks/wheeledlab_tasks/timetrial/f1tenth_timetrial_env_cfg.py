from collections import Counter
import os
import time
import torch
import random
import numpy as np
from scipy.spatial.transform import Rotation as R

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, RigidObject, AssetBaseCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    CurriculumTermCfg as CurrTerm,
    SceneEntityCfg,
)
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs import ManagerBasedRLEnvCfg

from wheeledlab.envs.mdp import increase_reward_weight_over_time
from wheeledlab_assets import WHEELEDLAB_ASSETS_DATA_DIR
from wheeledlab_assets.mushr import MUSHR_SUS_CFG
from wheeledlab_assets.f1tenth import F1TENTH_CFG
from wheeledlab_tasks.common import Mushr4WDActionCfg
from wheeledlab_tasks.common import F1Tenth4WDActionCfg
from .disable_lidar import disable_all_lidars

from .utils import create_maps_from_png, generate_random_poses, TraversabilityHashmapUtil, WaypointsUtil
from . import mdp_sensors
from .mdp import reset_root_state

import omni.usd


##############################
###### OBSERVATION #######
##############################

def base_lin_vel_xy(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame. 2D, only x and y"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:,0:2]

def base_ang_vel_z(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame. Only z, yaw rade"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    return asset.data.root_ang_vel_b[:,2].unsqueeze(-1)

def deviation_from_waypoints(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lookahead: int = 5
) -> torch.Tensor:
    """Calculate signed lateral deviation from the reference waypoint path.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.
        lookahead: Number of waypoints to look ahead for track direction.
        
    Returns:
        Tensor of signed deviations (positive = left of reference, negative = right).
    """
    # Get current state
    asset = env.scene[asset_cfg.name]
    pos_xy_world = asset.data.root_pos_w[:, :2]
    
    # Get waypoints (ensure float32 for consistency)
    waypoints_xy_world = torch.tensor(
        env.scene.terrain.cfg.waypoints, 
        device=env.device,
        dtype=torch.float32
    )[:, :2]

    inner_xy_world = torch.tensor(
        env.scene.terrain.cfg.inner, 
        device=env.device,
        dtype=torch.float32
    )[:, :2]

    # Find nearest waypoint (use inner trackbound )
    current_idx, _ = find_nearest_waypoint(inner_xy_world, pos_xy_world)
    current_waypoint = waypoints_xy_world[current_idx]
    
    # Get lookahead waypoint for track direction
    next_idx = (current_idx + lookahead) % len(waypoints_xy_world)
    next_waypoint = waypoints_xy_world[next_idx]
    
    # Vector from current waypoint to next waypoint (track direction)
    track_dir = next_waypoint - current_waypoint
    # Vector from current waypoint to car position
    car_offset = pos_xy_world - current_waypoint
    
    # Cross product (track_dir × car_offset) determines left/right sign
    cross_product = track_dir[:, 0] * car_offset[:, 1] - track_dir[:, 1] * car_offset[:, 0]
    sign = torch.sign(cross_product)
    
    # Absolute distance to nearest waypoint (lateral deviation magnitude)
    distance = torch.norm(car_offset, dim=1)
    
    # Signed lateral deviation (positive = left, negative = right)
    signed_deviation = sign * distance
    
    return signed_deviation.unsqueeze(-1)


def heading_error_from_waypoints(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lookahead: int = 5
) -> torch.Tensor:
    """
    Calculate heading error between car's current orientation and direction to lookahead waypoint.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        lookahead: Number of waypoints to look ahead (default 30)
        
    Returns:
        Tensor of heading errors in radians (range [-π, π])
    """
    # Get current state
    asset = env.scene[asset_cfg.name]
    pos_xy_world = asset.data.root_pos_w[:, :2]
    heading_w = asset.data.heading_w
    
    # Get waypoints (ensure float32 for consistency)
    waypoints_xy_world = torch.tensor(
        env.scene.terrain.cfg.waypoints, 
        device=env.device,
        dtype=torch.float32
    )[:, :2]
    
    inner_xy_world = torch.tensor(
        env.scene.terrain.cfg.inner, 
        device=env.device,
        dtype=torch.float32
    )[:, :2]

    # Find nearest waypoint (use inner trackbound )
    current_idx, _ = find_nearest_waypoint(inner_xy_world, pos_xy_world)
    current_waypoint = waypoints_xy_world[current_idx]
    
    # Get lookahead waypoint (with circular buffer handling)
    next_idx = (current_idx + lookahead) % len(waypoints_xy_world)
    next_waypoint = waypoints_xy_world[next_idx]
    
    # Calculate desired heading vector
    desired_heading_w = torch.atan2(
        next_waypoint[:, 1] - pos_xy_world[:, 1],
        next_waypoint[:, 0] - pos_xy_world[:, 0]
    )
    
    # Calculate smallest angle difference (normalized to [-π, π])
    heading_error = torch.atan2(
        torch.sin(desired_heading_w - heading_w),
        torch.cos(desired_heading_w - heading_w)
    )
    
    return heading_error.unsqueeze(-1)


def deviation_centerline_horizon(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lookahead: int = 10,
    horizon: int = 5
) -> torch.Tensor:
    """
    Calculate signed lateral deviations from multiple reference waypoint segments.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        lookahead: Base number of waypoints to look ahead
        horizon: Number of lookahead points to return
        
    Returns:
        Tensor of signed deviations (positive = left of reference, negative = right)
        for each horizon segment. Shape: [num_envs, horizon]
    """
    # Get current state
    asset = env.scene[asset_cfg.name]
    pos_xy_world = asset.data.root_pos_w[:, :2]
    
    # Get waypoints
    waypoints_xy_world = torch.tensor(
        env.scene.terrain.cfg.waypoints,
        device=env.device,
        dtype=torch.float32
    )[:, :2]
    
    inner_xy_world = torch.tensor(
        env.scene.terrain.cfg.inner,
        device=env.device,
        dtype=torch.float32
    )[:, :2]

    # Find nearest waypoint
    current_idx, _ = find_nearest_waypoint(inner_xy_world, pos_xy_world)  # shape: [num_envs]
    
    # Create horizon indices [num_envs, horizon + 1]
    steps = torch.linspace(0, lookahead * horizon, horizon + 1, device=env.device)
    horizon_indices = (current_idx.unsqueeze(-1) + steps.unsqueeze(0)) % len(waypoints_xy_world)
    horizon_indices = horizon_indices.long()
    
    # Get waypoint pairs for each segment [num_envs, horizon, 2, 2]
    segment_starts = waypoints_xy_world[horizon_indices[:, :-1]]
    segment_ends = waypoints_xy_world[horizon_indices[:, 1:]]
    
    # Calculate track directions [num_envs, horizon, 2]
    track_dirs = segment_ends - segment_starts
    
    # Calculate car offsets [num_envs, horizon, 2]
    car_offsets = pos_xy_world.unsqueeze(1) - segment_starts
    
    # Cross products (track_dir × car_offset) [num_envs, horizon]
    cross_products = (track_dirs[:, :, 0] * car_offsets[:, :, 1] - 
                     track_dirs[:, :, 1] * car_offsets[:, :, 0])
    signs = torch.sign(cross_products)
    
    # Perpendicular distances [num_envs, horizon]
    segment_lengths = torch.norm(track_dirs, dim=2)
    distances = torch.abs(cross_products) / (segment_lengths + 1e-6)
    
    # Signed deviations
    signed_deviations = signs * distances
    
    return signed_deviations

def heading_error_horizon(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lookahead: int = 10,
    horizon: int = 5
) -> torch.Tensor:
    """
    Calculate heading errors between car's current orientation and multiple lookahead waypoints.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        lookahead: Base number of waypoints to look ahead
        horizon: Number of lookahead points to return
        
    Returns:
        Tensor of heading errors in radians (range [-π, π]) for each horizon point.
        Shape: [num_envs, horizon]
    """
    # Get current state
    asset = env.scene[asset_cfg.name]
    pos_xy_world = asset.data.root_pos_w[:, :2]
    heading_w = asset.data.heading_w
    
    # Get waypoints
    waypoints_xy_world = torch.tensor(
        env.scene.terrain.cfg.waypoints, 
        device=env.device,
        dtype=torch.float32
    )[:, :2]
    
    inner_xy_world = torch.tensor(
        env.scene.terrain.cfg.inner,
        device=env.device,
        dtype=torch.float32
    )[:, :2]

    # Find nearest waypoint
    current_idx, _ = find_nearest_waypoint(inner_xy_world, pos_xy_world)  # shape: [num_envs]
    
    # Create lookahead indices for all environments
    lookahead_steps = torch.linspace(lookahead, lookahead*horizon, horizon, device=env.device)
    horizon_indices = (current_idx.unsqueeze(-1) + lookahead_steps.unsqueeze(0)) % len(waypoints_xy_world)
    horizon_indices = horizon_indices.long()
    
    # Get lookahead waypoints [num_envs, horizon, 2]
    lookahead_points = waypoints_xy_world[horizon_indices]
    
    # Calculate desired heading vectors [num_envs, horizon]
    desired_headings = torch.atan2(
        lookahead_points[:, :, 1] - pos_xy_world[:, 1].unsqueeze(-1),
        lookahead_points[:, :, 0] - pos_xy_world[:, 0].unsqueeze(-1)
    )
    
    # Calculate smallest angle differences [num_envs, horizon]
    heading_errors = torch.atan2(
        torch.sin(desired_headings - heading_w.unsqueeze(-1)),
        torch.cos(desired_headings - heading_w.unsqueeze(-1))
    )
    
    return heading_errors

def d_lat_horizon(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lookahead: int = 10,
    horizon: int = 5
) -> torch.Tensor:
    """
    Calculate heading error between car's current orientation and direction to lookahead waypoint.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        interval_lookahead: Number of waypoints to look ahead (default 10)
        horizon: horizon length of lookaheads (total horizon = lookahead * horizon)
        
    Returns:
        Tensor of normalized lateral space from centerline to trackbounds to the left (inner) and right (outer)
    """
    # Get current state
    asset = env.scene[asset_cfg.name]
    pos_xy_world = asset.data.root_pos_w[:, :2]
    
    # Get waypoints (ensure float32 for consistency)
    waypoints_xy_world = torch.tensor(
        env.scene.terrain.cfg.waypoints, 
        device=env.device,
        dtype=torch.float32
    )[:, :2]
    
    inner_xy_world = torch.tensor(
        env.scene.terrain.cfg.inner, 
        device=env.device,
        dtype=torch.float32
    )[:, :2]

    d_lat = torch.tensor(
        env.scene.terrain.cfg.d_lat, 
        device=env.device,
        dtype=torch.float32
    )

    # Find nearest waypoint (use inner trackbound)
    current_idx, _ = find_nearest_waypoint(inner_xy_world, pos_xy_world)  # shape: [num_envs]
    
    # Create lookahead indices for all environments
    lookahead_horizon = torch.linspace(lookahead, lookahead*horizon, horizon, device=env.device)  # shape: [horizon]
    
    # Add current_idx (num_envs, 1) with lookahead_horizon (1, horizon) to get (num_envs, horizon)
    next_idx_horizon = (current_idx.unsqueeze(-1) + lookahead_horizon.unsqueeze(0)) % len(waypoints_xy_world)
    next_idx_horizon = next_idx_horizon.long()  # Convert to integer indices
    
    # Gather d_lat values - output shape will be [num_envs, horizon, 2]
    next_d_lat_horizon = d_lat[next_idx_horizon, :]
    
    # Reshape to [num_envs, horizon * 2] for concatenation with other observations
    return next_d_lat_horizon.reshape(-1, horizon * 2)

def centerline_absolute_heading_horizon(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lookahead: int = 10,
    horizon: int = 5,
    angle_window: int = 3  # Number of waypoints to use for angle calculation
) -> torch.Tensor:
    """
    Calculate the relative heading angles of the centerline at horizon points.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        lookahead: Base number of waypoints to look ahead
        horizon: Number of lookahead points to return
        angle_window: Number of waypoints to use for calculating each heading angle
        
    Returns:
        Tensor of normalized heading angles (range [-1, 1] where 1 = π radians)
    """
    # Get current state
    asset = env.scene[asset_cfg.name]
    pos_xy_world = asset.data.root_pos_w[:, :2]
    
    # Get waypoints
    waypoints_xy_world = torch.tensor(
        env.scene.terrain.cfg.waypoints,
        device=env.device,
        dtype=torch.float32
    )[:, :2]
    
    inner_xy_world = torch.tensor(
        env.scene.terrain.cfg.inner,
        device=env.device,
        dtype=torch.float32
    )[:, :2]

    # Find nearest waypoint
    current_idx, _ = find_nearest_waypoint(inner_xy_world, pos_xy_world)  # shape: [num_envs]
    
    # Create lookahead indices
    lookahead_horizon = torch.linspace(lookahead, lookahead*horizon, horizon, device=env.device)
    horizon_indices = (current_idx.unsqueeze(-1) + lookahead_horizon.unsqueeze(0)) % len(waypoints_xy_world)
    horizon_indices = horizon_indices.long()
    
    # Calculate heading angles using a window of waypoints for smoother results
    window_indices = torch.arange(-angle_window//2, angle_window//2 + 1, device=env.device)
    all_indices = (horizon_indices.unsqueeze(-1) + window_indices.unsqueeze(0).unsqueeze(0)) % len(waypoints_xy_world)
    
    # Get waypoint positions for all windows [num_envs, horizon, window_size, 2]
    window_waypoints = waypoints_xy_world[all_indices]
    
    # Calculate vectors between first and last points in each window
    vectors = window_waypoints[:, :, -1, :] - window_waypoints[:, :, 0, :]
    
    # Calculate heading angles [num_envs, horizon]
    heading_angles = torch.atan2(vectors[:, :, 1], vectors[:, :, 0])
    
    # Normalize angles to [-1, 1] range where 1 = π radians
    normalized_angles = heading_angles / torch.pi
    
    return normalized_angles.reshape(-1, horizon)  # [num_envs, horizon]

def centerline_relative_heading_horizon(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lookahead: int = 10,
    horizon: int = 5,
    angle_window: int = 1  # Smoothing window
) -> torch.Tensor:
    """Calculate the *relative* heading change between consecutive horizon points."""
    # Get waypoints
    waypoints_xy_world = torch.tensor(
        env.scene.terrain.cfg.waypoints,
        device=env.device,
        dtype=torch.float32
    )[:, :2]
    
    # Find nearest waypoint
    asset = env.scene[asset_cfg.name]
    pos_xy_world = asset.data.root_pos_w[:, :2]
    inner_xy_world = torch.tensor(
        env.scene.terrain.cfg.inner,
        device=env.device,
        dtype=torch.float32
    )[:, :2]
    current_idx, _ = find_nearest_waypoint(inner_xy_world, pos_xy_world)  # [num_envs]
    
    # Get horizon indices [num_envs, horizon + 1] (extra point for differences)
    lookahead_steps = torch.linspace(0, lookahead * horizon, horizon + 1, device=env.device)
    horizon_indices = (current_idx.unsqueeze(-1) + lookahead_steps.unsqueeze(0)) % len(waypoints_xy_world)
    horizon_indices = horizon_indices.long()
    
    # Compute smoothed headings for each point [num_envs, horizon + 1]
    window_indices = torch.arange(-angle_window//2, angle_window//2 + 1, device=env.device)
    all_indices = (horizon_indices.unsqueeze(-1) + window_indices.unsqueeze(0).unsqueeze(0)) % len(waypoints_xy_world)
    window_waypoints = waypoints_xy_world[all_indices]  # [num_envs, horizon+1, window_size, 2]
    
    # Vectors from first to last point in each window
    vectors = window_waypoints[:, :, -1, :] - window_waypoints[:, :, 0, :]
    headings = torch.atan2(vectors[:, :, 1], vectors[:, :, 0])  # [num_envs, horizon + 1]
    
    # Compute relative heading changes (delta between consecutive points)
    relative_headings = headings[:, 1:] - headings[:, :-1]  # [num_envs, horizon]
    
    # Normalize to [-1, 1] (1 = π radians)
    normalized = relative_headings / torch.pi
    
    return normalized.reshape(-1, horizon)  # [num_envs, horizon]

def centerline_features_horizon(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lookahead: int = 10,
    horizon: int = 5,
    angle_window: int = 3
) -> torch.Tensor:
    """
    Returns BOTH relative and absolute heading features.
    Output shape: [num_envs, horizon * 4] where each point has:
    - [0]: Normalized absolute heading (∈ [-1, 1])
    - [1]: Normalized relative heading (∈ [-1, 1])
    - [2:4]: Waypoint XY offset (normalized)
    """
    # Get waypoints and current state
    waypoints = torch.tensor(env.scene.terrain.cfg.waypoints, device=env.device, dtype=torch.float32)[:, :2]
    asset = env.scene[asset_cfg.name]
    pos_xy = asset.data.root_pos_w[:, :2]
    
    # Find nearest waypoint [num_envs]
    current_idx, _ = find_nearest_waypoint(
        torch.tensor(env.scene.terrain.cfg.inner, device=env.device)[:, :2],
        pos_xy
    )
    
    # Horizon indices [num_envs, horizon + 1]
    steps = torch.linspace(0, lookahead * horizon, horizon + 1, device=env.device)
    idxs = (current_idx.unsqueeze(-1) + steps.unsqueeze(0)) % len(waypoints)
    idxs = idxs.long()
    
    # Smoothed headings [num_envs, horizon + 1]
    window = torch.arange(-angle_window//2, angle_window//2 + 1, device=env.device)
    wp_window = waypoints[(idxs.unsqueeze(-1) + window.unsqueeze(0).unsqueeze(0)) % len(waypoints)]
    vectors = wp_window[:, :, -1] - wp_window[:, :, 0]  # [num_envs, horizon+1, 2]
    headings = torch.atan2(vectors[:, :, 1], vectors[:, :, 0])
    
    # Features
    abs_headings = headings[:, :-1] / torch.pi  # [num_envs, horizon]
    rel_headings = (headings[:, 1:] - headings[:, :-1]) / torch.pi  # [num_envs, horizon]
    xy_offsets = waypoints[idxs[:, :-1]] - pos_xy.unsqueeze(1)  # [num_envs, horizon, 2]
    
    # Normalize XY offsets by track width
    d_lat = torch.tensor(env.scene.terrain.cfg.d_lat, device=env.device).mean(dim=1).max()
    xy_offsets = xy_offsets / (d_lat + 1e-6)
    
    # Concatenate and reshape [num_envs, horizon * 4]
    features = torch.cat([
        abs_headings.unsqueeze(-1),  # [num_envs, horizon, 1]
        rel_headings.unsqueeze(-1),  # [num_envs, horizon, 1]
        xy_offsets                  # [num_envs, horizon, 2]
    ], dim=-1)
    return features.reshape(features.shape[0], -1)  # Flatten last two dims

HORIZON = 10
LOOKAHEAD = 20
@configclass
class F1TenthTimeTrialObsCfg:
    """Observation specifications for the environment."""
    @configclass
    class PolicyCfg(ObsGroup):
        """
        [vx, vy, vz, wx, wy, wz, action1(vel), action2(steering)]
        """
        # lidar = ObsTerm(func=mdp_sensors.lidar_ranges, params={"sensor_cfg":SceneEntityCfg("lidar")})
        base_lin_vel_xy = ObsTerm(func=base_lin_vel_xy, noise=Unoise(n_min=-.0, n_max=.0))
        base_ang_vel_z = ObsTerm(func=base_ang_vel_z, noise=Unoise(n_min=-.0, n_max=.0))
        last_action = ObsTerm(
            func=mdp.last_action,
            clip=(-1., 1.), # TODO: get from ClipAction wrapper
            noise=Unoise(0, 0)
        )

        heading_error = ObsTerm(
            func=heading_error_horizon,
            params={'lookahead': LOOKAHEAD,
                    'horizon': HORIZON}
        )
        deviation_error = ObsTerm(
            func=deviation_centerline_horizon,
            params={'lookahead': LOOKAHEAD,
                    'horizon': HORIZON}
        )
        d_lat_horizon = ObsTerm(
            func=d_lat_horizon,
            params={'lookahead': LOOKAHEAD,
                    'horizon': HORIZON}
        )
        centerline_horizon = ObsTerm(
            func=centerline_relative_heading_horizon,
            params={'lookahead': LOOKAHEAD,
                    'horizon': HORIZON}
        )
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True


    policy: PolicyCfg = PolicyCfg()

@configclass
class InitialPoseCfg:
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot_euler_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)


##############################
###### TERRAIN / TRACK #######
##############################

@configclass
class F1TenthTimeTrialTerrainImporterCfg(TerrainImporterCfg):
    # map generation parameters

    # generate a colored plane geometry
    MAP_NAME = 'THETRACK'
    file_name = os.path.join(WHEELEDLAB_ASSETS_DATA_DIR, 'maps', MAP_NAME+'.usd')
    maps_folder_path = '/home/tongo/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/timetrial/utils/maps'    
    traversability_hashmap, waypoints, outer, inner, d_lat, psi_rad, kappa_radpm, vx_mps, spacing_meters, map_size_pixels = create_maps_from_png(maps_folder_path, MAP_NAME, file_name, visualization_scale=0.5)

    spacing = spacing_meters
    row_spacing, col_spacing = spacing

    num_cols, num_rows = map_size_pixels
    map_size = (num_rows+1, num_cols+1)

    # environments size are generated in a grid
    env_size = map_size


    width = num_rows * row_spacing
    height = num_cols * col_spacing
    
    """
    traversability_hashmap is a 2D numpy array of traversability values, 1.0 or 0.0
    shape: [num_rows, num_cols]
    """

    # Defined logic for driveable track/out of bounds
    # traversability_hashmap, env_boundaries = create_oval(
    #     file_name, map_size, spacing, env_size
    # )


    prim_path = "/World/ground"
    terrain_type="usd"
    usd_path = file_name
    collision_group = -1
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    )
    debug_vis=False
    
    def generate_poses_from_init_points(self, env : ManagerBasedEnv, env_ids : torch.Tensor):
        # temp = self.init_points[0][0]
        # self.init_points = [[(temp[0], temp[1]) for _ in range(len(row))] for row in self.init_points]
        env_ids = random.choices(range(0, int(self.num_rows / self.env_num_rows) * int(self.num_cols / self.env_num_cols)), k=env_ids.shape[0])

        counts = Counter(env_ids)

        sampled = {}

        for env_id, count in counts.items():
            # random.sample automatically picks unique items without replacement
            sampled_poses_in_env = random.sample(self.init_points[int(env_id)], count)

            env_x = env_id // (self.num_cols / self.env_num_cols)
            env_y = env_id % (self.num_cols / self.env_num_cols)
            center_x = env_x * self.env_num_rows * self.row_spacing + self.env_num_rows * self.row_spacing / 2
            center_y = env_y * self.env_num_cols * self.col_spacing + self.env_num_cols * self.col_spacing / 2

            sampled[env_id] = []
            for pose in sampled_poses_in_env:
                x = pose[0] * self.row_spacing - self.width / 2
                y = pose[1] * self.col_spacing - self.height / 2
                sampled[env_id].append(
                    InitialPoseCfg(
                        pos=(x, y, 0.1),
                        rot_euler_xyz_deg=(0., 0., np.rad2deg(np.arctan2(y - center_y, x - center_x)))
                    )
                )
        
        sample_ind = {}
        for env_id in sampled:
            sample_ind[env_id] = 0

        result = []
        for i in env_ids:
            result.append(sampled[i][sample_ind[i]])
            sample_ind[i] += 1
        
        return result

    def generate_random_poses(self, num_poses):
        # generate random initial poses with margin
        init_poses = generate_random_poses(num_poses, self.row_spacing, self.col_spacing, self.traversability_hashmap, self.waypoints, self.outer, self.inner, margin=0.1)
        valid_init_poses = [
            InitialPoseCfg(
                pos=(x, y, 0.02),
                rot_euler_xyz_deg=(0., 0., angle)
            ) for x, y, angle in init_poses
        ]
        return valid_init_poses

         
    """
    Get traversability value of an x, y coordinate
    """
    def get_traversability(self, poses):
        traversability = []
        xs, ys = poses[:, 0], poses[:, 1]
        x_idx, y_idx = self.get_map_id(xs, ys)
        traversability = torch.tensor(self.traversability_hashmap).to(x_idx.device)[x_idx, y_idx]
        return traversability
    
    """
    Helper function to get the map id given x, y coordinates
    """
    def get_map_id(self, x, y):
        x_idx = torch.floor((x + self.width/2 - self.row_spacing/2) / self.row_spacing).long()
        y_idx = torch.floor((y + self.height/2 - self.col_spacing/2) / self.col_spacing).long()
        x_idx = torch.clamp(x_idx, 0, self.num_rows-1)
        y_idx = torch.clamp(y_idx, 0, self.num_cols-1)
        return x_idx, y_idx

@configclass
class F1TenthTimeTrialSceneCfg(InteractiveSceneCfg):
    """Configuration for a Mushr car Scene with racetrack terrain and Sensors."""
    terrain = F1TenthTimeTrialTerrainImporterCfg()
    # waypoint_loader = WaypointLoader(map_name="f")
    # Add ground config (ground is slightly below terrain)
    ground = AssetBaseCfg(
        prim_path="/World/base",
        spawn = sim_utils.GroundPlaneCfg(size=(terrain.width, terrain.height),
                                         color=(0,0,0),
                                         physics_material=sim_utils.RigidBodyMaterialCfg(
                                            friction_combine_mode="multiply",
                                            restitution_combine_mode="multiply",
                                            static_friction=1.3,
                                            dynamic_friction=1.3,
                                         ),
        )
    )

    # # Add light configuration
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    robot: AssetBaseCfg = F1TENTH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # robot: ArticulationCfg = MUSHR_SUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ground.init_state.pos = (0.0, 0.0, -1e-4)

    def __post_init__(self):
        """Post intialization."""
        super().__post_init__()

        self.robot.init_state = self.robot.init_state.replace(
            pos=(0.0, 0.0, 0.0)
        )

#####################
###### EVENTS #######
#####################
def store_data(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: RigidObject = env.scene[asset_cfg.name]
    waypoints = torch.tensor(env.scene.terrain.cfg.waypoints, 
                        device=env.device)[:, :2]
    position_xy = asset.data.root_pos_w[:, :2]

    # Find nearest waypoint (vectorized)
    current_idx, _ = find_nearest_waypoint(waypoints, position_xy)
    num_waypoints = len(waypoints)

    #there should be a more elegant way to store these...
    env.extras['v_x'] = asset.data.root_lin_vel_b[:,0]
    env.extras['s_idx'] = current_idx.clone()
    env.extras['time'] = torch.tensor(env.sim.current_time, device=env.device)
    env.extras['s_idx_max'] = torch.tensor(num_waypoints-1, device=env.device)

@configclass
class F1TenthTimeTrialEventsCfg:
    # on startup

    reset_root_state = EventTerm(
        func=reset_root_state,
        mode="reset",
    )

    # store_data = EventTerm( 
    #     func= store_data,
    #     mode="interval",
    #     interval_range_s=(0.025, 0.025),
    #     params={
    #     },
    # )



@configclass
class F1TenthTimeTrialEventsRandomCfg(F1TenthTimeTrialEventsCfg):
    # change_wheel_friction = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "static_friction_range": (0.0, 0.0),
    #         "dynamic_friction_range": (0.0, 0.0),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 10,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_.*link"),
    #         "make_consistent": False,
    #     },
    # )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
    #         "mass_distribution_params": (0.0, 0.0),
    #         "operation": "abs",
    #     },
    # )

    # add_wheel_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_.*link"),
    #         "mass_distribution_params": (.0, 0.0),
    #         "operation": "abs",
    #     },
    # )

    # Override randomize_gains to target all four wheel motors (front and back)
    # randomize_gains = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_(back|front)_.*"]),
    #         "damping_distribution_params": (0.0, 0.0),
    #         "operation": "abs",
    #     },
    # )

    # change_wheel_friction = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "static_friction_range": (0.0, 0.0),
    #         "dynamic_friction_range": (0.0, 0.0),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 20,
    #         "asset_cfg": SceneEntityCfg("robot", body_names="wheel.*"),
    #         "make_consistent": True,
    #     },
    # )

    kill_lidar = EventTerm(
        func=disable_all_lidars,
        mode="startup",
        params={}          
    )


######################
###### REWARDS #######
######################
def is_traversable(env):
    poses = mdp.root_pos_w(env)[..., :2]
    traversability = TraversabilityHashmapUtil().get_traversability(poses)
    return torch.where(traversability, 1., 0.)

def traversable_reward(env):
    poses = mdp.root_pos_w(env)[..., :2]
    traversability = TraversabilityHashmapUtil().get_traversability(poses)
    return torch.where(traversability, 1, 0.)

def out_of_track_penalty(env):
    poses = mdp.root_pos_w(env)[..., :2]
    traversability = TraversabilityHashmapUtil().get_traversability(poses)
    return torch.where(traversability, 0., -1.)

def bool_is_not_traversable(env):
    num_episodes = env.common_step_counter // env.max_episode_length
    # delay the penalty for the first 300 episodes
    if num_episodes < 10:
        return torch.zeros(env.num_envs, device=env.device) == 1

    traversability = is_traversable(env)
    return traversability == 0

def is_traversable_speed_scaled(env):
    return is_traversable(env) * mdp.base_lin_vel(env)[:, 0]

# def is_traversable_wheels(env):
#     body_asset_cfg = SceneEntityCfg("robot", body_names=".*wheel_link")
#     body_asset_cfg.resolve(env.scene)
#     # B x num_body x 2
#     poses = env.scene[body_asset_cfg.name].data.body_pos_w[:, body_asset_cfg.body_ids][:, :, :2]
#     B, num_body = poses.shape[:2]
#     # terrain = env.scene[SceneEntityCfg("terrain").name]
#     traversability = TraversabilityHashmapUtil().get_traversability(poses.reshape(-1, 2)).reshape(B, num_body)
#     return torch.where(traversability == 1, 1., -5.).sum(dim=-1)

# def binary_is_traversable_wheels(env):
#     body_asset_cfg = SceneEntityCfg("robot", body_names=".*wheel_link")
#     body_asset_cfg.resolve(env.scene)
#     # B x num_body x 2
#     poses = env.scene[body_asset_cfg.name].data.body_pos_w[:, body_asset_cfg.body_ids][:, :, :2]
#     B, num_body = poses.shape[:2]
#     # terrain = env.scene[SceneEntityCfg("terrain").name]
#     traversability = TraversabilityHashmapUtil().get_traversability(poses.reshape(-1, 2)).reshape(B, num_body)
#     return torch.where(traversability == 1, 1., 0.).sum(dim=-1) == 0

def upright_penalty(env, thresh_deg):
    rot_mat = math_utils.matrix_from_quat(mdp.root_quat_w(env))
    up_dot = rot_mat[:, 2, 2]
    up_dot = torch.rad2deg(torch.arccos(up_dot))
    penalty = torch.where(up_dot > thresh_deg, up_dot - thresh_deg, 0.)
    return penalty

def vel_rew_trav(env, speed_target_on_trav: float=1., speed_target_on_non_trav: float=2.):
    poses = mdp.root_pos_w(env)[..., :2]
    terrain = env.scene[SceneEntityCfg("terrain").name]
    traversability = TraversabilityHashmapUtil().get_traversability(poses)
    speed_target = torch.where(traversability, speed_target_on_trav, speed_target_on_non_trav)

    lin_vel = mdp.base_lin_vel(env)
    speed_dist = -((torch.norm(lin_vel, dim=-1) - speed_target) ** 2) + speed_target ** 2
    return torch.where(speed_dist > 0., speed_dist, 0.) # speed target

def off_track(env, straight, corner_out_radius):
    poses = mdp.root_pos_w(env)
    penalty = torch.where(torch.abs(poses[...,1]) < straight,
                torch.where(torch.abs(poses[...,0]) > corner_out_radius, 1, 0),
                torch.where(poses[...,1] > 0,
                    torch.where((poses[...,1] - straight)**2 + poses[...,0]**2 > corner_out_radius**2, 1, 0),
                    torch.where((poses[...,1] + straight)**2 + poses[...,0]**2 > corner_out_radius**2, 1, 0)))
    return 

def low_speed_penalty(env, low_speed_thresh: float=0.3):
    lin_speed = torch.norm(mdp.base_lin_vel(env), dim=-1)
    pen = torch.where(lin_speed < low_speed_thresh, 1., 0.)
    return pen

def forward_vel(env):
    return mdp.base_lin_vel(env)[:, 0]

def progress_rew(env):
    """Reward +1 for passing each new waypoint, handling lap transitions."""
    progress_bool, progress = progress_waypoint_bool(env)
    return torch.where(progress_bool, progress*0.1, 0.0)

def progress_waypoint_bool(env):
    # Initialize buffer if first run
    if not hasattr(env, '_last_waypoint_indices'):
        env._last_waypoint_indices = torch.zeros(env.num_envs, 
                                               dtype=torch.long,
                                               device=env.device)
    
    # Get current positions and waypoints (single operation)
    position_xy = env.scene["robot"].data.root_pos_w[:, :2]
    waypoints = torch.tensor(env.scene.terrain.cfg.waypoints, 
                           device=env.device)[:, :2]
    
    # Find nearest waypoint (vectorized)
    current_idx, _ = find_nearest_waypoint(waypoints, position_xy)
    num_waypoints = len(waypoints)
    
    # Calculate progress (handles lap transitions)
    progress = (current_idx - env._last_waypoint_indices) % num_waypoints
    
    # Valid progress must be:
    # 1. Forward movement (progress > 0)
    # 2. Not cutting corners or reset far (progress <= 10)
    progress_bool = (progress > 0) & (progress <= 10)
    # Update indices to current_idx 


    #there should be a more elegant way to store these...
    env.extras['s_idx'] = current_idx.clone()
    env.extras['time'] = torch.tensor(env.sim.current_time, device=env.device)
    env.extras['s_idx_max'] = torch.tensor(num_waypoints-1, device=env.device)

    # Always update
    env._last_waypoint_indices = current_idx.clone()  
 
    return progress_bool, progress

def find_nearest_waypoint(waypoints: torch.Tensor,  # Shape: [M, 2] - M waypoints
                         positions: torch.Tensor, # [N,2] - N cars
                         ) -> tuple[int, torch.Tensor]:
    """
    Finds closest waypoints for all cars, handling circular track wrapping.
    If lookahead is None, searches all waypoints (accurate but slower).
    With lookahead, only checks next K waypoints from current closest (faster).
    """
    M = waypoints.shape[0]
    N = positions.shape[0]
    
    # First find rough closest without wrapping [N]
    diffs = positions.unsqueeze(1) - waypoints.unsqueeze(0)  # [N,M,2]
    dists = torch.norm(diffs, p=2, dim=2)  # [N,M]
    closest_idx = torch.argmin(dists, dim=1)  # [N]
    
    return closest_idx, dists[torch.arange(N), closest_idx]


####### F1TenthTimeTrial Environment #######
@configclass
class F1TenthTimeTrialRewardsCfg:
    # """Reward terms for the MDP."""
    # traversablility = RewTerm(
    #     func=traversable_reward,
    #     weight=1.,
    # )
    out_of_track = RewTerm(
        func=out_of_track_penalty,
        weight=1.,
    )

    vel_rew = RewTerm(
        func=forward_vel,
        weight= 0.,
    )

    progress_rew = RewTerm(
        func=progress_rew,
        weight=1.,
    )


########################
###### CURRICULUM ######
########################

@configclass
class TimeTrialCurriculumCfg:

    more_velocity = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "vel_rew",
            "increase": 0.0,
            "first_episode_increase": 25,
            "episodes_per_increase": 25,
            "max_num_increases": 10,
        }
    )

    more_out_of_bounds_penalty = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "out_of_track",
            "increase": 1,
            "first_episode_increase": 50,
            "episodes_per_increase": 50,
            "max_num_increases": 5,
        }
    )

    # less_traversability = CurrTerm(
    #     func=increase_reward_weight_over_time,
    #     params={
    #         "reward_term_name": "traversablility",
    #         "increase": -0.25,
    #         "first_episode_increase": 25,
    #         "episodes_per_increase": 25,
    #         "max_num_increases": 2,
    #     }
    # )

##########################
###### TERMINATION #######
##########################

def out_of_map(env):
    poses = mdp.root_pos_w(env)
    poses = poses[..., :2]
    terrain = env.scene[SceneEntityCfg("terrain").name]
    width = terrain.cfg.width
    height = terrain.cfg.height
    x_out_range = torch.logical_or(poses[..., 0] > width / 2, poses[..., 0] < -width / 2)
    y_out_range = torch.logical_or(poses[..., 1] > height / 2, poses[..., 1] < -height / 2)
    return torch.logical_or(x_out_range, y_out_range)


def upright_bool(env, thresh_deg):
    return upright_penalty(env, thresh_deg) > 0.0


def is_not_traversable(env):
    poses = mdp.root_pos_w(env)[..., :2]
    traversability = TraversabilityHashmapUtil().get_traversability(poses)

    num_episodes = env.common_step_counter // env.max_episode_length
    # delay the termination for the first 10 episodes
    # if num_episodes < 10:
    #     return torch.zeros(env.num_envs, device=env.device) == 1
    
    return torch.logical_not(traversability)

def is_reverse(env):
    reverse = reverse_waypoint_bool(env)
    return reverse

def reverse_waypoint_bool(env):
    # Safe access to buffer with fallback
    if not hasattr(env, '_last_waypoint_indices'):
        env._last_waypoint_indices = torch.zeros(env.num_envs, 
                                               dtype=torch.long,
                                               device=env.device)
        
    num_episodes = env.common_step_counter // env.max_episode_length
    if num_episodes < 10:
        return torch.zeros(env.num_envs, device=env.device) == 0
    
    # Get current positions and waypoints
    position_xy = env.scene["robot"].data.root_pos_w[:, :2]
    # waypoints = torch.tensor(env.scene.terrain.cfg.waypoints, 
    #                        device=env.device)[:, :2]

    waypoints = torch.tensor(env.scene.terrain.cfg.inner, 
                           device=env.device)[:, :2]
    
    # Find nearest waypoint
    current_idx, _ = find_nearest_waypoint(waypoints, position_xy)
    

    if current_idx == 0:
        return torch.zeros(env.num_envs, device=env.device) == 1
    
    # Handle lap transitions by checking modulo distance
    num_waypoints = len(waypoints)

    progress = current_idx - env._last_waypoint_indices
    
    # Consider progress if moved forward (even across lap boundary)
    reverse_bool = progress < 0
    
    # Update stored indices
    env._last_waypoint_indices = current_idx.clone()
    
    return reverse_bool

@configclass
class F1TenthTimeTrialTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_range = DoneTerm(
        func=out_of_map,
    )

    non_traversable = DoneTerm(
        func=is_not_traversable
    )

    # reverse = DoneTerm(
    #     func=reverse_waypoint_bool
    # )

    rollover = DoneTerm(
        func=upright_bool,
        params={"thresh_deg": 90.},
    )


@configclass
class F1TenthTimeTrialRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    seed: int = 42
    num_envs: int = 1
    env_spacing: float = 0.

    # Reset config
    events: F1TenthTimeTrialEventsCfg = F1TenthTimeTrialEventsCfg()
    # actions: Mushr4WDActionCfg = Mushr4WDActionCfg()
    actions: F1Tenth4WDActionCfg = F1Tenth4WDActionCfg()

    # MDP settings
    observations: F1TenthTimeTrialObsCfg = F1TenthTimeTrialObsCfg()
    rewards: F1TenthTimeTrialRewardsCfg = F1TenthTimeTrialRewardsCfg()
    terminations: F1TenthTimeTrialTerminationsCfg = F1TenthTimeTrialTerminationsCfg()
    curriculum: TimeTrialCurriculumCfg = TimeTrialCurriculumCfg()


    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # viewer settings
        self.viewer.eye = [10., 0.0, 15.0] 
        self.viewer.lookat = [0.0, 0.0, -3.]
        self.sim.dt = 0.025
        self.decimation = 8
        self.sim.render_interval = self.decimation

        # Terminations config
        self.episode_length_s = 15

        self.actions.throttle_steer.scale = (10, 0.488)

        # Scene settings
        self.scene = F1TenthTimeTrialSceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing,
        )

        # Set the environment class
        self.env_class = F1TenthTimeTrialEnv


class F1TenthTimeTrialEnv(ManagerBasedEnv):
    def __init__(self, cfg: F1TenthTimeTrialRLEnvCfg, **kwargs):
        # Initialize parent class first
        super().__init__(cfg, **kwargs)
        
        # Then initialize your buffers
        self._last_waypoint_indices = torch.zeros(self.num_envs, 
                                                dtype=torch.long,
                                                device=self.device)
        self._current_progress = torch.zeros(self.num_envs,
                                           device=self.device)


@configclass
class F1TenthTimeTrialRLRandomEnvCfg(F1TenthTimeTrialRLEnvCfg):
    events: F1TenthTimeTrialEventsRandomCfg = F1TenthTimeTrialEventsRandomCfg()

######################
###### PLAY ENV ######
######################

@configclass
class F1TenthTimeTrialPlayEnvCfg(F1TenthTimeTrialRLEnvCfg):
    """no terminations"""

    events: F1TenthTimeTrialEventsCfg = F1TenthTimeTrialEventsRandomCfg(
        reset_root_state = EventTerm(
            func=reset_root_state,
            params={
                "dist_noise": 0.,
                "yaw_noise": 0.,
            },
            mode="reset",
        )
    )

    rewards: F1TenthTimeTrialRewardsCfg = None
    terminations: F1TenthTimeTrialTerminationsCfg = None

    def __post_init__(self):
        super().__post_init__()