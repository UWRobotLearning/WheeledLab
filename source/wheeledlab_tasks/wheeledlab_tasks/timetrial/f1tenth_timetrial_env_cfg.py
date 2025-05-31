from collections import Counter
import os
import time
import torch
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import MISSING

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
from wheeledlab_assets.f1tenth import F1TENTH_CFG, LB_CFG
from wheeledlab_tasks.common import Mushr4WDActionCfg
from wheeledlab_tasks.common import F1Tenth4WDActionCfg, LB4WDActionCfg
from .disable_lidar import disable_all_lidars

from .utils import create_maps_from_waypoints, generate_random_poses, TraversabilityHashmapUtil, find_nearest_waypoint 
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
    pos_xy_world =mdp.root_pos_w(env)[..., :2]
    
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
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
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
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
    
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
    norm_distance = 10
    norm_signed_deviations = signed_deviations/norm_distance

    return norm_signed_deviations 


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
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
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
    
    norm_heading_errors = np.pi

    return heading_errors/norm_heading_errors

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
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
    
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
    norm_d_lat = 2

    norm_next_d_lat_horizon = next_d_lat_horizon.reshape(-1, horizon * 2)/norm_d_lat
    # Reshape to [num_envs, horizon * 2] for concatenation with other observations
    return norm_next_d_lat_horizon

def kappa_radpm_horizon(
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
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
    
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

    kappa_radpm = torch.tensor(
        env.scene.terrain.cfg.kappa_radpm, 
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
    norm_kappa_radpm = 3
    next_kappa_radpm_horizon = kappa_radpm[next_idx_horizon, :]
    norm_next_kappa_radpm_horizon = next_kappa_radpm_horizon/norm_kappa_radpm

    # Reshape to [num_envs, horizon * 2] for concatenation with other observations
    return torch.reshape(norm_next_kappa_radpm_horizon, (-1, horizon)) 

def vx_mps_horizon(
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
    pos_xy_world = mdp.root_pos_w(env)[..., :2]
    
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

    vx_mps = torch.tensor(
        env.scene.terrain.cfg.vx_mps, 
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
    norm_vx = 10
    next_vx_mps_horizon = vx_mps[next_idx_horizon, :]
    norm_next_vx_mps_horizon = next_vx_mps_horizon/norm_vx
    # Reshape to [num_envs, horizon * 2] for concatenation with other observations
    return torch.reshape(norm_next_vx_mps_horizon, (-1, horizon))

HORIZON = 10
LOOKAHEAD = 15
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
        #only one env gives problem
        kappa_radpm_horizon = ObsTerm(
            func=kappa_radpm_horizon,
            params={'lookahead': LOOKAHEAD,
                    'horizon': HORIZON}
        )
        # vx_mps_horizon = ObsTerm(
        #     func=vx_mps_horizon,
        #     params={'lookahead': LOOKAHEAD,
        #             'horizon': HORIZON}
        # )
        # centerline_horizon = ObsTerm(
        #     func=centerline_relative_heading_horizon,
        #     params={'lookahead': LOOKAHEAD,
        #             'horizon': HORIZON}
        # )
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
    NUM_ENVS  = 5
    MAP_NAME = 'f'
    file_name = os.path.join(WHEELEDLAB_ASSETS_DATA_DIR, 'maps', 'preinit.usd')
    maps_folder_path = '/home/tongo/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/timetrial/utils/maps' 
    origin = [0, 0, 0]   
    traversability_hashmap, waypoints, outer, inner, d_lat, psi_rad, kappa_radpm, vx_mps, spacing_meters, map_size_pixels = create_maps_from_waypoints(maps_folder_path, MAP_NAME, NUM_ENVS, origin, file_name, resolution=0.1)

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
    env_spacing = 0

    prim_path = "/World/ground"
    terrain_type="usd"
    usd_path = file_name
    collision_group = -1
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.1,
        dynamic_friction=1.1,
    )
    debug_vis=False
    
    def generate_random_poses(self, env : ManagerBasedEnv, env_ids, num_poses):
        # generate random initial poses with margin
        env_origins = env.scene.env_origins
        init_poses = generate_random_poses(env_origins, env_ids, num_poses, self.row_spacing, self.col_spacing, self.traversability_hashmap, self.waypoints, self.outer, self.inner, margin=0.1)
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
    # waypoint_loader = WaypointLoader(map_name="f")
    # Add ground config (ground is slightly below terrain)
    terrain = None
    terrain_1 = None


    ground = AssetBaseCfg(
        prim_path="/World/base",
        spawn = sim_utils.GroundPlaneCfg(size=(500, 500),
                                         color=(0,0,0),
                                         physics_material=sim_utils.RigidBodyMaterialCfg(
                                            friction_combine_mode="multiply",
                                            restitution_combine_mode="multiply",
                                            static_friction=1.0,
                                            dynamic_friction=1.0,
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
    # robot: ArticulationCfg = LB_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
    position_xy = mdp.root_pos_w(env)[..., :2]

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
    poses =mdp.root_pos_w(env)[..., :2]
    traversability = TraversabilityHashmapUtil().get_traversability(poses)
    return torch.where(traversability, 1., 0.)

def traversable_reward(env):
    poses =mdp.root_pos_w(env)[..., :2]
    traversability = TraversabilityHashmapUtil().get_traversability(poses)
    return torch.where(traversability, 1, 0.)

def out_of_track_penalty(env):
    poses =mdp.root_pos_w(env)[..., :2]
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

def upright_penalty(env, thresh_deg):
    rot_mat = math_utils.matrix_from_quat(mdp.root_quat_w(env))
    up_dot = rot_mat[:, 2, 2]
    up_dot = torch.rad2deg(torch.arccos(up_dot))
    penalty = torch.where(up_dot > thresh_deg, up_dot - thresh_deg, 0.)
    return penalty

def vel_rew_trav(env, speed_target_on_trav: float=1., speed_target_on_non_trav: float=2.):
    poses =mdp.root_pos_w(env)[..., :2]
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
    if not hasattr(env, '_history_waypoint_indices'):
        env._history_length = 10  # Store last 10 waypoints
        env._history_waypoint_indices = torch.zeros(
            (env.num_envs, env._history_length), 
            dtype=torch.long,
            device=env.device
        )
        env._reset_env_bool = torch.ones(  # Tracks where to insert the next index
            env.num_envs,
            dtype=torch.bool,
            device=env.device
        )
    
    # Get current waypoint
    position_xy = mdp.root_pos_w(env)[..., :2]
    waypoints = torch.tensor(env.scene.terrain.cfg.waypoints, device=env.device)[:, :2]
    current_idx, _ = find_nearest_waypoint(waypoints, position_xy)
    num_waypoints = len(waypoints)

    # Shift old values to the right (discard oldest)
    env._history_waypoint_indices[:, 1:] = env._history_waypoint_indices[:, :-1].clone()
    
    # Insert newest at position 0
    env._history_waypoint_indices[:, 0] = current_idx

    # Progress = (current - previous) % num_waypoints
    # Previous is now at [:, 1] (since [:, 0] is current)
    progress = (current_idx - env._history_waypoint_indices[:, 1]) % num_waypoints
    # progress_bool = (progress > 0) & (progress <= 10)

    progress_bool = (progress > 0) & (progress <= 20) & (env._reset_env_bool == False)
    env._reset_env_bool = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Store extras
    env.extras['s_idx'] = current_idx.clone()
    env.extras['time'] = torch.tensor(env.sim.current_time, device=env.device)
    env.extras['s_idx_max'] = torch.tensor(num_waypoints - 1, device=env.device)

    return progress_bool, progress

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
            "first_episode_increase": 50,
            "episodes_per_increase": 100,
            "max_num_increases": 0,
        }
    )

    more_out_of_bounds_penalty = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "out_of_track",
            "increase": 0,
            "first_episode_increase": 50,
            "episodes_per_increase": 50,
            "max_num_increases": 0,
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
    poses =mdp.root_pos_w(env)[..., :2]
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
    if not hasattr(env, '_history_waypoint_indices'):
        env._history_waypoint_indices = torch.zeros(env.num_envs, 
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

    progress = current_idx - env._history_waypoint_indices
    
    # Consider progress if moved forward (even across lap boundary)
    reverse_bool = progress < 0
    
    # Update stored indices
    env._history_waypoint_indices = current_idx.clone()
    
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

    # These will be overwritten by the rss_cfgs
    seed: int = 42
    num_envs: int = 1
    env_spacing: int = 0

    map_name: str = 'THETRACK'

    # Reset config
    events: F1TenthTimeTrialEventsCfg = F1TenthTimeTrialEventsCfg()

    # actions: Mushr4WDActionCfg = Mushr4WDActionCfg()
    actions: F1Tenth4WDActionCfg = F1Tenth4WDActionCfg()
    # actions: LB4WDActionCfg = LB4WDActionCfg()

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
        self.sim.dt = 0.01
        self.decimation = 8
        self.sim.render_interval = self.decimation

        # Terminations config
        self.episode_length_s = 15

        self.actions.throttle_steer.scale = (10, 0.488)


        #terrain
        MAP_NAME = self.map_name

        file_name = os.path.join(WHEELEDLAB_ASSETS_DATA_DIR, 'maps', MAP_NAME+'.usd')

        origin = [0, 0, 0]

        NUM_ENVS = self.num_envs

        maps_folder_path = '/home/tongo/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/timetrial/utils/maps'    

        traversability_hashmap, waypoints, outer, inner, d_lat, psi_rad, kappa_radpm, vx_mps, spacing_meters, map_size_pixels = create_maps_from_waypoints(maps_folder_path, MAP_NAME, NUM_ENVS, origin, file_name, resolution=0.1)

        # terrain
        self.terrain  = F1TenthTimeTrialTerrainImporterCfg(
            prim_path="/World/envs/env_.*",
            env_spacing=self.env_spacing,
            NUM_ENVS=self.num_envs,
            usd_path=file_name,
            traversability_hashmap=traversability_hashmap,
            waypoints = waypoints,
            outer = outer, 
            inner = inner,
            d_lat=d_lat,
            psi_rad = psi_rad,
            kappa_radpm=kappa_radpm,
            vx_mps=vx_mps,
            spacing_meters=spacing_meters,
            map_size_pixels=map_size_pixels,
            origin=origin,
            # Override or add new terrain parameters
            MAP_NAME=MAP_NAME,  # Example: Change map name
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.1,  # Example: Adjust friction
                dynamic_friction=1.1,
            ),
            debug_vis=True,  # Example: Enable debug visualization
        )


        # MAP_NAME_1 = 'GLC_smile_small'
        # file_name_1 = os.path.join(WHEELEDLAB_ASSETS_DATA_DIR, 'maps', MAP_NAME_1+'.usd')
        # origin_1 = [10, 0, 0]
        # traversability_hashmap_1, waypoints_1, outer_1, inner_1, d_lat_1, psi_rad_1, kappa_radpm_1, vx_mps_1, spacing_meters_1, map_size_pixels_1 = create_maps_from_waypoints(maps_folder_path, MAP_NAME_1, NUM_ENVS, origin_1, file_name_1, resolution=0.1)

        # self.terrain_1  = F1TenthTimeTrialTerrainImporterCfg(
        #     prim_path="/World/terrain_1",
        #     env_spacing=self.env_spacing,
        #     NUM_ENVS=self.num_envs,
        #     usd_path=file_name_1,
        #     traversability_hashmap=traversability_hashmap_1,
        #     waypoints = waypoints_1,
        #     outer = outer_1, 
        #     inner = inner_1,
        #     d_lat=d_lat_1,
        #     psi_rad = psi_rad_1,
        #     kappa_radpm=kappa_radpm_1,
        #     vx_mps=vx_mps_1,
        #     spacing_meters=spacing_meters_1,
        #     map_size_pixels=map_size_pixels_1,
        #     origin=origin_1,
        #     # Override or add new terrain parameters
        #     MAP_NAME='f',  # Example: Change map name
        #     physics_material=sim_utils.RigidBodyMaterialCfg(
        #         static_friction=1.1,  # Example: Adjust friction
        #         dynamic_friction=1.1,
        #     ),
        #     debug_vis=True,  # Example: Enable debug visualization
        # )

        # Scene settings
        self.scene = F1TenthTimeTrialSceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing, terrain = self.terrain
        )

        # Set the environment class
        self.env_class = F1TenthTimeTrialEnv


class F1TenthTimeTrialEnv(ManagerBasedEnv):
    def __init__(self, cfg: F1TenthTimeTrialRLEnvCfg, **kwargs):
        # Initialize parent class first
        super().__init__(cfg, **kwargs)
        
        # Then initialize your buffers
        self._initial_waypoint_indices = torch.zeros(self.num_envs, 
                                                dtype=torch.long,
                                                device=self.device)

        self._history_length = 10

        self._history_waypoint_indices = torch.zeros(
            (self.num_envs, self._history_length),  # Shape: (num_envs, history_length)
            dtype=torch.long,
            device=self.device
        )

        self._reset_env_bool = torch.zeros(  # Tracks where to insert the next index
            self.num_envs,
            dtype=torch.bool,
            device=self.device
        )

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