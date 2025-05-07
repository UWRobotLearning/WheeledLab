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
from isaaclab.assets import ArticulationCfg, RigidObject, AssetBaseCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
)
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs import ManagerBasedRLEnvCfg

from wheeledlab_assets import WHEELEDLAB_ASSETS_DATA_DIR
from wheeledlab_assets.mushr import MUSHR_SUS_CFG
from wheeledlab_tasks.common import Mushr4WDActionCfg

from .utils import create_oval, create_maps_from_png, generate_random_poses, TraversabilityHashmapUtil, WaypointsUtil
from . import mdp_sensors
from .mdp import reset_root_state

import omni.usd




@configclass
class MushrTimeTrialObsCfg:
    """Observation specifications for the environment."""
    @configclass
    class PolicyCfg(ObsGroup):
        """
        [vx, vy, vz, wx, wy, wz, action1(vel), action2(steering)]
        """
        # lidar = ObsTerm(func=mdp_sensors.lidar_ranges, params={"sensor_cfg":SceneEntityCfg("lidar")})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-.1, n_max=.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-.1, n_max=.1))
        last_action = ObsTerm(
            func=mdp.last_action,
            clip=(-1., 1.), # TODO: get from ClipAction wrapper
            noise=Unoise(-.1, .1)
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
class MushrTimeTrialTerrainImporterCfg(TerrainImporterCfg):
    # map generation parameters

    # generate a colored plane geometry
    file_name = os.path.join(WHEELEDLAB_ASSETS_DATA_DIR, 'rgb_maps', time.strftime("%Y%m%d_%H%M%S.usd"))
    # '/home/tongo/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/timetrial/utils/maps/'
    map_name = 'f'
    traversability_hashmap, waypoints, spacing_meters, map_size_pixels = create_maps_from_png(map_name, file_name, 0.25)

    # Grid resolution
    
    # row_spacing = 0.10
    # col_spacing = 0.10
    # spacing = (row_spacing, col_spacing)

    spacing = spacing_meters
    row_spacing, col_spacing = spacing
    ####################################
    # Debugging map size
    # num_rows = 100
    # num_cols = num_rows

    num_rows, num_cols = map_size_pixels
    map_size = (num_rows+1, num_cols+1)

    # environments size are generated in a grid
    
    # env_num_rows = 500
    # env_num_cols = env_num_rows

    # env_size = (env_num_rows, env_num_cols)
    env_size = map_size
    # # sub group size
    # group_num_rows = 1
    # group_num_cols = 1 
    # sub_group_size = (group_num_rows, group_num_cols)
    # num_walkers = 1
    ####################################

    # whether to sample colors
    # color_sampling = False

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


    prim_path = "/World/plane"
    terrain_type="usd"
    usd_path = file_name
    collision_group = -1
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=2.0,
        dynamic_friction=2.0,
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
        init_poses = generate_random_poses(num_poses, self.row_spacing, self.col_spacing, self.traversability_hashmap, self.waypoints, margin=0.1)
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
class MushrTimeTrialSceneCfg(InteractiveSceneCfg):
    """Configuration for a Mushr car Scene with racetrack terrain and Sensors."""
    terrain = MushrTimeTrialTerrainImporterCfg()
    # waypoint_loader = WaypointLoader(map_name="f")
    # Add ground config (ground is slightly below terrain)
    ground = AssetBaseCfg(
        prim_path="/World/base",
        spawn = sim_utils.GroundPlaneCfg(size=(terrain.width, terrain.height),
                                         color=(0,0,0),
                                         physics_material=sim_utils.RigidBodyMaterialCfg(
                                            friction_combine_mode="multiply",
                                            restitution_combine_mode="multiply",
                                            static_friction=2.0,
                                            dynamic_friction=2.0,
                                         ),
        )
    )
    # # Add light configuration
    # light = AssetBaseCfg(
    #     prim_path="/World/light",
    #     spawn=sim_utils.DistantLightCfg(
    #         color=(1.0, 1.0, 1.0),
    #         intensity=5000.0,
    #     )
    # )

    robot: ArticulationCfg = MUSHR_SUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
@configclass
class MushrTimeTrialEventsCfg:
    # on startup

    reset_root_state = EventTerm(
        func=reset_root_state,
        mode="reset",
    )

@configclass
class MushrTimeTrialEventsRandomCfg(MushrTimeTrialEventsCfg):
    change_wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.4, 0.6),
            "dynamic_friction_range": (0.4, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 10,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_.*link"),
            "make_consistent": False,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (1.0, 3.0),
            "operation": "abs",
        },
    )

    add_wheel_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_.*link"),
            "mass_distribution_params": (.01, 0.3),
            "operation": "abs",
        },
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
    return torch.where(traversability, 1., -10.)

def bool_is_not_traversable(env):
    num_episodes = env.common_step_counter // env.max_episode_length
    # delay the penalty for the first 300 episodes
    if num_episodes < 50:
        return torch.zeros(env.num_envs, device=env.device) == 1

    traversability = is_traversable(env)
    return traversability == 0

def is_traversable_speed_scaled(env):
    return is_traversable(env) * mdp.base_lin_vel(env)[:, 0]

def is_traversable_wheels(env):
    body_asset_cfg = SceneEntityCfg("robot", body_names=".*wheel_link")
    body_asset_cfg.resolve(env.scene)
    # B x num_body x 2
    poses = env.scene[body_asset_cfg.name].data.body_pos_w[:, body_asset_cfg.body_ids][:, :, :2]
    B, num_body = poses.shape[:2]
    # terrain = env.scene[SceneEntityCfg("terrain").name]
    traversability = TraversabilityHashmapUtil().get_traversability(poses.reshape(-1, 2)).reshape(B, num_body)
    return torch.where(traversability == 1, 1., -5.).sum(dim=-1)

def binary_is_traversable_wheels(env):
    body_asset_cfg = SceneEntityCfg("robot", body_names=".*wheel_link")
    body_asset_cfg.resolve(env.scene)
    # B x num_body x 2
    poses = env.scene[body_asset_cfg.name].data.body_pos_w[:, body_asset_cfg.body_ids][:, :, :2]
    B, num_body = poses.shape[:2]
    # terrain = env.scene[SceneEntityCfg("terrain").name]
    traversability = TraversabilityHashmapUtil().get_traversability(poses.reshape(-1, 2)).reshape(B, num_body)
    return torch.where(traversability == 1, 1., 0.).sum(dim=-1) == 0

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
    progress = progress_waypoint_bool(env)
    return torch.where(progress, 1.0, 0.0)

def progress_waypoint_bool(env):
    # Safe access to buffer with fallback
    if not hasattr(env, '_last_waypoint_indices'):
        env._last_waypoint_indices = torch.zeros(env.num_envs, 
                                               dtype=torch.long,
                                               device=env.device)
    
    poses = mdp.root_pos_w(env)
    position_xy = poses[:, 0:2]
    waypoints = torch.tensor(env.scene.terrain.cfg.waypoints, device=env.device)[:, 0:2]
    
    # Get current closest
    current_idx, _ = find_nearest_waypoint(waypoints, position_xy)
    progress_bool = current_idx > env._last_waypoint_indices
    
    # Initialize if first run
    if not hasattr(env, '_last_waypoint_indices'):
        env._last_waypoint_indices = current_idx.clone()
        return torch.zeros(env.num_envs, device=env.device)
    env._last_waypoint_indices = current_idx.clone()

    return progress_bool

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



####### MushrTimeTrial Environment #######
@configclass
class MushrTimeTrialRewardsCfg:
    # """Reward terms for the MDP."""
    traversablility = RewTerm(
        func=traversable_reward,
        weight=1.,
    )

    vel_rew = RewTerm(
        func=forward_vel,
        weight= 1.,
    )

    progress_rew = RewTerm(
        func=progress_rew,
        weight=1.,
    )

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

def roll_over(env):
    roll = euler_xyz_from_quat(mdp.root_quat_w(env))[0] - torch.pi
    return torch.logical_and(roll < torch.pi / 2, roll > -torch.pi / 2)

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
    
    poses = mdp.root_pos_w(env)
    position_xy = poses[:, 0:2]
    waypoints = torch.tensor(env.scene.terrain.cfg.waypoints, device=env.device)[:, 0:2]
    
    # Get current closest
    current_idx, _ = find_nearest_waypoint(waypoints, position_xy)
    reverse_bool = current_idx < env._last_waypoint_indices
    
    # Initialize if first run
    if not hasattr(env, '_last_waypoint_indices'):
        env._last_waypoint_indices = current_idx.clone()
        return torch.zeros(env.num_envs, device=env.device)
    env._last_waypoint_indices = current_idx.clone()

    return reverse_bool

@configclass
class MushrTimeTrialTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_range = DoneTerm(
        func=out_of_map,
    )

    non_traversable = DoneTerm(
        func=is_not_traversable
    )

    reverse = DoneTerm(
        func=is_reverse
    )

@configclass
class MushrTimeTrialRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    seed: int = 42
    num_envs: int = 16
    env_spacing: float = 0.

    # Reset config
    events: MushrTimeTrialEventsCfg = MushrTimeTrialEventsCfg()
    actions: Mushr4WDActionCfg = Mushr4WDActionCfg()

    # MDP settings
    observations: MushrTimeTrialObsCfg = MushrTimeTrialObsCfg()
    rewards: MushrTimeTrialRewardsCfg = MushrTimeTrialRewardsCfg()
    terminations: MushrTimeTrialTerminationsCfg = MushrTimeTrialTerminationsCfg()


    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # viewer settings
        self.viewer.eye = [40., 0.0, 45.0] 
        self.viewer.lookat = [0.0, 0.0, -3.]
        self.sim.dt = 0.02
        self.decimation = 10

        # Terminations config
        self.episode_length_s = 15

        # Scene settings
        self.scene = MushrTimeTrialSceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing,
        )

        # Set the environment class
        self.env_class = MushrTimeTrialEnv


class MushrTimeTrialEnv(ManagerBasedEnv):
    def __init__(self, cfg: MushrTimeTrialRLEnvCfg, **kwargs):
        # Initialize parent class first
        super().__init__(cfg, **kwargs)
        
        # Then initialize your buffers
        self._last_waypoint_indices = torch.zeros(self.num_envs, 
                                                dtype=torch.long,
                                                device=self.device)
        self._current_progress = torch.zeros(self.num_envs,
                                           device=self.device)


@configclass
class MushrTimeTrialRLRandomEnvCfg(MushrTimeTrialRLEnvCfg):
    events: MushrTimeTrialEventsRandomCfg = MushrTimeTrialEventsRandomCfg()

######################
###### PLAY ENV ######
######################

@configclass
class MushrTimeTrialPlayEnvCfg(MushrTimeTrialRLEnvCfg):
    """no terminations"""

    events: MushrTimeTrialEventsCfg = MushrTimeTrialEventsRandomCfg(
        reset_root_state = EventTerm(
            func=reset_root_state,
            params={
                "dist_noise": 0.,
                "yaw_noise": 0.,
            },
            mode="reset",
        )
    )

    rewards: MushrTimeTrialRewardsCfg = None
    terminations: MushrTimeTrialTerminationsCfg = None

    def __post_init__(self):
        super().__post_init__()