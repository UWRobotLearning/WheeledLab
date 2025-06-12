import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab.sensors import RayCasterCfg, patterns
import isaaclab.envs.mdp as mdp
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    CurriculumTermCfg as CurrTerm,
    SceneEntityCfg,
)

from isaaclab.utils.noise import (
    AdditiveUniformNoiseCfg as Unoise,
    AdditiveGaussianNoiseCfg as Gnoise,
)

from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.commands import UniformPose2dCommandCfg
from isaaclab.envs.mdp.events import reset_root_state_uniform

from wheeledlab_assets import WHEELEDLAB_ASSETS_DATA_DIR
from wheeledlab.envs.mdp.observations import root_euler_xyz
from wheeledlab_assets.mushr import MUSHR_SUS_CFG
from wheeledlab.envs.mdp import increase_reward_weight_over_time
from wheeledlab_tasks.common import Mushr4WDActionCfg

# ##########################
# ###### OBSERVATIONS ######
# ##########################

def world_height_map(env, sensor_cfg:SceneEntityCfg, offset:int, plane_init_value:int):
    height_scan = -mdp.height_scan(env, sensor_cfg, offset)
    world_pos_z = mdp.root_pos_w(env)[..., 2] - plane_init_value
    corr_height_scan = height_scan + world_pos_z.unsqueeze(-1)
    return corr_height_scan

def goal_relative_xyz(env : ManagerBasedEnv):
    pos = mdp.root_pos_w(env)
    goal_pos = mdp.generated_commands(env, "goal_pose")
    goal_pos = goal_pos[:, :2]  # we only need the x, y coordinates
    rel_pos = goal_pos - pos[:, :2]
    return torch.nan_to_num(rel_pos, nan=0)

@configclass
class ElevationObsCfg:
    """Observation specification for the elevation environment."""
    @configclass
    class ConcatObs(ObsGroup):
        goal_relative_xyz = ObsTerm(
            func=goal_relative_xyz,
        )
        world_euler_xyz = ObsTerm(
            func=root_euler_xyz,
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-10., 10.))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-10., 10.))
        last_action = ObsTerm(
            func=mdp.last_action,
            clip=(-1., 1.)
        )
        elevation_map = ObsTerm(
            func=world_height_map,
            params={
                    "sensor_cfg":SceneEntityCfg("height_scanner"),
                    "offset": 0.084,
                    "plane_init_value": 0.19
                },
            clip=(-10., 10.),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ConcatObs = ConcatObs()

##########################
### Scene Setup Config ###
##########################

@configclass
class ElevationTerrainImporterCfg(TerrainImporterCfg):

    height = 0.25
    prim_path="/World/elevation"
    terrain_type = "usd"
    usd_path=f"{WHEELEDLAB_ASSETS_DATA_DIR}/Terrains/huge_compact.usd"
    collision_group = -1
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    )
    debug_vis=False


@configclass
class ElevationSceneCfg(InteractiveSceneCfg):
    
    terrain = ElevationTerrainImporterCfg()
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    ground = AssetBaseCfg(
        prim_path="/World/base",
        spawn = sim_utils.GroundPlaneCfg(size=(1600.0, 1200.0),
                                         color=(3,3,3),
                                         physics_material=sim_utils.RigidBodyMaterialCfg(
                                            static_friction=1.0,
                                            dynamic_friction=0.5,
                                            restitution=0.0)),
    )

    robot: ArticulationCfg = MUSHR_SUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mushr_nano/base_link",
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 20.0),
            # rot=(0.0, 1.0, 0.0, 0.0),
        ),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(size=[2.5, 2.5], resolution=0.1),
        debug_vis=False,
        mesh_prim_paths=["/World/elevation/terrain"],
    )

    def __post_init__(self):
        """Post intialization."""
        super().__post_init__()
        self.robot.init_state = self.robot.init_state.replace(
            pos=(0.0, 0.0, self.terrain.height)
        )

#############################
########## REWARDS ##########
#############################

def forward_vel(env):
    lin_vel = mdp.base_lin_vel(env)
    return torch.clamp(lin_vel[..., 0], max=1.2)

def forward_wheel_spin(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    throttle_joints = asset.find_joints(".*_throttle")[0]
    throttle_joint_vel = mdp.joint_vel(env)[..., throttle_joints]
    sum_vels = torch.sum(throttle_joint_vel, dim=-1)
    return torch.clamp(sum_vels, max=200)

def higher_elevation(env):
    pos = mdp.root_pos_w(env)
    z_value = pos[..., 2] -  0.19
    vel = mdp.base_lin_vel(env)[..., 0]
    # print(z_value)
    condition = (z_value > 0.1) & (vel>0.1)
    rew = torch.where(condition, z_value, torch.zeros_like(z_value)) 
    return torch.clip(rew, min=0, max=1) # weight this

def change_in_elevation(env):
    vel = mdp.root_lin_vel_w(env)
    change_in_z = vel[..., 2]
    return torch.where(change_in_z > 0, change_in_z, torch.zeros_like(change_in_z))

def steep_penalty(env, thresh_pitch):
    orient = mdp.root_quat_w(env)
    euler_xyz = mdp.euler_xyz_from_quat(orient)
    euler_xyz = torch.stack(euler_xyz, dim=-1)
    pitch = euler_xyz[:, 1]
    steep_ramp = torch.clamp(pitch - thresh_pitch, min=0)
    return steep_ramp

def elevation_continuity(env, threshold_elev):
    pos = mdp.root_pos_w(env)    
    z_value = pos[..., 2] - 0.19  # Base elevation adjustment
    # print((z_value > 0.1).sum())
    if not hasattr(elevation_continuity, "prev_elevation"):
        elevation_continuity.prev_elevation = z_value.clone()
    delta_z = z_value - elevation_continuity.prev_elevation
    on_ramp = z_value > threshold_elev
    ascending = delta_z > 0
    descending = delta_z < 0
    rew_ascend = torch.where(on_ramp & ascending, 50*delta_z, torch.zeros_like(z_value))
    rew_descend = torch.where(on_ramp & descending, -50*delta_z, torch.zeros_like(z_value))  # Note: -delta_z makes it positive
    rew = rew_ascend + rew_descend
    elevation_continuity.prev_elevation = z_value.clone()
    return rew


def yaw_change_onElev(env, threshold_yaw, threshold_z):
    pos = mdp.root_pos_w(env)    
    z_value = pos[..., 2] - 0.19
    ang_vel_yaw = mdp.base_ang_vel(env)[..., 2]    
    condition = (z_value > threshold_z) & (abs(ang_vel_yaw)>threshold_yaw)
    rew = torch.where(condition, 2*abs(ang_vel_yaw)**2, torch.zeros_like(ang_vel_yaw))
    return rew

#############################
########## REWARDS ##########
#############################

def upright_penalty(env, thresh_deg):
    rot_mat = math_utils.matrix_from_quat(mdp.root_quat_w(env))
    up_dot = rot_mat[:, 2, 2]
    up_dot = torch.rad2deg(torch.arccos(up_dot))
    penalty = torch.where(up_dot > thresh_deg, up_dot - thresh_deg, 0.)
    return penalty

def roll_on_elev(env, z_start, roll_rate_thresh):
    ang_vel_roll = mdp.base_ang_vel(env)[..., 0]
    pos = mdp.root_pos_w(env)
    z_value = pos[..., 2] - 0.19
    condition = (z_value > z_start) & (abs(ang_vel_roll)>roll_rate_thresh) 
    rew = torch.where(condition, abs(ang_vel_roll)*2.0, torch.zeros_like(ang_vel_roll))
    # elev_map_visualizer(env)
    return rew

def is_falling_penalty(env, map_length_px=26, sensor_cfg=SceneEntityCfg("height_scanner")):
    pos = mdp.root_pos_w(env)
    vel = mdp.base_lin_vel(env)

    return torch.where(vel[:, 2] > 1.0, 1.0, 0.0)

def goal_progress_rate(env):
    pos = mdp.root_pos_w(env)
    vel = mdp.root_lin_vel_w(env)
    goal_pos = mdp.generated_commands(env, "goal_pose")
    goal_pos = goal_pos[:, :2] # we only need the x, y coordinates

    vel_vector = vel[:, :2]
    goal_vector = goal_pos - pos[:, :2]
    proj_scal = torch.sum(vel_vector * goal_vector, dim=-1) / torch.norm(goal_vector, dim=-1)

    return 5 + proj_scal

def is_falling_penalty(env, max_body_z_vel:float = 0.10):
    lin_vel = mdp.base_lin_vel(env)
    is_falling = lin_vel[..., 2] > max_body_z_vel
    return is_falling

def ascending(env,):
    vel_w = mdp.root_lin_vel_w(env)
    rew = torch.clamp(vel_w[..., 2], min=0.)
    return rew

def low_vel_penalty(env, min_vel:float = 0.1):
    lin_vel = mdp.base_lin_vel(env)
    vel = lin_vel[..., 0]
    penalty = torch.where(vel < min_vel, 1., 0.)
    return penalty
    return proj_scal

def close_to_goal(env, dist):
    pos = mdp.root_pos_w(env)
    goal_pos = mdp.generated_commands(env, "goal_pose")
    goal_pos = goal_pos[:, :2]
    curr_dist = torch.norm(goal_pos - pos[:, :2], dim=-1)
    return curr_dist < dist

def upright_bool(env, thresh_deg):
    return upright_penalty(env, thresh_deg) > 0.0

def stuck(env, min_vel, wheel_spin_thr):
    not_moving = forward_vel(env) < min_vel
    spinning_wheels = forward_wheel_spin(env) > wheel_spin_thr
    return torch.logical_and(not_moving, spinning_wheels) 

@configclass
class ElevationRewardsCfg:

    vel_towards_goal = RewTerm(
        func = goal_progress_rate,
        weight = 200.0,
    )
    
    height_z = RewTerm(
        func = higher_elevation, 
        weight = 5000.,
    )

    falling_penalty = RewTerm(
        func = is_falling_penalty,
        weight = 0.,
    )
    
    termination_penalty = RewTerm(
        func = mdp.rewards.is_terminated_term,
        params={"term_keys": "stuck"},
        weight = -200.0
    )

########################
###### CURRICULUM ######
########################

@configclass
class ElevationCurriculumCfg:
    """Configuration for the elevation policy curriculum."""

    more_goal = CurrTerm(
        func = increase_reward_weight_over_time,
        params={
            "reward_term_name" : "vel_towards_goal",
            "increase": 5.,
            "episodes_per_increase": 50, 
            "max_increases": 5,
        }
    )

    more_falling_pen = CurrTerm(
        func = increase_reward_weight_over_time,
        params={
            "reward_term_name" : "falling_penalty",
            "increase": 1.,
            "episodes_per_increase": 50, 
            "max_increases": 10,
        }
    )

##########################
###### TERMINATION #######
##########################

def upright_bool(env, thresh_deg):
    return upright_penalty(env, thresh_deg) > 0.0

def stuck(env, min_vel, wheel_spin_thr):
    not_moving = forward_vel(env) < min_vel
    throttle_joints_asset = SceneEntityCfg("robot", joint_names=".*throttle")
    joint_vels = mdp.joint_vel(env, asset_cfg=throttle_joints_asset)
    spinning_wheels = torch.sum(joint_vels, dim=-1) > wheel_spin_thr
    return torch.logical_and(not_moving, spinning_wheels)

@configclass
class ElevationTerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.15},
    )
    stuck = DoneTerm(
        func=stuck,
        params={
            "min_vel": 0.02,
            "wheel_spin_thr": 5.,
        },
    )
    # (4) Stuck (upside down)
    rollover = DoneTerm(
        func=upright_bool,
        params={"thresh_deg": 60.},
    )

    at_goal = DoneTerm(
        func=close_to_goal,
        params={"dist": 0.5},
    )

#####################
###### EVENTS #######
#####################

@configclass
class ElevationSceneEventsCfg:
    """Configuration for the events."""

    # on startup
    change_wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (2.0, 2.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 5,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_.*link"),
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (0.2, 0.5),
            "operation": "add",
        },
    )

    set_goal = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-19., 19.), "y": (-19., 19.), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.1, 0.2),
                "y": (0.1, 0.2),
            },
        }
    )

@configclass
class ElevationCommandCfg:
    """Configuration for the elevation commands."""

    goal_pose = UniformPose2dCommandCfg(
        asset_name="robot",
        ranges=UniformPose2dCommandCfg.Ranges(
            pos_x=(-19.0, 19.0),
            pos_y=(-19.0, 19.0),
            heading=(-3.14, 3.14),
        ),
        resampling_time_range=(10.0, 10.0),
        simple_heading=True,
        debug_vis=True
    )

@configclass
class MushrElevationRLEnvCfg(ManagerBasedRLEnvCfg):

    seed: int = 42
    num_envs: int = 512
    env_spacing: float = 0.

    # Basic Settings
    observations: ElevationObsCfg = ElevationObsCfg()
    actions: Mushr4WDActionCfg = Mushr4WDActionCfg()

    # MDP settings
    events : ElevationSceneEventsCfg = ElevationSceneEventsCfg()
    curriculum: ElevationCurriculumCfg = ElevationCurriculumCfg()
    rewards: ElevationRewardsCfg = ElevationRewardsCfg()
    terminations: ElevationTerminationsCfg = ElevationTerminationsCfg()

    commands = ElevationCommandCfg = ElevationCommandCfg()

    def __post_init__(self):
        super().__post_init__()
        self.viewer.eye = [20., -20.0, 20.0]
        self.viewer.lookat = [0.0, 0.0, 0.]

        self.sim.dt = 0.01  # 100 Hz
        self.decimation = 10  # 10 Hz
        self.actions.throttle_steer.scale = (3.0, 0.488)
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20

        self.scene = ElevationSceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing,
        )


@configclass
class MushrElevationPlayEnvCfg(MushrElevationRLEnvCfg):
    """no terminations"""
    terminations: ElevationTerminationsCfg = None