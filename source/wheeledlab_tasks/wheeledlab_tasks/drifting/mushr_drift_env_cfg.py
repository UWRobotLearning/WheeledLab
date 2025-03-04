import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, RigidObject, AssetBaseCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    CurriculumTermCfg as CurrTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)
from wheeledlab.envs.mdp import increase_reward_weight_over_time
from wheeledlab_assets import MUSHR_SUS_2WD_CFG
from wheeledlab_tasks.common import BlindObsCfg, MushrRWDActionCfg

from .mdp import reset_root_state_along_track

##############################
###### COMMON CONSTANTS ######
##############################

CORNER_IN_RADIUS = 0.3        # For termination
CORNER_OUT_RADIUS = 2.0       # For termination
LINE_RADIUS = 0.8             # For spawning and reward
STRAIGHT = 0.8                # Shaping
SLIP_THRESHOLD = 0.55         # (rad) For reward
MAX_SPEED = 3.0               # (m/s) For action and reward

###################
###### SCENE ######
###################

@configclass
class DriftTerrainImporterCfg(TerrainImporterCfg):

    height = 0.0
    prim_path = "/World/ground"
    terrain_type="plane"
    collision_group = -1
    physics_material=sim_utils.RigidBodyMaterialCfg( # Material for carpet
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.1,
        dynamic_friction=1.0,
    )
    debug_vis=False

@configclass
class MushrDriftSceneCfg(InteractiveSceneCfg):
    """Configuration for a Mushr car Scene with racetrack terrain with no sensors"""

    terrain = DriftTerrainImporterCfg()

    robot: ArticulationCfg = MUSHR_SUS_2WD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    def __post_init__(self):
        """Post intialization."""
        super().__post_init__()
        self.robot.init_state = self.robot.init_state.replace(
            pos=(0.0, 0.0, 0.0),
        )

#####################
###### EVENTS #######
#####################

@configclass
class DriftEventsCfg:
    # on startup

    reset_root_state = EventTerm(
        func=reset_root_state_along_track,
        params={
            "track_radius": LINE_RADIUS,
            "track_straight_dist": STRAIGHT,
            "num_points": 20,
            "pos_noise": 0.5,
            "yaw_noise": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
        mode="reset",
    )

@configclass
class DriftEventsRandomCfg(DriftEventsCfg):

    change_wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.5),
            "dynamic_friction_range": (0.3, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 20,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_link"),
            "make_consistent": True,
        },
    )

    randomize_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*back.*throttle"]),
            "damping_distribution_params": (10.0, 50.0),
            "operation": "abs",
        },
    )

    push_robots_hf = EventTerm( # High frequency small pushes
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.1, 0.4),
        params={
            "velocity_range":{
                "x": (-0.1, 0.1),
                "y": (-0.03, 0.03),
                "yaw": (-0.3, 0.3)
            },
        },
    )

    push_robots_lf = EventTerm( # Low frequency large pushes
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.8, 1.2),
        params={
            "velocity_range":{
                "yaw": (-0.6, 0.6)
            },
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (0.3, 0.5),
            "operation": "add",
            "distribution": "uniform",
        },
    )

######################
###### REWARDS #######
######################

def track_progress_rate(env):
    '''Estimate track progress by positive z-axis angular velocity around the environment'''
    asset : RigidObject = env.scene[SceneEntityCfg("robot").name]
    root_ang_vel = asset.data.root_link_ang_vel_w # this is different than the mdp one
    progress_rate = root_ang_vel[..., 2]
    return progress_rate

def vel_dist(env, speed_target: float=MAX_SPEED, offset: float=-MAX_SPEED**2):
    lin_vel = mdp.base_lin_vel(env)
    ground_speed = torch.norm(lin_vel[..., :2], dim=-1)
    speed_dist = (ground_speed - speed_target) ** 2 + offset
    return speed_dist # speed target

def cross_track_dist(env,
                    straight: float,
                    track_radius: float=(CORNER_IN_RADIUS + CORNER_OUT_RADIUS) / 2,
                    offset: float= -1.,
                    p: float=1.0):
    """Measures distance from a given radius on the track. Defaults
             to the middle of the track."""
    poses = mdp.root_pos_w(env)
    on_straights = torch.abs(poses[...,1]) < straight
    sq_ctd = torch.where(on_straights,
                torch.where(poses[...,0] > 0, # Straights
                    (poses[...,0] - track_radius)**2, # Quadrant 1
                    (poses[...,0] + track_radius)**2), # Quadrant 2
                torch.where(poses[...,1] > 0, # Corners
                    (torch.sqrt((poses[...,1] - straight)**2 + poses[...,0]**2) - track_radius)**2, # Positive y Turn
                    (torch.sqrt((poses[...,1] + straight)**2 + poses[...,0]**2) - track_radius)**2 # Negative y Turn
                )
    )
    ctd = torch.sqrt(sq_ctd) + offset

    return torch.pow(ctd, p)

def energy_through_turn(env, straight: float):
    poses = mdp.root_pos_w(env)
    speed = torch.norm(mdp.base_lin_vel(env), dim=-1)
    energy_through_turn = torch.where(torch.abs(poses[...,1]) > straight, speed**2, 0.)
    return energy_through_turn

def in_range(env, straight, corner_in_radius):
    poses = mdp.root_pos_w(env)
    penalty = torch.where(torch.abs(poses[...,1]) < straight,
                torch.where(torch.abs(poses[...,0]) < corner_in_radius, 1, 0),
                torch.where(poses[...,1] > 0,
                    torch.where((poses[...,1] - straight)**2 + poses[...,0]**2 < corner_in_radius**2, 1, 0),
                    torch.where((poses[...,1] + straight)**2 + poses[...,0]**2 < corner_in_radius**2, 1, 0)))
    return penalty

def off_track(env, straight, corner_out_radius):
    poses = mdp.root_pos_w(env)
    penalty = torch.where(torch.abs(poses[...,1]) < straight,
                torch.where(torch.abs(poses[...,0]) > corner_out_radius, 1, 0),
                torch.where(poses[...,1] > 0,
                    torch.where((poses[...,1] - straight)**2 + poses[...,0]**2 > corner_out_radius**2, 1, 0),
                    torch.where((poses[...,1] + straight)**2 + poses[...,0]**2 > corner_out_radius**2, 1, 0)))
    return penalty

def side_slip(env, min_thresh: float, max_thresh: float, min_vel_x: float=0.5):
    vel = mdp.base_lin_vel(env)
    slip_angle = torch.abs(torch.atan2(vel[...,1], vel[...,0]))
    valid_angle = torch.where(torch.logical_or(
        torch.abs(vel[..., 0]) < min_vel_x, slip_angle > max_thresh),
        0.0, slip_angle
    )
    # Discount lateral vel from steering
    valid_angle = torch.where(valid_angle < min_thresh, 0.0, valid_angle)
    # Clamp unstable angles. Harder than zeroing for heavy, unstable vehicles
    # valid_angle = torch.clamp(valid_angle, max=max_thresh)
    return valid_angle

def turn_left_go_right(env, ang_vel_thresh: float=torch.pi/4):
    asset = env.scene[SceneEntityCfg("robot").name]
    steer_joints = asset.find_joints(".*_steer")[0]
    steer_joint_pos = mdp.joint_pos(env)[..., steer_joints].mean(dim=-1)
    ang_vel = mdp.base_ang_vel(env)[..., 2]
    ang_vel = torch.clamp(ang_vel, max=ang_vel_thresh, min=-ang_vel_thresh)
    tlgr = steer_joint_pos * ang_vel * -1.
    rew = torch.clamp(tlgr, min=0.)
    return rew

@configclass
class DriftRewardsCfg:

    """Reward terms for the MDP."""
    side_slip = RewTerm(
        func=side_slip,
        weight=10., # Increases over time
        params={
            "min_thresh": 0.25, # 15 degrees
            "max_thresh": SLIP_THRESHOLD,
            "min_vel_x": 1.0
        }
    )

    vel = RewTerm(
        func=vel_dist,
        weight=-5.,
        params={
            "speed_target": MAX_SPEED,
            # "offset": 1.,
        }
    )

    progress = RewTerm(
        func=track_progress_rate,
        weight=40.
    )

    tlgr = RewTerm(
        func=turn_left_go_right,
        params={"ang_vel_thresh": 1.},
        weight=0.,
    )

    turn_energy = RewTerm(
        func=energy_through_turn,
        weight=20.,
        params={"straight": STRAIGHT}
    )

    ## Penalties

    cross_track = RewTerm(
        func=cross_track_dist,
        weight=-50.,
        params={
            "straight": STRAIGHT,
            "track_radius": LINE_RADIUS,
            "p": 1,
            "offset": -1.,
        }
    )

    term_pens = RewTerm(
        func = mdp.rewards.is_terminated_term,
        params={"term_keys": ["out_of_bounds"]},
        weight = -5000.,
    )


########################
###### CURRICULUM ######
########################

@configclass
class DriftCurriculumCfg:

    more_slip = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "side_slip",
            "increase": 20.,
            "episodes_per_increase": 20,
            "max_increases": 10,
        }
    )

    more_tlgr = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "tlgr",
            "increase": 10.,
            "episodes_per_increase": 20,
            "max_increases": 5,
        }
    )

    more_term_pens = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "term_pens",
            "increase": -1000.,
            "episodes_per_increase": 50,
            "max_increases": 5,
        }
    )

##########################
###### TERMINATION #######
##########################

def cart_off_track(env, straight:float, corner_in_radius:float, corner_out_radius:float):
    out = torch.logical_or(
        off_track(env, straight, corner_out_radius) > 0.5,
        in_range(env, straight, corner_in_radius) > 0.5
    )
    return out

@configclass
class DriftTerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    out_of_bounds = DoneTerm(
        func=cart_off_track,
        params={
            "straight": STRAIGHT,
            "corner_in_radius": CORNER_IN_RADIUS,
            "corner_out_radius": CORNER_OUT_RADIUS
        }
    )

######################
###### RL ENV ########
######################

@configclass
class MushrDriftRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    seed: int = 42
    num_envs: int = 1024
    env_spacing: float = 0.

    # Basic Settings
    observations: BlindObsCfg = BlindObsCfg()
    actions: MushrRWDActionCfg = MushrRWDActionCfg()

    # MDP Settings
    rewards: DriftRewardsCfg = DriftRewardsCfg()
    events: DriftEventsCfg = DriftEventsRandomCfg()
    terminations: DriftTerminationsCfg = DriftTerminationsCfg()
    curriculum: DriftCurriculumCfg = DriftCurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # viewer settings
        self.viewer.eye = [4., -4., 4.]
        self.viewer.lookat = [0.0, 0.0, 0.]

        self.sim.dt = 0.005  # 200 Hz
        self.decimation = 4  # 50 Hz
        self.sim.render_interval = 20 # 10 Hz
        self.episode_length_s = 5
        self.actions.throttle_steer.scale = (MAX_SPEED, 0.488)

        self.observations.policy.enable_corruption = True

        # Scene settings
        self.scene = MushrDriftSceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing,
        )

######################
###### PLAY ENV ######
######################

@configclass
class MushrDriftPlayEnvCfg(MushrDriftRLEnvCfg):
    """no terminations"""

    events: DriftEventsCfg = DriftEventsRandomCfg(
        reset_root_state = EventTerm(
            func=reset_root_state_along_track,
            params={
                "dist_noise": 0.,
                "yaw_noise": 0.,
            },
            mode="reset",
        )
    )

    rewards: DriftRewardsCfg = None
    terminations: DriftTerminationsCfg = None
    curriculum: DriftCurriculumCfg = None

    def __post_init__(self):
        super().__post_init__()