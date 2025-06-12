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
from wheeledlab_assets import F1TENTH_CFG
import wheeledlab_tasks.drifting.mushr_drift_env_cfg as mushr_drift_cfg
from wheeledlab_tasks.common import BlindObsCfg, F1Tenth4WDActionCfg
from .disable_lidar import disable_all_lidars

from .mdp import reset_root_state_along_track

##############################
###### COMMON CONSTANTS ######
##############################
# Reusing Mushr drift environment constants
CORNER_IN_RADIUS = 0.3        # For termination (inner track radius threshold)
CORNER_OUT_RADIUS = 2.0       # For termination (outer track radius threshold)
LINE_RADIUS = 0.8             # For spawning and reward (half track width on straights)
STRAIGHT = 0.8                # Straight segment half-length for track shape
SLIP_THRESHOLD = 0.55         # (rad) maximum considered slip angle for reward
MAX_SPEED = 3.0               # (m/s) target speed for action scaling and reward

###################
###### SCENE ######
###################

@configclass
class F1TenthDriftSceneCfg(mushr_drift_cfg.MushrDriftSceneCfg):
    # Override robot
    robot: AssetBaseCfg = F1TENTH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

#####################
###### EVENTS #######
#####################

# Reuse the drift event logic from Mushr environment, but adapt actuator targeting for 4WD
@configclass
class F1TenthDriftEventsRandomCfg(mushr_drift_cfg.DriftEventsRandomCfg):
    """Randomized events for F1Tenth drifting, extending base drift events."""

    # Override randomize_gains to target all four wheel motors (front and back)
    randomize_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_(back|front)_.*"]),
            "damping_distribution_params": (10.0, 50.0),
            "operation": "abs",
        },
    )

    change_wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.5),
            "dynamic_friction_range": (0.3, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 20,
            "asset_cfg": SceneEntityCfg("robot", body_names="wheel.*"),
            "make_consistent": True,
        },
    )
    # (All other event terms such as wheel friction changes, pushes, etc.,
    # are inherited from DriftEventsRandomCfg without modification.)
    

    kill_lidar = EventTerm(
        func=disable_all_lidars,
        mode="startup",
        params={}          
    )

######################
###### REWARDS #######
######################

def turn_left_go_right_f1(env, ang_vel_thresh: float = torch.pi/4):
    """
    Reward component: turning wheels left while the car's angular velocity is to the right (and vice versa).
    Adapted for F1Tenth (uses steering joints named "rotator_left/right" instead of the previous naming convention).
    Similar to Mushr's turn_left_go_right.
    """
    asset = env.scene[SceneEntityCfg("robot").name]
    # Updated regex: now matches "rotator_left" and "rotator_right"
    steer_joints = asset.find_joints("rotator_.*")[0]
    steer_joint_pos = mdp.joint_pos(env)[..., steer_joints].mean(dim=-1)
    ang_vel = mdp.base_ang_vel(env)[..., 2]
    ang_vel = torch.clamp(ang_vel, max=ang_vel_thresh, min=-ang_vel_thresh)
    tlgr = steer_joint_pos * ang_vel * -1.0
    rew = torch.clamp(tlgr, min=0.0)
    return rew

@configclass
class F1TenthDriftRewardsCfg(mushr_drift_cfg.DriftRewardsCfg):
    """Reward terms for F1Tenth drifting, reusing Mushr drift rewards with adapted TLGR term."""
    # Inherit all drift reward terms (side_slip, velocity penalty, progress, etc.)
    # Override the turn_left_go_right (tlgr) reward to use F1Tenth steering joints
    tlgr = RewTerm(
        func=turn_left_go_right_f1,
        params={"ang_vel_thresh": 1.0},
        weight=0.0,
    )
    # (Other reward terms remain identical to DriftRewardsCfg)

########################
###### CURRICULUM ######
########################

##########################
###### TERMINATION #######
##########################

######################
###### RL ENV ########
######################
@configclass
class F1TenthDriftRLEnvCfg(mushr_drift_cfg.MushrDriftRLEnvCfg):
    """RL environment configuration for drifting task with the F1Tenth robot."""

    # Environment settings
    seed: int = 42
    num_envs: int = 256
    env_spacing: float = 0.0

    # MDP Components
    actions: F1Tenth4WDActionCfg = F1Tenth4WDActionCfg()          # 4WD throttle/steer actions
    rewards: F1TenthDriftRewardsCfg = F1TenthDriftRewardsCfg()       # use adapted drift rewards
    events: F1TenthDriftEventsRandomCfg = F1TenthDriftEventsRandomCfg()    # use adapted random events

    def __post_init__(self):
        """Post initialization configuration for simulation and viewer."""

        # Viewer camera setup (same as Mushr drift)
        self.viewer.eye = [4.0, -4.0, 4.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]
        # Simulation time-step and frequency settings
        self.sim.dt = 0.005             # 200 Hz physics simulation
        self.decimation = 4             # 50 Hz control (action) frequency
        self.sim.render_interval = 20   # render every 20 steps (~10 Hz)
        self.episode_length_s = 5       # each episode lasts 5 seconds
        # Scale actions: (MAX_SPEED, max_steering_angle)
        self.actions.throttle_steer.scale = (MAX_SPEED, 0.488)
        # Enable sensor noise corruption in observations (for realism)
        self.observations.policy.enable_corruption = True
        # Scene setup with F1Tenth robot
        self.scene = F1TenthDriftSceneCfg(num_envs=self.num_envs, env_spacing=self.env_spacing)
