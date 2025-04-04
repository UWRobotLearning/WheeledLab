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
from wheeledlab_tasks.common import BlindObsCfg, Mushr4WDActionCfg
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
class DriftTerrainImporterCfg(TerrainImporterCfg):
    """Terrain: a flat plane with carpet-like friction (same as Mushr drift)"""
    height = 0.0
    prim_path = "/World/ground"
    terrain_type = "plane"
    collision_group = -1
    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.1,
        dynamic_friction=1.0,
    )  # carpet-like friction
    debug_vis = False

@configclass
class F1TenthDriftSceneCfg(InteractiveSceneCfg):
    """Scene configuration for F1Tenth drifting on a flat track"""
    # Use same flat terrain and lighting as Mushr drift scene
    terrain = DriftTerrainImporterCfg()
    # Integrate F1Tenth robot by reusing its predefined articulation config&#8203;:contentReference[oaicite:0]{index=0}
    robot: ArticulationCfg = F1TENTH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Add a distant light for illumination (same as Mushr drift)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    def __post_init__(self):
        """Post-initialization to ensure correct spawn position."""
        super().__post_init__()
        # Ensure the robot starts at the origin (F1Tenth default is already (0,0,0))
        self.robot.init_state = self.robot.init_state.replace(pos=(0.0, 0.0, 0.0))

#####################
###### EVENTS #######
#####################
# Reuse the drift event logic from Mushr environment, but adapt actuator targeting for 4WD
@configclass
class F1TenthDriftEventsRandomCfg(mdp.DriftEventsRandomCfg):
    """Randomized events for F1Tenth drifting, extending base drift events."""
    # Override randomize_gains to target all four wheel motors (front and back)
    randomize_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*wheel_(back|front)"]),
            "damping_distribution_params": (10.0, 50.0),
            "operation": "abs",
        },
    )
    # (All other event terms such as wheel friction changes, pushes, etc., 
    # are inherited from DriftEventsRandomCfg without modification.)

######################
###### REWARDS #######
######################
def turn_left_go_right_f1(env, ang_vel_thresh: float = torch.pi/4):
    """
    Reward component: turning wheels left while car's angular velocity is to the right (and vice versa).
    Adapted for F1Tenth (uses steering joints named "*_rotator" instead of "*_steer"). 
    Similar to Mushr's turn_left_go_right.
    """
    asset = env.scene[SceneEntityCfg("robot").name]
    # Find steering joint indices (F1Tenth uses "rotator" joints for steering)
    steer_joints = asset.find_joints(".*_rotator")[0]
    steer_joint_pos = mdp.joint_pos(env)[..., steer_joints].mean(dim=-1)
    ang_vel = mdp.base_ang_vel(env)[..., 2]
    # Limit considered angular velocity to a threshold
    ang_vel = torch.clamp(ang_vel, max=ang_vel_thresh, min=-ang_vel_thresh)
    # Compute reward: product of steering and opposite angular velocity
    tlgr = steer_joint_pos * ang_vel * -1.0
    rew = torch.clamp(tlgr, min=0.0)
    return rew

@configclass
class F1TenthDriftRewardsCfg(mdp.DriftRewardsCfg):
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
# Reuse the same drift curriculum shaping from Mushr (increasing side_slip reward, etc.)
F1TenthDriftCurriculumCfg = mdp.DriftCurriculumCfg

##########################
###### TERMINATION #######
##########################
# Use the same termination conditions as Mushr drift (timeout and off-track)
F1TenthDriftTerminationsCfg = mdp.DriftTerminationsCfg

######################
###### ACTIONS #######
######################
@configclass
class F1Tenth4WDActionCfg(Mushr4WDActionCfg):
    """Action configuration for F1Tenth 4WD, using RCCar4WDActionCfg with F1Tenth's joint names."""
    throttle_steer = Mushr4WDActionCfg.throttle_steer.replace(
        wheel_joint_names=[
            "left_wheel_back", "right_wheel_back",
            "left_wheel_front", "right_wheel_front",
        ],
        steering_joint_names=[
            "left_wheel_rotator", "right_wheel_rotator",
        ],
        base_length=0.365,    
        base_width=0.284,
        wheel_radius=0.05,
        asset_name="robot",
    )


######################
###### RL ENV ########
######################
@configclass
class F1TenthDriftRLEnvCfg(ManagerBasedRLEnvCfg):
    """RL environment configuration for drifting task with the F1Tenth robot."""
    # Environment settings
    seed: int = 42
    num_envs: int = 1024
    env_spacing: float = 0.0

    # MDP Components
    observations: BlindObsCfg = BlindObsCfg()                    # no sensors (blind observations)
    actions: F1Tenth4WDActionCfg = F1Tenth4WDActionCfg()          # 4WD throttle/steer actions
    rewards: mdp.DriftRewardsCfg = F1TenthDriftRewardsCfg()       # use adapted drift rewards
    events: mdp.DriftEventsCfg = F1TenthDriftEventsRandomCfg()    # use adapted random events
    terminations: mdp.DriftTerminationsCfg = F1TenthDriftTerminationsCfg()
    curriculum: mdp.DriftCurriculumCfg = F1TenthDriftCurriculumCfg()

    def __post_init__(self):
        """Post initialization configuration for simulation and viewer."""
        super().__post_init__()
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
