import torch
import numpy as np
import noise
from scipy.ndimage import gaussian_filter1d, distance_transform_edt
from scipy.interpolate import splprep, splev, griddata

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, FlatPatchSamplingCfg
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh

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

from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.commands import TerrainBasedPose2dCommandCfg, UniformPose2dCommandCfg
from isaaclab.envs.mdp.events import reset_root_state_from_terrain, reset_root_state_uniform

from wheeledlab_assets import WHEELEDLAB_ASSETS_DATA_DIR
from wheeledlab.envs.mdp.observations import root_euler_xyz
from wheeledlab_assets.mushr import MUSHR_SUS_CFG
from wheeledlab.envs.mdp import decrease_reward_weight_over_time
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

######################### 
###### SCENE SETUP ######
#########################

#==== SUBTERRAIN CFGS ====
@configclass
class NoiseHfCfg(HfTerrainBaseCfg):
    """Config for noise-based terrain generation."""

    generate_roads: bool = True
    road_num_nodes: int = 10                        # increase for gnarlier roads
    road_width_range:tuple[float, float] = (.2, .5) # width is narrower at high difficulty
    octaves: int = 3                                # layers of noise; increase for more complex terrain
    freq: float = 250.0                             # higher = smoother/wider hills

    # clipping helps create some flat surfaces and plateaus
    upper_clip_prop: float = 0.25                   # clips hf by upper_clip_prop proportion of ht
    lower_clip_prop: float = 0.25                   # clips hf by lower_clip_prop proportion of ht
    vary_ht: bool = False                           # if true, linearly increases ht as x, y increase.
    ht_range: tuple[float, float] = (0.3, 1.4)      # range across difficulties. Represents max height before clipping

    
    flat_patch_sampling = {
        "init_pos" : # reset
        FlatPatchSamplingCfg(
            num_patches = 256, 
            patch_radius = 0.16,
            max_height_diff = 0.01, # 1cm height variation
            x_range = (-4., 4.),
            y_range = (-4., 4.),
            #z_range = ()
        ),
        "target" : # goal pose sampling
        FlatPatchSamplingCfg(
            num_patches = 512,
            patch_radius = 0.16,
            max_height_diff = 0.1,
            x_range = (-4.5, 4.5),
            y_range = (-4.5, 4.5),
        )
    }

def add_roads(hf, difficulty, horizontal_scale, num_nodes, road_width_range):
    rows, cols = hf.shape
    road_width_px = (road_width_range[1] - difficulty * (road_width_range[1] - road_width_range[0])) / horizontal_scale
    shoulder_width_px = 10.0
    
    # === Random walk ===
    current_row, current_col = rows / 2, cols / 2 # road starts from center for now
    heading = np.random.uniform(0, 2 * np.pi)
    step_dist = rows * 0.8 / num_nodes
    key_pts = [[current_row, current_col]]

    for _ in range(num_nodes - 1):
        best_heading = heading
        min_effort = float("inf")
        candidates = np.linspace(-np.pi / 4, np.pi / 4, 5)

        for angle_off in candidates:
            test_heading = heading + angle_off
            test_row = np.clip(current_row + step_dist * np.sin(test_heading), 0, rows - 1)
            test_col = np.clip(current_col + step_dist * np.cos(test_heading), 0, cols - 1)

            # try to keep road near center
            effort = abs(hf[int(test_row), int(test_col)] - hf[int(current_row), int(current_col)])
            effort += np.sqrt((test_row - rows/2)**2 + (test_col - cols/2)**2) * 0.01 

            if effort < min_effort:
                min_effort, best_heading = effort, test_heading

        heading = best_heading

        current_row += step_dist * np.sin(heading)
        current_col += step_dist * np.cos(heading)
        if (current_col >= cols or current_row >= rows):
            break
        key_pts.append([current_row, current_col])

    key_pts = np.array(key_pts)

    # === Spline path ===
    tck, _ = splprep([key_pts[:,0], key_pts[:,1]], s=0)
    u = np.linspace(0, 1, 500)
    path_row, path_col = splev(u, tck)
    path = np.stack([np.clip(path_row, 0, rows-1), np.clip(path_col, 0, cols-1)], axis=1)

    # === Smooth elevation along road & shoulder ===
    road_z = hf[path[:,0].astype(int), path[:,1].astype(int)]
    road_z = gaussian_filter1d(road_z, sigma=10)

    road_mask = np.zeros((rows, cols), dtype=bool)
    road_mask[path[:,0].astype(int), path[:,1].astype(int)] = True
    distance_field = distance_transform_edt(~road_mask)

    height_map = griddata(path, road_z, (np.indices((rows, cols)).transpose(1, 2, 0)), method="nearest")

    new_hf = hf.copy()
    core = distance_field <= road_width_px
    shoulder = (distance_field > road_width_px) & (distance_field <= road_width_px + shoulder_width_px)
    
    new_hf[core] = height_map[core]
    
    t = (distance_field[shoulder] - road_width_px) / shoulder_width_px
    smooth = 3 * t**2 - 2 * t**3
    new_hf[shoulder] = height_map[shoulder] * (1 - smooth) + hf[shoulder] * smooth

    return new_hf

@height_field_to_mesh
def create_noise_hf(difficulty: float, cfg: NoiseHfCfg):
    """Procedurally generates height field using simplex noise."""
    # === Initialize ===
    rows = int(cfg.size[0] / cfg.horizontal_scale)
    cols = int(cfg.size[1] / cfg.horizontal_scale)
    curr_ht_mult  = cfg.ht_range[0] + (cfg.ht_range[1] - cfg.ht_range[0]) * difficulty
    curr_max_height = curr_ht_mult * (1.0 - cfg.upper_clip_prop)
    curr_floor_clip = curr_ht_mult * cfg.lower_clip_prop
    max_ht_adj = curr_max_height / cfg.vertical_scale
    ht_mult_adj = curr_ht_mult / cfg.vertical_scale
    min_ht_adj = curr_floor_clip / cfg.vertical_scale
    offset = 10000.0 * np.random.rand()

    # == Raw noise ==
    hf = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            hf[i, j] = noise.snoise2(
                (i + offset) / cfg.freq,
                (j + offset) / cfg.freq,
                octaves = cfg.octaves,
            )

    hf = (hf + 1.0) / 2.0 # Raw noise is [-1, 1]

    # == Scaling & clipping ==
    rx = np.linspace(0.2, 1.0, rows)
    ry = np.linspace(0.2, 1.0, cols)
    rxv, ryv = np.meshgrid(rx, ry, indexing = "ij")
    ramp = (rxv + ryv) * 0.5

    if cfg.vary_ht:
        hf = hf * ht_mult_adj * ramp
    else:
        hf = hf * ht_mult_adj
    
    hf = np.clip(hf, min_ht_adj, max_ht_adj)

    if cfg.generate_roads:
        hf = add_roads(hf, difficulty, cfg.horizontal_scale, cfg.road_num_nodes, cfg.road_width_range)

    return (hf - min_ht_adj).astype(np.float32)

#==== TERRAIN CFGS ====
@configclass
class ElevationTerrainGeneratorCfg(TerrainGeneratorCfg):
    vertical_scale = 0.005
    horizontal_scale = 0.02
    border_width = 1.0
    border_height = - 7
    slope_threshold = .5

    size = (10.0, 10.0)
    num_rows = 4
    num_cols = 4

    curriculum = True

    sub_terrains = {
        "noise_terrain" : NoiseHfCfg(
            proportion = 1.0,
            function = create_noise_hf,
            vertical_scale = vertical_scale,
            horizontal_scale = horizontal_scale
        )
    }

@configclass
class ElevationTerrainImporterCfg(TerrainImporterCfg):

    prim_path = "/World/elevation"
    terrain_type = "generator"
    terrain_generator = ElevationTerrainGeneratorCfg()
    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    )
    debug_vis=False

@configclass
class ElevationUSDTerrainImporterCfg(TerrainImporterCfg): # old

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

@configclass
class ElevationUSDSceneCfg(InteractiveSceneCfg): # old
        
    terrain = ElevationUSDTerrainImporterCfg()
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

def yaw_change_onElev(env, threshold_yaw, threshold_z):
    pos = mdp.root_pos_w(env)    
    z_value = pos[..., 2] - 0.19
    ang_vel_yaw = mdp.base_ang_vel(env)[..., 2]    
    condition = (z_value > threshold_z) & (abs(ang_vel_yaw)>threshold_yaw)
    rew = torch.where(condition, 2*abs(ang_vel_yaw)**2, torch.zeros_like(ang_vel_yaw))
    return rew

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
    return rew

def goal_progress_rate(env):
    pos = mdp.root_pos_w(env)
    vel = mdp.root_lin_vel_w(env)
    goal_pos = mdp.generated_commands(env, "goal_pose")
    goal_pos = goal_pos[:, :2] # we only need the x, y coordinates

    vel_vector = vel[:, :2]
    goal_vector = goal_pos - pos[:, :2]
    proj_scal = torch.sum(vel_vector * goal_vector, dim=-1) / torch.norm(goal_vector, dim=-1)

    return proj_scal

def goal_progress(env): 
    pos = mdp.root_pos_w(env) 
    goal_pos = mdp.generated_commands(env, "goal_pose")[:, :2] 
    dist = torch.norm(goal_pos - pos[:, :2], dim=-1) # next-step distance 
    vel = mdp.root_lin_vel_w(env)[:, :2] 
    dt = env.step_dt 
    next_pos = pos[:, :2] + vel * dt 
    next_dist = torch.norm(goal_pos - next_pos, dim=-1) 

    return dist - next_dist

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

@configclass
class ElevationRewardsCfg:

    goal_reached = RewTerm(
        func = mdp.rewards.is_terminated_term,
        params = {"term_keys": ["at_goal"]},
        weight = 2000.0,
    )

    vel_towards_goal = RewTerm(
        func = goal_progress_rate,
        weight = 20.0,
    )
    
    stuck_penalty = RewTerm(
        func = mdp.rewards.is_terminated_term,
        params={"term_keys": ["stuck"]},
        weight = -100.0
    )

    rollover_penalty = RewTerm(
        func = mdp.rewards.is_terminated_term,
        params={"term_keys": ["rollover"]},
        weight = -1000.0
    )

    time_penalty = RewTerm(
        func = mdp.is_alive,
        weight = -1.
    )


########################
###### CURRICULUM ######
########################

@configclass
class ElevationCurriculumCfg:
    """Configuration for the elevation policy curriculum."""

    less_vel_reward = CurrTerm(
        func = decrease_reward_weight_over_time,
        params={
            "reward_term_name" : "vel_towards_goal",
            "decrease": 3.,
            "episodes_per_decrease": 50, 
            "max_decreases": 10,
            "lower_limit": 5.
        }
    )

##########################
###### TERMINATION #######
##########################

def upright_bool(env, thresh_deg):
    return upright_penalty(env, thresh_deg) > 0.0

def is_stuck(env, min_vel, wheel_spin_thr):
    not_moving = forward_vel(env) < min_vel
    throttle_joints_asset = SceneEntityCfg("robot", joint_names=".*throttle")
    joint_vels = mdp.joint_vel(env, asset_cfg=throttle_joints_asset)
    spinning_wheels = torch.sum(joint_vels, dim=-1) > wheel_spin_thr
    return torch.logical_and(not_moving, spinning_wheels)

def close_to_goal(env, dist):
    pos = mdp.root_pos_w(env)
    goal_pos = mdp.generated_commands(env, "goal_pose")
    goal_pos = goal_pos[:, :2]
    curr_dist = torch.norm(goal_pos - pos[:, :2], dim=-1)
    return curr_dist < dist

@configclass
class ElevationTerminationsCfg:
    """Termination terms for the MDP."""

    # Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -1.0}
    )

    # Stuck (unmoving)
    stuck = DoneTerm(
        func=is_stuck,
        params={
            "min_vel": 0.02,
            "wheel_spin_thr": 5.,
        },
    )

    # Stuck (upside down)
    rollover = DoneTerm(
        func=upright_bool,
        params={"thresh_deg": 60.},
    )

    # Reached goal
    at_goal = DoneTerm(
        func=close_to_goal,
        params={"dist": 0.5},
    )


@configclass
class ElevationPlayTerminationsCfg:
    """Terminations for play cfg"""

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
        func=reset_root_state_from_terrain,
        mode="reset",
        params={
            "pose_range": {"x": (-9., 9.), "y": (-9., 9.), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.1, 0.2),
                "y": (0.1, 0.2),
            },
        }
    )

@configclass
class ElevationCommandCfg:
    """Configuration for the elevation commands."""

    goal_pose = TerrainBasedPose2dCommandCfg(
        asset_name="robot",
        ranges=TerrainBasedPose2dCommandCfg.Ranges(
            heading=(-3.14, 3.14),
        ),
        resampling_time_range=(10.0, 10.0),
        simple_heading=True,
        debug_vis=True
    )

@configclass
class MushrElevationRLEnvCfg(ManagerBasedRLEnvCfg):

    seed: int = 42
    num_envs: int = 2048
    env_spacing: float = 0.

    # Basic Settings
    observations: ElevationObsCfg = ElevationObsCfg()
    actions: Mushr4WDActionCfg = Mushr4WDActionCfg()

    # MDP settings
    events : ElevationSceneEventsCfg = ElevationSceneEventsCfg()
    curriculum: ElevationCurriculumCfg = ElevationCurriculumCfg()
    rewards: ElevationRewardsCfg = ElevationRewardsCfg()
    terminations: ElevationTerminationsCfg = ElevationTerminationsCfg()

    commands: ElevationCommandCfg = ElevationCommandCfg()

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
    """Play env from USD"""
    def __post_init__(self):
        super().__post_init__()
        self.terminations: ElevationPlayTerminationsCfg = None
        self.rewards = None
        self.curriculum = None
        self.num_envs: int = 64
        self.seed: int = 67
        self.scene = ElevationUSDSceneCfg(num_envs=self.num_envs, env_spacing=self.env_spacing)
        self.events.set_goal = EventTerm(
            func=reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (0., 0.), "y": (0., 0.), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (0.1, 0.2),
                    "y": (0.1, 0.2),
                },
            }
        )
        self.commands.goal_pose = UniformPose2dCommandCfg(
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