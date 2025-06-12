import torch
import numpy as np
import isaaclab.utils.math as math_utils

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.terrains import TerrainImporter
from ..utils import find_nearest_waypoint

def reset_root_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    # valid_posns_and_rots: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # access the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    if not hasattr(env, '_map_levels'):
        env._map_levels = torch.zeros(env.num_envs, 
                                    dtype=torch.long,
                                    device=env.device)
        
    env._map_levels[env_ids] = torch.tensor(np.floor(np.random.rand(len(env_ids))*len(env.cfg.scene.terrain.traversability_hashmap_list)), device = env.device, dtype=torch.long)

    # valid_poses = terrain.cfg.generate_poses_from_init_points(env, env_ids)
    valid_poses, current_idx_np = terrain.cfg.generate_random_poses(env=env, env_ids=env_ids, num_poses=len(env_ids))
    current_idx = torch.tensor(current_idx_np, dtype=torch.long, device=env.device)

    # Tensorizes the valid poses
    posns = torch.stack(list(map(lambda x: torch.tensor(x.pos, device=env.device), valid_poses))).float()
    oris = list(map(lambda x: torch.deg2rad(torch.tensor(x.rot_euler_xyz_deg, device=env.device)), valid_poses))
    oris = torch.stack([math_utils.quat_from_euler_xyz(*ori) for ori in oris]).float()
    lin_vels = torch.stack(list(map(lambda x: torch.tensor(x.lin_vel, device=env.device), valid_poses))).float()
    ang_vels = torch.stack(list(map(lambda x: torch.tensor(x.ang_vel, device=env.device), valid_poses))).float()

    positions = posns
    positions += asset.data.default_root_state[env_ids, :3]
    orientations = oris

    lin_vels = lin_vels
    lin_vels += asset.data.default_root_state[env_ids, 7:10]
    ang_vels = ang_vels
    ang_vels += asset.data.default_root_state[env_ids, 10:13]

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.cat([lin_vels, ang_vels], dim=-1), env_ids=env_ids)

    # new position
    # pos_xy_world = asset.data.root_pos_w[env_ids, :2]

    # Get waypoints (ensure float32 for consistency)
    # waypoints_xy_world = torch.tensor(
    #     env.scene.terrain.cfg.waypoints_list[0], 
    #     device=env.device,
    #     dtype=torch.float32
    # )[:, :2]

    # inner_xy_world = torch.tensor(
    #     env.scene.terrain.cfg.inner_list[0], 
    #     device=env.device,
    #     dtype=torch.float32
    # )[:, :2]

    # current_idx, _ = find_nearest_waypoint(inner_xy_world, pos_xy_world)
    # current_waypoint = waypoints_xy_world[current_idx]

    if not hasattr(env, '_initial_waypoint_indices'):
        env._initial_waypoint_indices = torch.zeros(env.num_envs, 
                                                dtype=torch.long,
                                                device=env.device)
        
    if not hasattr(env, '_reset_env_bool'):
        env._reset_env_bool = torch.ones(  # Tracks where to insert the next index
            env.num_envs,
            dtype=torch.bool,
            device=env.device
        )

    if not hasattr(env, '_history_waypoint_indices'):
        env._history_length = 10  # Store last 10 waypoints
        env._history_waypoint_indices = torch.zeros(
            (env.num_envs, env._history_length), 
            dtype=torch.long,
            device=env.device
        )

    # set boolean so that it knows it just resetted
    env._initial_waypoint_indices[env_ids] = current_idx.clone()  
    env._reset_env_bool[env_ids] = torch.ones(len(env_ids), dtype=bool, device=env.device)
    env._history_waypoint_indices[env_ids, :] =  torch.zeros(env._history_length, dtype=torch.long, device=env.device)

 
