import torch

import isaaclab.utils.math as math_utils

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.terrains import TerrainImporter


def reset_root_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    # valid_posns_and_rots: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # access the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # valid_poses = terrain.cfg.generate_poses_from_init_points(env, env_ids)
    valid_poses = terrain.cfg.generate_random_poses(len(env_ids))

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

