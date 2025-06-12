import torch
import cv2
import numpy as np

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import (
    AdditiveUniformNoiseCfg as Unoise,
    AdditiveGaussianNoiseCfg as Gnoise,
)

from wheeledlab.envs.mdp import root_euler_xyz

### Commonly used observation terms with emprically determined noise levels

@configclass
class BlindObsCfg:
    """Default observation configuration (no sensors; no corruption)"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        root_pos_w_term = ObsTerm( # meters
            func=mdp.root_pos_w,
            noise=Gnoise(mean=0., std=0.1),
        )

        root_euler_xyz_term = ObsTerm( # radians
            func=root_euler_xyz,
            noise=Gnoise(mean=0., std=0.1),
        )

        base_lin_vel_term = ObsTerm( # m/s
            func=mdp.base_lin_vel,
            noise=Gnoise(mean=0., std=0.5),
        )

        base_ang_vel_term = ObsTerm( # rad/s
            func=mdp.base_ang_vel,
            noise=Gnoise(std=0.4),
        )

        last_action_term = ObsTerm( # [m/s, (-1, 1)]
            func=mdp.last_action,
            clip=(-1., 1.), # TODO: get from ClipAction wrapper or action space
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()