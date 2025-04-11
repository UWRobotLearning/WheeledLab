import gymnasium as gym

########################################
############ DRIFT ENVS ################
########################################

from .drifting import MushrDriftRLEnvCfg, MushrDriftPlayEnvCfg
from .visual import MushrVisualRLEnvCfg, MushrVisualPlayEnvCfg
from .elevation import MushrElevationRLEnvCfg, MushrElevationPlayEnvCfg
import wheeledlab_tasks.drifting.config.agents.mushr as mushr_drift_agents
import wheeledlab_tasks.visual.config.agents.mushr as mushr_visual_agents
import wheeledlab_tasks.elevation.config.agents.mushr as mushr_elevation_agents

gym.register(
    id="Isaac-MushrDriftRL-v0",
    entry_point='isaaclab.envs:ManagerBasedRLEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":MushrDriftRLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{mushr_drift_agents.__name__}.rsl_rl_ppo_cfg:MushrPPORunnerCfg",
        "play_env_cfg_entry_point": MushrDriftPlayEnvCfg
    }
)


gym.register(
    id="Isaac-MushrVisualRL-v0",
    entry_point='isaaclab.envs:ManagerBasedRLEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":MushrVisualRLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{mushr_visual_agents.__name__}.rsl_rl_ppo_cfg:MushrPPORunnerCfg",
        "play_env_cfg_entry_point": MushrVisualPlayEnvCfg
    }
)

gym.register(
    id="Isaac-MushrElevationRL-v0",
    entry_point='isaaclab.envs:ManagerBasedRLEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MushrElevationRLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{mushr_elevation_agents.__name__}.rsl_rl_ppo_cfg:MushrPPORunnerCfg",
        "play_env_cfg_entry_point": MushrElevationPlayEnvCfg
    }
)

#######################################
############ F1TENTH ENVS #############
#######################################

from .drifting import F1TenthDriftRLEnvCfg
import wheeledlab_tasks.drifting.config.agents.f1tenth as f1tenth_drift_agents

gym.register(
    id="Isaac-F1TenthDriftRL-v0",
    entry_point='isaaclab.envs:ManagerBasedRLEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":F1TenthDriftRLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{f1tenth_drift_agents.__name__}.rsl_rl_ppo_cfg:F1TenthPPORunnerCfg",
    }
)