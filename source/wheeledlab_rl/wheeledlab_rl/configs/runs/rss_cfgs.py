from isaaclab.utils import configclass

from wheeledlab_rl.configs import *

@configclass
class RSS_DRIFT_CONFIG(RslRlRunConfig):
    env_setup = EnvSetup(
        num_envs=1024,
        task_name="Isaac-MushrDriftRL-v0"
    )
    train = RLTrainConfig(
        num_iterations=5000,
        rl_algo_lib="rsl",
        rl_algo_class="ppo",
        log=LogConfig(
            video_interval=15000
        ),
    )
    agent_setup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point"
    )

@configclass
class RSS_VISUAL_CONFIG(RslRlRunConfig):
    env_setup = EnvSetup(
        num_envs=512,
        task_name="Isaac-MushrVisualRL-v0"
    )
    train = RLTrainConfig(
        num_iterations=5000,
        rl_algo_lib="rsl",
        rl_algo_class="ppo"
    )
    agent_setup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point"
    )

@configclass
class RSS_ELEV_CONFIG(RslRlRunConfig):
    env_setup = EnvSetup(
        num_envs=1024,
        task_name="Isaac-MushrElevationRL-v0"
    )
    train = RLTrainConfig(
        num_iterations=5000,
        rl_algo_lib="rsl",
        rl_algo_class="ppo"
    )
    agent_setup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point"
    )
