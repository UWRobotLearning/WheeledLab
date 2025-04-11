from isaaclab.utils import configclass

from wheeledlab_rl.configs import (
    EnvSetup, RslRlRunConfig, RLTrainConfig, AgentSetup, LogConfig
)
@configclass
class F1TENTH_DRIFT_CONFIG(RslRlRunConfig):
    env_setup = EnvSetup(
        num_envs=1024,
        task_name="Isaac-F1TenthDriftRL-v0"
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