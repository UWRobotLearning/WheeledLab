from isaaclab.utils import configclass

from dataclasses import MISSING

from . import TrainConfig, RunConfig, AgentSetup

@configclass
class RLTrainConfig(TrainConfig):
    agent_n_steps: int = 200            # Number of steps to run the agent
    num_iterations: int = 2048       # RL Policy training iterations
    rl_algo_lib: str = MISSING            # RL Algorithm library
    rl_algo_class: str = MISSING          # RL Algorithm class
    set_env_step: int = 0   # Load a previously trained model at this environment step

@configclass
class RslRlRunConfig(RunConfig):
    train: RLTrainConfig = RLTrainConfig(
        rl_algo_lib="rsl",
        rl_algo_class="ppo"
    )
    agent_setup: AgentSetup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point"
    )
    train.log.video_interval = 15000
    train.set_env_step = train.load_run_checkpoint

@configclass
class SB3RLRunConfig(RunConfig):
    train: RLTrainConfig = RLTrainConfig(
        rl_algo_lib="sb3",
        rl_algo_class="ppo"
    )
    agent_setup: AgentSetup = AgentSetup(
        entry_point="sb3_rl_cfg_entry_point"
    )
