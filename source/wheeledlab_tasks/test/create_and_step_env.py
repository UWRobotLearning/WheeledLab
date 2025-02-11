"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser()
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.envs.utils.spaces import sample_space
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import torch
import gymnasium as gym

import wheeledlab_tasks

def main(task_name: str="Isaac-MushrDriftRL-v0", num_envs: int = 16, num_steps: int = 1000):
    env_cfg = parse_env_cfg(task_name, num_envs=num_envs)
    env = gym.make(task_name, cfg=env_cfg)

    # reset environment
    obs, _ = env.reset()

    # simulate environment for num_steps steps
    with torch.inference_mode():
        for _ in range(num_steps):
            # sample actions according to the defined space
            actions = sample_space(
                env.unwrapped.single_action_space, device=env.unwrapped.device, batch_size=num_envs
            )
            # apply actions
            transition = env.step(actions)

if __name__ == "__main__":
    main()