import torch

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from collections.abc import Sequence
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)

def increase_reward_weight_over_time(
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        reward_term_name : str,
        increase : float,
        first_episode_increase : int = 1,
        episodes_per_increase : int = 1,
        max_num_increases: int = torch.inf,
        ) -> torch.Tensor:
    """
    Increase the weight of a reward term after some amount of given time in episodes.
    Default amount of time is one episode.
    Stops increasing the weight after `stop_after_n_changes` changes. Defaults to inf.
    """
    num_episodes = env.common_step_counter // env.max_episode_length
    num_increases = num_episodes // episodes_per_increase

    if num_episodes < first_episode_increase:
        return

    if num_increases > max_num_increases:
        return # do nothing

    if env.common_step_counter % env.max_episode_length != 0:
        return # only process at the beginning of an episode (not per step)

    if (num_episodes + 1) % episodes_per_increase == 0: # discount the first episode
        term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        term_cfg.weight += increase
        env.reward_manager.set_term_cfg(reward_term_name, term_cfg)

def multiplier_reward_weight_over_time(
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        reward_term_name : str,
        multiplier : float,
        first_episode_multiplier : int = 1,
        episodes_per_multiplier : int = 1,
        max_num_multipliers: int = torch.inf,
        ) -> torch.Tensor:
    """
    multiplier the weight of a reward term after some amount of given time in episodes.
    Default amount of time is one episode.
    Stops increasing the weight after `stop_after_n_changes` changes. Defaults to inf.
    """
    num_episodes = env.common_step_counter // env.max_episode_length
    num_multipliers = num_episodes // episodes_per_multiplier

    if num_episodes < first_episode_multiplier:
        return

    if num_multipliers > max_num_multipliers:
        return # do nothing

    if env.common_step_counter % env.max_episode_length != 0:
        return # only process at the beginning of an episode (not per step)

    if (num_episodes + 1) % episodes_per_multiplier == 0: # discount the first episode
        term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        term_cfg.weight *= multiplier
        env.reward_manager.set_term_cfg(reward_term_name, term_cfg)