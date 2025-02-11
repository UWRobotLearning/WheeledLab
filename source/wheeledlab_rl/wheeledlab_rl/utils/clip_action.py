import torch

import gymnasium as gym

class ClipAction(gym.ActionWrapper):
    """ Adapted from https://github.com/openai/gym/blob/master/gym/wrappers/clip_action.py
        Applies torch clipping instead of numpy
    """

    def __init__(self, env: gym.Env):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

    def action(self, action):
        """Clips the action within the valid bounds.

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """
        return torch.clip(action, min=self.action_space.low, max=self.action_space.high)