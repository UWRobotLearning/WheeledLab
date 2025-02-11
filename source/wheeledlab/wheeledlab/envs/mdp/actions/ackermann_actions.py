# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg

class AckermannAction(ActionTerm):
    r"""
    Action term for controlling Ackermann steering vehicles.

    This class encapsulates the logic required to process and apply actions for an Ackermann steering vehicle
    in a simulation environment. It handles the transformation of raw actions, including clipping or applying
    a tanh function to bound the actions within a specified range, and computes the appropriate steering angles
    and wheel velocities.

    Attributes:
    ----------
    cfg : actions_cfg.AckermannActionCfg
        The configuration of the action term, including parameters like scaling, offset, and bounding strategy.
    _asset : Articulation
        The articulation asset on which the action term is applied.
    _scale : torch.Tensor
        The scaling factor applied to the input action. Shape is (1, 2).
    _offset : torch.Tensor
        The offset applied to the input action. Shape is (1, 2).
    _bounding_strategy : str | None
        The strategy used to bound the actions. Can be 'clip' or 'tanh'. If None, no bounding is applied.
    _raw_actions : torch.Tensor
        Tensor to store raw actions before processing.
    _processed_actions : torch.Tensor
        Tensor to store processed actions after applying the bounding strategy, scaling, and offset.
    base_length : torch.Tensor
        The length of the vehicle base.
    base_width : torch.Tensor
        The width of the vehicle base.
    wheel_rad : torch.Tensor
        The radius of the vehicle wheels.

    Methods:
    -------
    process_actions(actions):
        Processes the raw actions based on the bounding strategy, scaling, and offset.

    apply_actions():
        Applies the processed actions to the articulation asset.

    calculate_ackermann_angles_and_velocities(target_steering_angle_rad, target_velocity):
        Calculates the steering angles for the left and right front wheels and the wheel velocities based on the
        Ackermann steering geometry.
    """

    cfg: actions_cfg.AckermannActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, 2)."""
    _offset: torch.Tensor
    """The offset applied to the input action. Shape is (1, 2)."""
    _bounding_strategy: str | None
    """The strategy used to bound the actions. Can be 'clip' or 'tanh'. If None, no bounding is applied."""

    def __init__(self, cfg: actions_cfg.AckermannActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        wheel_ids, wheel_names = self._asset.find_joints(cfg.wheel_joint_names)
        self._wheel_ids = wheel_ids
        self._wheel_names = wheel_names

        steering_ids, steering_names = self._asset.find_joints(cfg.steering_joint_names)
        self._steering_ids = steering_ids
        self._steering_names = steering_names

        # Action scaling and offset
        self._scale = torch.tensor(cfg.scale, device=self.device, dtype=torch.float32)
        self._offset = torch.tensor(cfg.offset, device=self.device, dtype=torch.float32)
        self._bounding_strategy = cfg.bounding_strategy

        # Initialize tensors for actions
        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=self.device)  # Placeholder for [velocity, steering_angle]

        self.base_length = torch.tensor(cfg.base_length, device=self.device)
        self.base_width = torch.tensor(cfg.base_width, device=self.device)
        self.wheel_rad = torch.tensor(cfg.wheel_radius, device=self.device)


    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        # store the raw actions
        self._raw_actions[:] = actions

        if self._bounding_strategy == 'clip':
            self._processed_actions = torch.clip(actions, min=-1.0, max=1.0) * self._scale + self._offset

        elif self._bounding_strategy == 'tanh':
            self._processed_actions = torch.tanh(actions) * self._scale + self._offset

        else:
            self._processed_actions = actions * self._scale + self._offset

        if self.cfg.no_reverse:
            self._processed_actions[:, 0] = torch.clamp(self._processed_actions[:, 0], min=0.0)


    def apply_actions(self):

        left_rotator_angle, right_rotator_angle, wheel_speeds = self._calculate_ackermann_angles_and_velocities(
            target_velocity=self.processed_actions[:, 0], # Velocity for all cars
            target_steering_angle=self.processed_actions[:, 1] # Steering angle for all cars
        )
        front_wheel_angles = torch.stack([left_rotator_angle, right_rotator_angle], dim=1)

        self._asset.set_joint_velocity_target(wheel_speeds, joint_ids=self._wheel_ids)
        self._asset.set_joint_position_target(front_wheel_angles, joint_ids=self._steering_ids)




    def _calculate_ackermann_angles_and_velocities(self, target_steering_angle, target_velocity):
        """
        Calculates the steering angles for the left and right front wheels and the wheel velocities based on the
        Ackermann steering geometry.

        Parameters:
        ----------
        target_steering_angle : torch.Tensor
            Target steering angles in radians for each environment.
        target_velocity : torch.Tensor
            Target velocities for each environment in meters/second.

        Returns:
        -------
        delta_left : torch.Tensor
            Steering angles for the left front wheels.
        delta_right : torch.Tensor
            Steering angles for the right front wheels.
        wheel_speeds : torch.Tensor
            Speeds for each wheel.
        """
        L = self.base_length
        W = self.base_width
        wheel_radius = self.wheel_rad

        # Ensure inputs are PyTorch tensors
        target_steering_angle = target_steering_angle.float()
        target_velocity = target_velocity.float()

        # Calculating the turn radius from the steering angle
        tan_steering = torch.tan(target_steering_angle)
        R = torch.where(tan_steering == 0, torch.full_like(tan_steering, 1e6), L / tan_steering)

        # Calculate the steering angles for the left and right front wheels in radians
        delta_left = torch.atan(L / (R - W / 2))
        delta_right = torch.atan(L / (R + W / 2))

        # Assuming the rear wheels follow the path's radius adjusted for their position
        R_rear_left = torch.sqrt((R - W/2)**2 + L**2)
        R_rear_right = torch.sqrt((R + W/2)**2 + L**2)

        # Velocity adjustment based on wheel's distance from the IC
        v_front_left = target_velocity * torch.abs(R_rear_left / (R*wheel_radius))
        v_front_right = target_velocity * torch.abs(R_rear_right / (R*wheel_radius))

        v_back_left = target_velocity * torch.abs((R - W/2) / (R*wheel_radius))
        v_back_right = target_velocity * torch.abs((R + W/2) / (R*wheel_radius))

        # Calculate target rotation for each wheel based on its velocity
        wheel_speeds = torch.stack([v_back_left, v_back_right, v_front_left, v_front_right], dim=1)

        return delta_left, delta_right, wheel_speeds