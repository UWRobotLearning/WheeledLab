"""
Adapted from
https://github.com/Lukas-Malthe-MSc/orbit/blob/project-wrap-up/source/extensions/omni.isaac.orbit/omni/isaac/orbit/envs/mdp/actions/actions_cfg.py
"""

from dataclasses import MISSING
from typing import Literal

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import ackermann_actions, rc_car_actions

@configclass
class AckermannActionCfg(ActionTermCfg):
    """Configuration for the Ackermann steering action term.

    This action term models a vehicle with Ackermann steering, where actions control
    the forward velocity and steering angle of the vehicle.

    Attributes:
        class_type: Reference to the AckermannAction class.
        velocity_joint_name: The name of the joint or mechanism controlling velocity.
        steering_joint_name: The name of the joint or mechanism controlling steering.
        scale: Scale factor for the action components (velocity, steering_angle).
        offset: Offset factor for the action components, useful for calibration.
    """
    class_type: type[ActionTerm] = ackermann_actions.AckermannAction

    wheel_joint_names: list[str] = MISSING
    """The name of the joint or mechanism controlling the vehicle's velocity."""

    steering_joint_names: list[str] = MISSING
    """The name of the joint or mechanism controlling the vehicle's steering angle."""

    scale: tuple[float, float] = (1.0, 1.0)
    """Scale factors for the action components: (velocity_scale, steering_angle_scale). Defaults to (1.0, 1.0)."""

    offset: tuple[float, float] = (0.0, 0.0)
    """Offset factors for the action components: (velocity_offset, steering_angle_offset). Defaults to (0.0, 0.0)."""

    bounding_strategy: str | None = "tanh"
    """The strategy to bound the action values. Defaults to "tanh"."""

    base_length: float = 1.0
    """The length of the vehicle's base. Defaults to 1.0."""

    base_width: float = 1.0
    """The width of the vehicle's base. Defaults to 1.0."""

    wheel_radius: float = 1.0
    """The radius of the vehicle's wheels. Defaults to 1.0."""

    no_reverse: bool = False
    """Whether the vehicle can reverse; sets throttle min to 0 if True. Defaults to False."""


@configclass
class RCCarRWDActionCfg(AckermannActionCfg):

    class_type: type[ActionTerm] = rc_car_actions.RCCarRWDAction


@configclass
class RCCar4WDActionCfg(AckermannActionCfg):

    class_type: type[ActionTerm] = rc_car_actions.RCCar4WDAction
