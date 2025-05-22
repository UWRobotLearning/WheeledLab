from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg, ActuatorNetMLPCfg, ActuatorNetMLPModCfg

## HOUND actuator config
THROTTLE_DCMOTOR_CFG = DCMotorCfg(
        joint_names_expr=[".*throttle"],
        saturation_effort=1.05,
        # effort_limit=0.25,
        effort_limit=0.28,
        velocity_limit=450.,
        stiffness=0,
        damping=1000.,
        friction=0.0,
    )


HOUND_ACTUATOR_CFG = {
    "steering_joints": ImplicitActuatorCfg(
        joint_names_expr= ["front_left_wheel_steer", "front_right_wheel_steer"],
        velocity_limit=10.0,
        effort_limit=3.2,
        stiffness=100.0,
        damping=10.,
        friction=0.,
    ),
    "throttle_joints": DCMotorCfg(
        joint_names_expr=[".*throttle"],
        saturation_effort=1.05,
        # effort_limit=0.25,
        effort_limit=0.28,
        velocity_limit=450.,
        stiffness=0,
        damping=1000.,
        friction=0.0,
    )
}

HOUND_SUS_ACTUATOR_CFG = { # 4WD
    **HOUND_ACTUATOR_CFG,
    "suspension": ImplicitActuatorCfg(
        joint_names_expr=[".*_suspension"],
        effort_limit=None, # Passive joint
        velocity_limit=None,
        stiffness=1e8,
        damping=0.,
        friction=.5,
    ),
}

## HOUND 2WD Configuration with suspension
HOUND_SUS_2WD_ACTUATOR_CFG = {
    "steering_joints": HOUND_SUS_ACTUATOR_CFG["steering_joints"],
    "suspension": HOUND_SUS_ACTUATOR_CFG["suspension"],
    "throttle_joints": HOUND_SUS_ACTUATOR_CFG["throttle_joints"].replace(
        joint_names_expr=["back_.*throttle"],
        effort_limit=0.6, # More torque for two wheel drive
    ),
    "passive_joints": ImplicitActuatorCfg(
        joint_names_expr=["front_.*throttle"],
        effort_limit=None,
        velocity_limit=None,
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
    ),
}


# Define MLP actuator configuration for throttle joints
THROTTLE_MLP_MOD_CFG = ActuatorNetMLPModCfg(
    joint_names_expr=[".*throttle"],
    network_file="/home/tongo/WheeledLab/source/wheeledlab_assets/wheeledlab_assets/MLP_model/dummy_throttle_mlp.pt",  # Replace with actual model path
    pos_scale=1.0,  # Adjust based on your training data normalization
    vel_scale=0.1,
    torque_scale=1.0,
    input_order="pos_vel",  # or "vel_pos" depending on your model
    input_idx=[0, 1],  # Current and previous timesteps (adjust as needed)
    # w parameters (used for clipping)
    saturation_effort=1.05,
    effort_limit=0.25,
    velocity_limit=450.0,
    damping=1000.0,
    friction=0.0
)