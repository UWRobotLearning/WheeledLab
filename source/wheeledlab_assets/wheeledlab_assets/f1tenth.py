import os
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

# Hardcoded paths to your USD and URDF files in the WheeledLab repo:
USD_PATH = "source/wheeledlab_assets/data/Robots/F1Tenth/f1tenth_robot.usd"
URDF_PATH = "source/wheeledlab_assets/data/Robots/F1Tenth/f1tenth.urdf"

# Example actuator configuration (customize for F1Tenth).
F1TENTH_ACTUATOR_CFG = {
    "steering_joints": ImplicitActuatorCfg(
        joint_names_expr=["front_left_wheel_steer", "front_right_wheel_steer"],
        velocity_limit=12.0,
        effort_limit=2.5,
        stiffness=120.0,
        damping=8.0,
        friction=0.0,
    ),
    "throttle_joints": DCMotorCfg(
        joint_names_expr=[".*throttle"],
        saturation_effort=1.0,
        effort_limit=0.3,
        velocity_limit=500.0,
        stiffness=0,
        damping=1100.0,
        friction=0.0,
    ),
}

# (Optional) If you have a suspension variant, define a second config:
F1TENTH_SUS_ACTUATOR_CFG = {
    "steering_joints": F1TENTH_ACTUATOR_CFG["steering_joints"],
    "throttle_joints": F1TENTH_ACTUATOR_CFG["throttle_joints"].replace(
        joint_names_expr=["back_.*throttle"],
        effort_limit=0.45,
    ),
    "passive_joints": ImplicitActuatorCfg(
        joint_names_expr=["front_.*throttle"],
        effort_limit=None,
        velocity_limit=None,
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
    ),
    "suspension": ImplicitActuatorCfg(
        joint_names_expr=[".*_suspension"],
        effort_limit=None,
        velocity_limit=None,
        stiffness=1e8,
        damping=0.0,
        friction=0.6,
    ),
}

# Define initial states (joint positions, etc.).
_ZERO_INIT_STATES = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "front_left_wheel_steer": 0.0,
        "front_right_wheel_steer": 0.0,
        "front_left_wheel_throttle": 0.0,
        "front_right_wheel_throttle": 0.0,
        # Add other joints if needed.
    },
)

# Tie together your USD file, physics settings, and initial state.
F1TENTH_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=100000.0,
            max_depenetration_velocity=100.0,
            max_contact_impulse=0.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=_ZERO_INIT_STATES,
    actuators=F1TENTH_ACTUATOR_CFG,
)

class F1Tenth:
    """
    Hardcoded F1Tenth wrapper for WheeledLab, referencing:
      f1tenth_robot.usd & f1tenth.urdf
    in: source/wheeledlab_assets/data/Robots/F1Tenth/
    """

    def __init__(self, prim_path="/World/F1Tenth", use_suspension=False, **kwargs):
        self.prim_path = prim_path
        # Pick the actuator config
        if use_suspension:
            actuator_cfg = F1TENTH_SUS_ACTUATOR_CFG
        else:
            actuator_cfg = F1TENTH_ACTUATOR_CFG
        # Create a copy of F1TENTH_CFG with possible overrides from kwargs
        self.cfg = F1TENTH_CFG.replace(actuators=actuator_cfg, **kwargs)

        # Keep the URDF path on hand if you need it
        self.urdf_path = URDF_PATH

    def spawn(self):
        """Spawn the F1Tenth model into the simulation at self.prim_path."""
        return sim_utils.spawn_articulation(self.cfg, prim_path=self.prim_path)
