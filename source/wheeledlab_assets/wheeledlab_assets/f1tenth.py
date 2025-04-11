import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
from isaaclab.assets import ArticulationCfg

from . import WHEELEDLAB_ASSETS_DATA_DIR

# F1Tenth 4WD actuator configuration.
# For 4WD, all throttle joints (front and back) are active.
F1TENTH_4WD_ACTUATOR_CFG = {
    "steering_joints": ImplicitActuatorCfg(
        joint_names_expr=["rotator_(left|right)"],
        velocity_limit=10.0,    # F1Tenth steering is slightly slower than Hound
        effort_limit=2.5,
        stiffness=120.0,
        damping=8.0,
        friction=0.0,
    ),
    "throttle_joints": DCMotorCfg(
        joint_names_expr=[".*wheel_(back|front)_.*"],  # Matches all throttle joints (e.g. wheel_back_left, wheel_front_right, etc.)
        saturation_effort=1.0,
        effort_limit=0.25,   # Adjusted for the 3s VXL-3s motor/ESC
        velocity_limit=400.0,  # Reduced speed compared to a 4s system
        stiffness=0,
        damping=1100.0,
        friction=0.0,
    ),
}

# Initial state configuration for F1Tenth.
_ZERO_INIT_STATES = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "rotator_left": 0.0,
        "rotator_right": 0.0,
        "wheel_back_left": 0.0,
        "wheel_back_right": 0.0,
        "wheel_front_left": 0.0,
        "wheel_front_right": 0.0,
    },
)

# Overall configuration tying together the asset, physics, and initial state.
F1TENTH_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{WHEELEDLAB_ASSETS_DATA_DIR}/Robots/F1TENTH/f1tenth.usd",
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
    actuators=F1TENTH_4WD_ACTUATOR_CFG,
)

