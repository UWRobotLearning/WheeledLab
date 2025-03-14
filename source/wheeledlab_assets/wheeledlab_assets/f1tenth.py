import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
from isaaclab.assets import ArticulationCfg

# Hardcoded path to the F1Tenth USD asset (we assume the file is named f1tenth.usd)
USD_PATH = "source/wheeledlab_assets/data/Robots/F1Tenth/f1tenth.usd"

# F1Tenth 4WD actuator configuration.
# For 4WD, all throttle joints (front and back) are active.
F1TENTH_4WD_ACTUATOR_CFG = {
    "steering_joints": ImplicitActuatorCfg(
        joint_names_expr=["front_left_wheel_steer", "front_right_wheel_steer"],
        velocity_limit=10.0,    # F1Tenth steering is slightly slower than Hound
        effort_limit=2.5,
        stiffness=120.0,
        damping=8.0,
        friction=0.0,
    ),
    "throttle_joints": DCMotorCfg(
        joint_names_expr=[".*throttle"],  # Matches all throttle joints (all four wheels)
        saturation_effort=1.0,
        effort_limit=0.25,   # Adjusted for the 3s VXL-3s motor/ESC
        velocity_limit=400.0,  # Reduced speed compared to a 4s system
        stiffness=0,
        damping=1100.0,
        friction=0.0,
    ),
    "suspension": ImplicitActuatorCfg(
        joint_names_expr=[".*_suspension"],
        effort_limit=None,
        velocity_limit=None,
        stiffness=5e7,       # Lower stiffness for a more springy suspension
        damping=0.0,         # Lower shock oil weight implies minimal damping
        friction=0.6,
    ),
}

# Initial state configuration for F1Tenth.
_ZERO_INIT_STATES = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "front_left_wheel_steer": 0.0,
        "front_right_wheel_steer": 0.0,
        "front_left_wheel_throttle": 0.0,
        "front_right_wheel_throttle": 0.0,
        # If there are additional throttle joints for the back wheels, they will also match ".*throttle"
    },
)

# Overall configuration tying together the asset, physics, and initial state.
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
    actuators=F1TENTH_4WD_ACTUATOR_CFG,
)

class F1Tenth:
    """
    F1Tenth robot integration for WheeledLab.

    """
    def __init__(self, prim_path="/World/F1Tenth", **kwargs):
        self.prim_path = prim_path
        # Use the 4WD actuator configuration
        self.cfg = F1TENTH_CFG.replace(**kwargs)
        self.asset_path = USD_PATH

    def spawn(self):
        """Spawn the F1Tenth model into the simulation at the specified prim path."""
        return sim_utils.spawn_articulation(self.cfg, prim_path=self.prim_path)
