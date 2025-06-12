from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

from . import WHEELEDLAB_ASSETS_DATA_DIR
from .hound import HOUND_ACTUATOR_CFG, HOUND_SUS_ACTUATOR_CFG, HOUND_SUS_2WD_ACTUATOR_CFG

_ZERO_INIT_STATES = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        'back_left_wheel_throttle' : 0.0,
        'back_right_wheel_throttle' : 0.0,
        'front_left_wheel_steer' : 0.0,
        'front_right_wheel_steer' : 0.0,
        'front_left_wheel_throttle' : 0.0,
        'front_right_wheel_throttle' : 0.0,
    },
)

MUSHR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{WHEELEDLAB_ASSETS_DATA_DIR}/Robots/UWPRL/mushr_nano.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0, # m/s
            max_angular_velocity=100000.0, # deg/s
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
    actuators = HOUND_ACTUATOR_CFG
)

MUSHR_SUS_CFG = MUSHR_CFG.replace(
    spawn=MUSHR_CFG.spawn.replace(
        usd_path=f"{WHEELEDLAB_ASSETS_DATA_DIR}/Robots/UWRLL/mushr_nano_v2.usd",
    ),
    init_state=_ZERO_INIT_STATES.replace(
        joint_pos={
            **_ZERO_INIT_STATES.joint_pos,
            'front_left_wheel_suspension' : 0.0,
            'front_right_wheel_suspension' : 0.0,
            'back_left_wheel_suspension' : 0.0,
            'back_right_wheel_suspension' : 0.0,
        }
    ),
    actuators = HOUND_SUS_ACTUATOR_CFG,
)

MUSHR_SUS_2WD_CFG = MUSHR_SUS_CFG.replace(
    actuators = HOUND_SUS_2WD_ACTUATOR_CFG,
)
