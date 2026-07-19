# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG


@configclass
class BipedRob6323Go2EnvCfg(DirectRLEnvCfg):
    """Bipedal (hind-leg) locomotion task for the Unitree Go2.

    The robot spawns in its normal quadruped pose, must rear up onto its hind legs,
    then stand/walk bipedally tracking (vx, yaw rate) commands. Front legs remain
    actuated but are rewarded to hold a tucked pose and penalized for ground contact.
    """

    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition
    action_scale = 0.5  # larger than quadruped (0.25): biped hind-thigh pose needs ~1.5 rad offsets
    action_space = 12
    observation_space = 48 + 2  # 48 base + 2 hind-feet clock inputs
    state_space = 0
    debug_vis = True

    # Command following parameters
    command_resample_time = 5.0  # Commands only matter once upright; fewer switches per episode
    lin_vel_x_range = (-0.3, 0.5)  # Forward velocity range (m/s) — biped Go2 is slow
    ang_vel_yaw_range = (-0.5, 0.5)  # Yaw rate range (rad/s); lateral vy is fixed to 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    # PD control gains
    Kp = 20.0  # Proportional gain
    Kd = 0.5   # Derivative gain
    torque_limits = 23.5  # Match DCMotor effort limit so penalties/logging see real torques

    # robot(s)
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # Disable implicit PD controller by setting stiffness and damping to 0 for all actuators
    for actuator_name, actuator in robot_cfg.actuators.items():
        if hasattr(actuator, "stiffness"):
            actuator.stiffness = 0.0
        if hasattr(actuator, "damping"):
            actuator.damping = 0.0

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=4.0, replicate_physics=True)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # Biped stance parameters
    biped_height_target = 0.52  # Target base height when standing on hind legs (m)
    quad_start_height = 0.30  # Reference base height in quadruped stance (m), start of height ramp
    # Uprightness gate: up_proj = -projected_gravity_b[:, 0], 1.0 when fully upright.
    # Gate is 0 below gate_lower (~53 deg pitch) and 1 above gate_upper (~72 deg).
    gate_lower = 0.80
    gate_upper = 0.95
    # Height gate on tracking rewards: crouch-walking (vertical trunk, base at quadruped
    # height) must not pay, or it becomes a local optimum. Tracking ramps in over this range.
    track_height_lower = 0.35
    track_height_upper = 0.45
    # Softer pitch gate on the height ramp: without it, standing tall on all fours
    # (tip-toe quadruped) collects the height reward with zero pitch. Opens from ~30 deg
    # pitch so the pitched-up crouch keeps a strong height gradient.
    height_ramp_gate_lower = 0.5
    height_ramp_gate_upper = 0.9
    # Width of the exp kernel rewarding base height at the biped target (wider -> gradient
    # reaches further down the crouch)
    height_fine_width = 0.02

    # Front-leg tuck pose targets (fold forearms against the belly in the nose-up frame)
    front_tuck_hip = 0.0
    front_tuck_thigh = 1.3
    front_tuck_calf = -2.4

    # Hind-leg gait clock
    gait_frequency = 1.5  # Hz, slower than the 3 Hz quadruped trot
    tracking_contact_force_norm = 200.0  # Force normalization constant for contact tracking

    # Contact detection
    contact_force_threshold = 5.0  # Newtons

    # termination criteria (projected gravity thresholds for irrecoverable falls)
    fall_backward_threshold = 0.7  # projected_gravity_b z: pitched past ~135 deg (on its back)
    fall_forward_threshold = 0.7   # projected_gravity_b x: nose down past ~-45 deg
    fall_sideways_threshold = 0.8  # |projected_gravity_b y|: rolled onto a side

    # reward scales
    # -- ungated shaping (drives rear-up from frame 1)
    upright_reward_scale = 2.0
    height_ramp_reward_scale = 3.0
    roll_penalty_scale = -2.0
    joint_limit_penalty_scale = -10.0
    # Ungated: sitting back on the hind calves must never be free
    hind_calf_contact_penalty_scale = -0.2
    # -- gated (pay only once upright)
    height_fine_reward_scale = 1.0
    lin_vel_reward_scale = 1.5
    yaw_rate_reward_scale = 0.75
    front_tuck_penalty_scale = -0.25
    front_contact_penalty_scale = -0.5
    tracking_contacts_shaped_force_reward_scale = 1.0
    undesired_contact_penalty_scale = -1.0
    vertical_velocity_penalty_scale = -0.5
    # -- regularization
    torque_magnitude_penalty_scale = -1e-4
    action_rate_penalty_scale = -2e-3
    action_jerk_penalty_scale = -1e-3
    dof_vel_penalty_scale = -1e-4
