# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition
    action_scale = 0.25
    action_space = 12
    observation_space = 48 + 4  # Added 4 for clock inputs
    state_space = 0
    debug_vis = True
    
    # Command following parameters
    command_resample_time = 3.0  # Resample commands every 3 seconds (2-4s range)
    lin_vel_x_range = (-1.0, 1.0)  # Forward velocity range (m/s)
    lin_vel_y_range = (-0.5, 0.5)  # Lateral velocity range (m/s)
    ang_vel_yaw_range = (-1.0, 1.0)  # Yaw rate range (rad/s)

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
    torque_limits = 100.0  # Max torque

    # robot(s)
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # Disable implicit PD controller by setting stiffness and damping to 0 for all actuators
    # CRITICAL: Set to 0 to disable implicit P-gain and D-gain
    # Modify all existing actuators to disable implicit PD control
    for actuator_name, actuator in robot_cfg.actuators.items():
        if hasattr(actuator, "stiffness"):
            actuator.stiffness = 0.0
        if hasattr(actuator, "damping"):
            actuator.damping = 0.0

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
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

    # termination criteria
    base_height_min = 0.20  # Terminate if base is lower than 20cm

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    action_rate_reward_scale = -0.1  # Legacy: combines first and second derivatives
    
    # Action regularization and smoothness reward scales (very small as per requirements)
    torque_magnitude_penalty_scale = -1e-4  # -0.0001 or smaller as required
    action_rate_penalty_scale = -2e-3  # First derivative: a_t - a_{t-1}
    action_jerk_penalty_scale = -1e-3  # Second derivative: a_t - 2a_{t-1} + a_{t-2}
    raibert_heuristic_reward_scale = -0.1  # Reduced to prioritize tracking over Raibert penalty
    
    # Gait shaping reward scales
    diagonal_phase_consistency_reward_scale = 0.5
    duty_factor_reward_scale = 0.3
    symmetry_reward_scale = 0.5
    pacing_penalty_scale = -1.0
    hopping_penalty_scale = -1.0
    feet_clearance_reward_scale = -10.0
    tracking_contacts_shaped_force_reward_scale = 4.0

    # Additional reward scales (Part 5 tutorial)
    orient_reward_scale = -5.0
    lin_vel_z_reward_scale = -0.02
    dof_vel_reward_scale = -0.0001
    ang_vel_xy_reward_scale = -0.001
    
    # Base stability reward scales (kept small to not interfere with locomotion)
    roll_pitch_penalty_scale = -0.05
    base_height_error_penalty_scale = -0.2
    vertical_velocity_penalty_scale = -0.01
    undesired_contact_penalty_scale = -0.5
    
    # Base stability parameters
    base_height_target = 0.30  # Target base height in meters (0.28-0.35 for Go2)
    base_height_tolerance = 0.05  # Tolerance around target height

    # Foot interaction parameters
    foot_clearance_target = 0.07  # Desired swing foot height (m)
    foot_stance_height_target = 0.02  # Desired stance foot clearance (m)
    tracking_contact_force_norm = 200.0  # Force normalization constant for contact tracking
