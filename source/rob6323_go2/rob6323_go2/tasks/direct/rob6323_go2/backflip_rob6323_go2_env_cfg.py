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
    observation_space = 52  # 48 (original) + 4 (backflip context: is_backflip, stage_norm, stage_time_norm, turn_progress_norm)
    state_space = 0
    debug_vis = True

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
    # robot(s)
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

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

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5

    # ============ Backflip Configuration (Stage-Wise Reward Shaping) ============
    # Based on "Stage-Wise Reward Shaping for Acrobatic Robots" (arXiv:2409.15755)
    enable_backflip: bool = True
    backflip_prob: float = 0.5  # Probability of sampling a backflip episode per reset

    # Stage durations (minimum durations; transitions also event-based)
    stage_durations_s: dict = {
        "stand": 1.0,
        "sit": 0.5,
        "jump": 0.3,
        "air": 2.0,
        "land": 1.0,
    }

    # Height targets and thresholds
    stand_height_target: float = 0.35  # Paper uses pz≈0.35 in stand/land
    sit_height_target: float = 0.20  # Paper uses pz≈0.20 in sit
    sit_to_jump_height_thresh: float = 0.25  # Paper Fig.3: sit→jump when base height < 0.25
    jump_height_cap: float = 0.5  # Paper height reward gated by 1_{pz<=0.5}

    # Action scaling during jump/air stages
    backflip_action_scale_mult: float = 2.0  # Applied during jump/air only

    # Termination thresholds for backflip
    backflip_min_height_crash: float = 0.05  # Crash if height < this while in Air
    backflip_max_ang_vel: float = 50.0  # Safety clamp for angular velocity (rad/s)
    backflip_base_contact_force_thresh: float = 10.0  # Base contact force threshold (N)

    # Reward scales per stage (stage-wise reward shaping from Table I)
    # Format: {stage_name: {reward_component: scale}}
    # Stages: "stand", "sit", "jump", "air", "land"
    reward_scales: dict = {
        "stand": {
            "height": 5.0,
            "velocity": 0.5,
            "balance": 2.0,
            "style": 0.1,
            "energy": 0.05,
        },
        "sit": {
            "height": 5.0,
            "velocity": 0.5,
            "balance": 2.0,
            "style": 0.1,
            "energy": 0.05,
        },
        "jump": {
            "height": 15.0,  # Increased to encourage jump height
            "velocity": 5.0,  # Increased to encourage more angular velocity for backflip rotation
            "balance": 3.0,
            "style": 0.05,  # Reduced to allow more deviation during jump
            "energy": 0.05,
            "turn_progress": 10.0,  # New: reward for completing rotation
        },
        "air": {
            "height": 10.0,
            "velocity": 5.0,  # Increased to encourage continued rotation
            "balance": 3.0,
            "style": 0.05,  # Reduced to allow tuck during flip
            "energy": 0.05,
            "turn_progress": 10.0,  # New: reward for completing rotation
        },
        "land": {
            "height": 5.0,
            "velocity": 0.5,
            "balance": 2.0,
            "style": 0.1,
            "energy": 0.05,
        },
    }

    # Cost penalty scales (applied as negative rewards)
    cost_scales: dict = {
        "body_contact": 10.0,
        "joint_position": 1.0,
        "joint_velocity": 0.1,
        "joint_torque": 0.01,
        "foot_contact": 5.0,  # Only active in jump stage
    }

    # Success termination criteria (in land stage)
    land_success_upright_thresh: float = 0.2  # Max angle from upright (rad)
    land_success_ang_vel_thresh: float = 1.0  # Max angular velocity (rad/s)
    land_success_stable_steps: int = 50  # Steps to maintain stability before success

    # Joint limits for cost computation
    joint_velocity_limit: float = 30.0  # rad/s
    joint_torque_limit: float = 40.0  # Nm (approximate for Go2)