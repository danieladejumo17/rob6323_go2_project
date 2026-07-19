# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg


@configclass
class BackflipRob6323Go2EnvCfg(DirectRLEnvCfg):
    """Single backflip with recovery to quadruped standing for the Unitree Go2.

    The robot spawns in its default quadruped stance and follows a timed phase
    schedule observed by the policy: stand (P0), crouch/launch/rotate backward
    (P1, flip window), then land and hold quadruped standing until timeout (P2).
    Standing rewards in P2 are gated by flip completion so skipping the flip
    never pays.
    """

    # env
    decimation = 4
    episode_length_s = 5.0
    # - spaces definition
    action_scale = 0.5  # deep crouch/tuck needs ~1 rad offsets; quadruped 0.25 is too tight
    action_space = 12
    observation_space = 47  # 45 proprio + episode-time + rotation-progress scalars
    state_space = 0

    # Phase schedule (seconds from episode start)
    flip_start_time = 0.5  # end of prepare/stand phase, start of flip window
    flip_end_time = 2.0    # end of flip window, start of recovery/stand phase

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
    # PD control gains (stiffer than locomotion's 20/0.5 for the explosive extension)
    Kp = 30.0
    Kd = 0.7
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # Flip parameters
    flip_target_rotation = 2.0 * math.pi  # full backward turn; cum_pitch target is -2*pi
    # Completion gate: 0 below 70% of a full turn, 1 at 95% (smooth ramp)
    flip_done_gate_lower = 0.7
    flip_done_gate_width = 0.25
    # At least this much backward pitch (rad) must happen fully airborne for the flip
    # to count — a ground tumble over the legs rotates fully but never ballistic
    aerial_rotation_threshold = math.pi
    flip_height_start = 0.30     # base height (m) where the jump-height reward starts paying
                                 # (= standing height; safe with the fully-airborne gate)
    flip_height_clip = 0.5       # cap on rewarded height gain (m)

    # Standing parameters
    stand_height_target = 0.30  # quadruped stance base height (m)
    stand_height_width = 0.01   # exp kernel width (m^2) for the height reward
    stand_pose_width = 1.0      # exp kernel width (rad^2) for the default-pose reward

    # Contact detection
    contact_force_threshold = 5.0  # Newtons (feet contact / airborne detection)
    base_contact_force_threshold = 1.0  # Newtons (crash termination, as in the quadruped task)

    # reward scales
    # -- flip window
    # Progress-based: pays only for NEW net backward rotation (running max), so the
    # total per episode is bounded by the scale — rate oscillation cannot farm it.
    flip_rotation_reward_scale = 20.0
    flip_height_reward_scale = 3.0
    # Per-radian shaping for backward rotation while fully airborne (capped at 2*pi)
    aerial_rotation_reward_scale = 5.0
    flip_completion_reward_scale = 1.0
    # -- standing (P0 ungated, P2 gated by flip completion)
    stand_upright_reward_scale = 1.0
    stand_height_reward_scale = 1.0
    stand_pose_reward_scale = 0.5
    stand_feet_contact_reward_scale = 0.5
    stand_still_penalty_scale = -0.1
    # -- always-on contact/regularization penalties
    # Kneeling/sitting on calves or thighs must never be free: it fakes both "standing"
    # and "airborne feet" (observed exploit in the first training run)
    undesired_contact_penalty_scale = -1.0
    joint_limit_penalty_scale = -10.0
    torque_magnitude_penalty_scale = -1e-4
    action_rate_penalty_scale = -1e-3
    action_jerk_penalty_scale = -5e-4
    dof_vel_penalty_scale = -5e-5
