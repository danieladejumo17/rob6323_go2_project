# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


import gymnasium as gym
import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils
 
from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        # Track time since last command update for periodic resampling
        self.command_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "rew_action_rate",     # <--- Added
                "raibert_heuristic",   # <--- Added
                "diagonal_phase_consistency",
                "duty_factor",
                "symmetry",
                "pacing_penalty",
                "hopping_penalty",
                "roll_pitch_penalty",
                "base_height_error",
                "vertical_velocity_penalty",
                "undesired_contact_penalty",
                "torque_magnitude_penalty",
                "action_rate_penalty",
                "action_jerk_penalty",
            ]
        }
        # variables needed for action rate penalization
        # Shape: (num_envs, action_dim, history_length)
        self.last_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), 3, 
            dtype=torch.float, device=self.device, requires_grad=False
        )
        # Store applied torques for regularization penalty
        self.applied_torques = torch.zeros(
            self.num_envs, self.robot.num_joints,
            dtype=torch.float, device=self.device, requires_grad=False
        )
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        # self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        # self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*thigh")
        
        # Variables needed for the raibert heuristic
        # Get specific body indices for feet (after robot is fully initialized in scene)
        self._feet_ids = []
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        for name in foot_names:
            id_list, _ = self.robot.find_bodies(name)
            self._feet_ids.append(id_list[0])
        
        # Find foot indices in contact sensor (for contact detection)
        self._feet_ids_sensor = []
        for name in foot_names:
            id_list, _ = self._contact_sensor.find_bodies(name)
            self._feet_ids_sensor.append(id_list[0])
        
        # Find non-foot body indices in contact sensor (for undesired contact detection)
        # These should not contact the ground (thigh, calf, hip links)
        self._undesired_contact_body_ids_sensor = []
        undesired_body_names = [".*thigh", ".*calf", ".*hip"]
        for pattern in undesired_body_names:
            id_list, _ = self._contact_sensor.find_bodies(pattern)
            self._undesired_contact_body_ids_sensor.extend(id_list)
        
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_indices = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Gait shaping state variables
        # Track contact time for duty factor calculation (rolling window)
        self.contact_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.total_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.duty_factor_window = 2.0  # Time window in seconds for duty factor calculation

        # PD control parameters
        self.Kp = torch.tensor([cfg.Kp] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.Kd = torch.tensor([cfg.Kd] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.motor_offsets = torch.zeros(self.num_envs, 12, device=self.device)
        self.torque_limits = cfg.torque_limits

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        # Compute desired joint positions from policy actions
        self.desired_joint_pos = (
            self.cfg.action_scale * self._actions 
            + self.robot.data.default_joint_pos
        )

    def _apply_action(self) -> None:
        # Compute PD torques
        torques = torch.clip(
            (
                self.Kp * (
                    self.desired_joint_pos 
                    - self.robot.data.joint_pos 
                )
                - self.Kd * self.robot.data.joint_vel
            ),
            -self.torque_limits,
            self.torque_limits,
        )

        # Store torques for regularization penalty
        self.applied_torques = torques.clone()

        # Apply torques to the robot
        self.robot.set_joint_effort_target(torques)

    @property
    def foot_positions_w(self) -> torch.Tensor:
        """Returns the feet positions in the world frame.
        Shape: (num_envs, num_feet, 3)
        """
        return self.robot.data.body_pos_w[:, self._feet_ids]

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.robot.data.root_lin_vel_b,
                    self.robot.data.root_ang_vel_b,
                    self.robot.data.projected_gravity_b,
                    self._commands,
                    self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                    self.robot.data.joint_vel,
                    self._actions,
                    self.clock_inputs,  # Add gait phase info
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Update gait state
        self._step_contact_targets()
        
        # Periodically resample commands for robust tracking
        self._update_commands()
        
        # linear velocity tracking (body frame)
        # Track v_x (forward) and v_y (lateral) with exponential reward
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        
        # yaw rate tracking (body frame)
        yaw_rate_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        
        # action rate penalization
        # First derivative (Current - Last)
        rew_action_rate = torch.sum(torch.square(self._actions - self.last_actions[:, :, 0]), dim=1) * (self.cfg.action_scale ** 2)
        # Second derivative (Current - 2*Last + 2ndLast)
        rew_action_rate += torch.sum(torch.square(self._actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1) * (self.cfg.action_scale ** 2)
        
        # Action regularization and smoothness rewards (compute BEFORE updating last_actions)
        penalty_torque_magnitude = self._penalty_torque_magnitude()
        penalty_action_rate = self._penalty_action_rate()
        penalty_action_jerk = self._penalty_action_jerk()
        
        # Update the prev action hist (roll buffer and insert new action)
        # IMPORTANT: Do this AFTER computing action_rate and action_jerk penalties
        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions[:]
        
        # Raibert heuristic reward
        rew_raibert_heuristic = self._reward_raibert_heuristic()
        
        # Gait shaping rewards
        rew_diagonal_phase = self._reward_diagonal_phase_consistency()
        rew_duty_factor = self._reward_duty_factor()
        rew_symmetry = self._reward_symmetry()
        penalty_pacing = self._penalty_pacing()
        penalty_hopping = self._penalty_hopping()
        
        # Base stability rewards
        penalty_roll_pitch = self._penalty_roll_pitch()
        penalty_base_height = self._penalty_base_height_error()
        penalty_vertical_vel = self._penalty_vertical_velocity()
        penalty_undesired_contact = self._penalty_undesired_contact()
        
        # Add to rewards dict
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale, # Removed step_dt
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale, # Removed step_dt
            "rew_action_rate": rew_action_rate * self.cfg.action_rate_reward_scale,
            # Note: This reward is negative (penalty) in the config
            "raibert_heuristic": rew_raibert_heuristic * self.cfg.raibert_heuristic_reward_scale,
            # Gait shaping rewards
            "diagonal_phase_consistency": rew_diagonal_phase * self.cfg.diagonal_phase_consistency_reward_scale,
            "duty_factor": rew_duty_factor * self.cfg.duty_factor_reward_scale,
            "symmetry": rew_symmetry * self.cfg.symmetry_reward_scale,
            "pacing_penalty": penalty_pacing * self.cfg.pacing_penalty_scale,
            "hopping_penalty": penalty_hopping * self.cfg.hopping_penalty_scale,
            # Base stability rewards
            "roll_pitch_penalty": penalty_roll_pitch * self.cfg.roll_pitch_penalty_scale,
            "base_height_error": penalty_base_height * self.cfg.base_height_error_penalty_scale,
            "vertical_velocity_penalty": penalty_vertical_vel * self.cfg.vertical_velocity_penalty_scale,
            "undesired_contact_penalty": penalty_undesired_contact * self.cfg.undesired_contact_penalty_scale,
            # Action regularization and smoothness
            "torque_magnitude_penalty": penalty_torque_magnitude * self.cfg.torque_magnitude_penalty_scale,
            "action_rate_penalty": penalty_action_rate * self.cfg.action_rate_penalty_scale,
            "action_jerk_penalty": penalty_action_jerk * self.cfg.action_jerk_penalty_scale,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_actual_contacts(self) -> torch.Tensor:
        """Extract binary contact state per foot from contact sensor.
        Returns: (num_envs, 4) binary tensor where 1 = contact, 0 = no contact
        """
        # Get contact forces for feet from sensor
        # net_forces_w_history shape: (num_envs, history_length, num_bodies, 3)
        # We need to index by body IDs in the sensor
        contact_forces = self._contact_sensor.data.net_forces_w_history[:, :, self._feet_ids_sensor, :]
        
        # Use most recent contact force (index 0 in history)
        # Compute magnitude of contact force
        contact_force_magnitude = torch.norm(contact_forces[:, 0, :, :], dim=-1)  # (num_envs, 4)
        
        # Threshold to determine contact (avoid noise)
        contact_threshold = 5.0  # Newtons
        actual_contacts = (contact_force_magnitude > contact_threshold).float()
        
        return actual_contacts

    def _reward_diagonal_phase_consistency(self) -> torch.Tensor:
        """Reward diagonal feet being in-phase and opposite pairs being out-of-phase.
        For trot: (FL + RR) in phase, (FR + RL) in phase, diagonal pairs 180° out of phase.
        """
        # Get phases from foot_indices (already in [0, 1])
        phases = self.foot_indices  # (num_envs, 4) [FL, FR, RL, RR]
        
        # Diagonal pair 1: FL and RR should be in phase
        phase_diff_FL_RR = torch.abs(phases[:, 0] - phases[:, 3])
        # Wrap around for circular distance
        phase_diff_FL_RR = torch.minimum(phase_diff_FL_RR, 1.0 - phase_diff_FL_RR)
        reward_diag1 = torch.exp(-phase_diff_FL_RR / 0.1)  # Reward when phases are close
        
        # Diagonal pair 2: FR and RL should be in phase
        phase_diff_FR_RL = torch.abs(phases[:, 1] - phases[:, 2])
        phase_diff_FR_RL = torch.minimum(phase_diff_FR_RL, 1.0 - phase_diff_FR_RL)
        reward_diag2 = torch.exp(-phase_diff_FR_RL / 0.1)
        
        # Opposite diagonal pairs should be ~0.5 out of phase
        phase_diff_FL_FR = torch.abs(phases[:, 0] - phases[:, 1])
        phase_diff_FL_FR = torch.minimum(phase_diff_FL_FR, 1.0 - phase_diff_FL_FR)
        # Reward when difference is close to 0.5
        target_diff = 0.5
        reward_opposite = torch.exp(-torch.square(phase_diff_FL_FR - target_diff) / 0.05)
        
        # Combine rewards
        reward = (reward_diag1 + reward_diag2 + reward_opposite) / 3.0
        
        return reward

    def _reward_duty_factor(self) -> torch.Tensor:
        """Reward duty factor in walking/trotting range (0.55-0.65).
        Penalize very low (< 0.4, hopping) and very high (> 0.8, shuffling).
        Uses a rolling window approach.
        """
        # Update contact time tracking
        actual_contacts = self._get_actual_contacts()  # (num_envs, 4)
        self.contact_time += actual_contacts * self.step_dt
        self.total_time += self.step_dt
        
        # Reset if window exceeded (rolling window)
        reset_mask = self.total_time > self.duty_factor_window
        self.contact_time[reset_mask] = actual_contacts[reset_mask] * self.step_dt
        self.total_time[reset_mask] = self.step_dt
        
        # Compute duty factor per foot (avoid division by zero)
        duty_factor = torch.zeros_like(self.contact_time)
        mask = self.total_time > 0.1  # Only compute after some time has passed
        duty_factor[mask] = self.contact_time[mask] / self.total_time[mask]
        
        # Target range: 0.55-0.65 (walking/trotting)
        target_min, target_max = 0.55, 0.65
        
        # Reward when in target range (smooth reward)
        in_range = (duty_factor >= target_min) & (duty_factor <= target_max)
        # Smooth reward: maximum at center, decreases towards edges
        target_center = (target_min + target_max) / 2.0
        distance_from_center = torch.abs(duty_factor - target_center)
        max_distance = (target_max - target_min) / 2.0
        reward_in_range = (1.0 - distance_from_center / max_distance) * in_range.float()
        
        # Penalize very low (< 0.4, hopping)
        too_low = duty_factor < 0.4
        penalty_low = torch.square((duty_factor - 0.4) / 0.4) * too_low.float()
        
        # Penalize very high (> 0.8, shuffling)
        too_high = duty_factor > 0.8
        penalty_high = torch.square((duty_factor - 0.8) / 0.2) * too_high.float()
        
        # Also penalize slightly outside target range (0.4-0.55 and 0.65-0.8)
        slightly_low = (duty_factor >= 0.4) & (duty_factor < target_min)
        slightly_high = (duty_factor > target_max) & (duty_factor <= 0.8)
        penalty_slight = torch.square((duty_factor - target_min) / 0.15) * slightly_low.float()
        penalty_slight += torch.square((duty_factor - target_max) / 0.15) * slightly_high.float()
        
        # Combine: reward in range, penalize outside
        reward = reward_in_range - penalty_low - penalty_high - penalty_slight * 0.5
        
        # Average across all feet
        reward = torch.mean(reward, dim=1)
        
        return reward

    def _reward_symmetry(self) -> torch.Tensor:
        """Reward left-right and front-rear symmetry in contact timing and phase.
        Includes direct leg pair symmetry checks for better FL-FR alignment.
        """
        actual_contacts = self._get_actual_contacts()  # (num_envs, 4) [FL, FR, RL, RR]
        phases = self.foot_indices  # (num_envs, 4) [FL, FR, RL, RR]
        
        # 1. Direct leg pair symmetry (FL-FR, RL-RR)
        # FL-FR should be 180° out of phase (0.5 phase difference) for trot
        phase_diff_FL_FR = torch.abs(phases[:, 0] - phases[:, 1])
        phase_diff_FL_FR = torch.minimum(phase_diff_FL_FR, 1.0 - phase_diff_FL_FR)
        # Reward when FL-FR are ~0.5 out of phase (symmetric)
        fl_fr_phase_symmetry = torch.exp(-torch.square(phase_diff_FL_FR - 0.5) / 0.05)
        
        # RL-RR should also be 180° out of phase
        phase_diff_RL_RR = torch.abs(phases[:, 2] - phases[:, 3])
        phase_diff_RL_RR = torch.minimum(phase_diff_RL_RR, 1.0 - phase_diff_RL_RR)
        rl_rr_phase_symmetry = torch.exp(-torch.square(phase_diff_RL_RR - 0.5) / 0.05)
        
        # Contact timing symmetry for FL-FR (should be opposite)
        fl_fr_contact_symmetry = 1.0 - torch.abs(actual_contacts[:, 0] - actual_contacts[:, 1])
        
        # 2. Left-right symmetry: (FL + RL) vs (FR + RR)
        left_contact = (actual_contacts[:, 0] + actual_contacts[:, 2]) / 2.0  # FL + RL
        right_contact = (actual_contacts[:, 1] + actual_contacts[:, 3]) / 2.0  # FR + RR
        lr_symmetry = 1.0 - torch.abs(left_contact - right_contact)
        
        # 3. Front-rear symmetry: (FL + FR) vs (RL + RR)
        front_contact = (actual_contacts[:, 0] + actual_contacts[:, 1]) / 2.0  # FL + FR
        rear_contact = (actual_contacts[:, 2] + actual_contacts[:, 3]) / 2.0  # RL + RR
        fr_symmetry = 1.0 - torch.abs(front_contact - rear_contact)
        
        # Combine all symmetry terms (weight direct leg pair symmetry more)
        reward = (
            0.4 * (fl_fr_phase_symmetry + rl_rr_phase_symmetry) / 2.0 +  # Direct leg pair phase symmetry
            0.2 * fl_fr_contact_symmetry +  # Direct FL-FR contact symmetry
            0.2 * lr_symmetry +  # Left-right aggregate symmetry
            0.2 * fr_symmetry   # Front-rear aggregate symmetry
        )
        
        return reward

    def _penalty_pacing(self) -> torch.Tensor:
        """Penalize pacing: (FL & RL) or (FR & RR) in contact simultaneously."""
        actual_contacts = self._get_actual_contacts()  # (num_envs, 4) [FL, FR, RL, RR]
        
        # Pacing pattern 1: FL and RL together (left side)
        pacing_left = actual_contacts[:, 0] * actual_contacts[:, 2]
        
        # Pacing pattern 2: FR and RR together (right side)
        pacing_right = actual_contacts[:, 1] * actual_contacts[:, 3]
        
        # Penalty is the sum of pacing occurrences
        penalty = pacing_left + pacing_right
        
        return penalty

    def _penalty_hopping(self) -> torch.Tensor:
        """Penalize hopping: all feet airborne or all feet in contact simultaneously."""
        actual_contacts = self._get_actual_contacts()  # (num_envs, 4) [FL, FR, RL, RR]
        
        # Sum of contacts per environment
        num_contacts = torch.sum(actual_contacts, dim=1)  # (num_envs,)
        
        # All feet airborne (0 contacts)
        all_airborne = (num_contacts == 0.0).float()
        
        # All feet in contact (4 contacts)
        all_contact = (num_contacts == 4.0).float()
        
        # Penalty is the sum of both undesirable states
        penalty = all_airborne + all_contact
        
        return penalty

    def _penalty_roll_pitch(self) -> torch.Tensor:
        """Penalize roll and pitch oscillations to keep base parallel to ground.
        Penalizes both instantaneous angles and angular velocities.
        Only penalizes significant deviations to allow necessary motion.
        """
        # Extract roll and pitch from base orientation
        # Use projected_gravity_b which gives us roll/pitch information
        # projected_gravity_b is gravity vector in body frame: [roll, pitch, yaw]
        # For small angles: roll ≈ -projected_gravity_b[:, 0], pitch ≈ projected_gravity_b[:, 1]
        projected_gravity = self.robot.data.projected_gravity_b  # (num_envs, 3)
        
        # Approximate roll and pitch angles (for small angles, this is accurate)
        # Roll: rotation around X axis (forward)
        # Pitch: rotation around Y axis (lateral)
        roll_approx = -projected_gravity[:, 0]  # Negative because gravity projects opposite
        pitch_approx = projected_gravity[:, 1]
        
        # Only penalize significant deviations (allow small angles for locomotion)
        # Use soft threshold: penalize more for larger angles
        roll_threshold = 0.1  # ~6 degrees
        pitch_threshold = 0.1  # ~6 degrees
        roll_excess = torch.clamp(torch.abs(roll_approx) - roll_threshold, min=0.0)
        pitch_excess = torch.clamp(torch.abs(pitch_approx) - pitch_threshold, min=0.0)
        penalty_angles = torch.square(roll_excess) + torch.square(pitch_excess)
        
        # Also penalize angular velocities in roll/pitch axes (only excessive ones)
        # root_ang_vel_b: (num_envs, 3) [roll_vel, pitch_vel, yaw_vel]
        ang_vel = self.robot.data.root_ang_vel_b
        roll_vel = ang_vel[:, 0]
        pitch_vel = ang_vel[:, 1]
        vel_threshold = 0.5  # rad/s
        roll_vel_excess = torch.clamp(torch.abs(roll_vel) - vel_threshold, min=0.0)
        pitch_vel_excess = torch.clamp(torch.abs(pitch_vel) - vel_threshold, min=0.0)
        penalty_velocities = torch.square(roll_vel_excess) + torch.square(pitch_vel_excess)
        
        # Combine angle and velocity penalties
        penalty = penalty_angles + 0.1 * penalty_velocities
        
        return penalty

    def _penalty_base_height_error(self) -> torch.Tensor:
        """Penalize deviation from target base height.
        Uses smooth penalty to avoid hard thresholds.
        Only penalizes significant deviations to allow natural locomotion.
        """
        base_height = self.robot.data.root_pos_w[:, 2]  # (num_envs,)
        target_height = self.cfg.base_height_target
        tolerance = self.cfg.base_height_tolerance
        
        # Compute error from target
        height_error = base_height - target_height
        
        # Use soft threshold: only penalize deviations beyond tolerance
        # This allows natural height variations during locomotion
        excess_error = torch.clamp(torch.abs(height_error) - tolerance, min=0.0)
        penalty = torch.square(excess_error / tolerance)
        
        # Additional small penalty for extreme cases (too low or too high)
        # But make it much smaller to not prevent locomotion
        too_low = base_height < (target_height - tolerance * 2)
        too_high = base_height > (target_height + tolerance * 2)
        penalty[too_low] += 0.5  # Reduced penalty for dragging
        penalty[too_high] += 0.3  # Reduced penalty for bouncing
        
        return penalty

    def _penalty_vertical_velocity(self) -> torch.Tensor:
        """Penalize excessive vertical linear velocity to reduce hopping and bobbing.
        Only penalizes excessive vertical motion to allow natural locomotion.
        """
        # Get vertical velocity in world frame
        # root_lin_vel_w: (num_envs, 3) [vx, vy, vz]
        vertical_vel = self.robot.data.root_lin_vel_w[:, 2]  # (num_envs,)
        
        # Only penalize excessive vertical velocity (allow small variations)
        # Threshold: 0.2 m/s (allows natural vertical motion during gait)
        vel_threshold = 0.2
        excess_vel = torch.clamp(torch.abs(vertical_vel) - vel_threshold, min=0.0)
        penalty = torch.square(excess_vel)
        
        return penalty

    def _penalty_undesired_contact(self) -> torch.Tensor:
        """Penalize ground contact on non-foot bodies (thigh, calf, hip).
        Penalty increases with contact force magnitude and frequency.
        """
        if len(self._undesired_contact_body_ids_sensor) == 0:
            # No undesired bodies found, return zero penalty
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # Get contact forces for undesired bodies from sensor
        # net_forces_w_history shape: (num_envs, history_length, num_bodies, 3)
        contact_forces = self._contact_sensor.data.net_forces_w_history[:, :, self._undesired_contact_body_ids_sensor, :]
        
        # Use most recent contact force (index 0 in history)
        # Compute magnitude of contact force for each undesired body
        contact_force_magnitude = torch.norm(contact_forces[:, 0, :, :], dim=-1)  # (num_envs, num_undesired_bodies)
        
        # Threshold to determine contact (avoid noise)
        contact_threshold = 5.0  # Newtons
        
        # Binary contact state
        is_contacting = (contact_force_magnitude > contact_threshold).float()
        
        # Penalty based on:
        # 1. Number of undesired bodies in contact
        # 2. Magnitude of contact forces
        num_contacts = torch.sum(is_contacting, dim=1)  # (num_envs,)
        max_force = torch.max(contact_force_magnitude, dim=1)[0]  # (num_envs,)
        
        # Combine: penalty increases with number of contacts and force magnitude
        penalty = num_contacts * (1.0 + max_force / 50.0)  # Scale force contribution
        
        return penalty

    def _penalty_torque_magnitude(self) -> torch.Tensor:
        """Penalize L2 norm of applied joint torques to discourage aggressive actuation.
        Normalized by number of joints and torque limits.
        """
        # Get applied torques (already clipped in _apply_action)
        torques = self.applied_torques  # (num_envs, num_joints)
        
        # Compute L2 norm squared per environment: mean(τ²)
        # This is equivalent to mean of squared torques, normalized by number of joints
        num_joints = torques.shape[1]
        torque_squared = torch.square(torques)  # (num_envs, num_joints)
        penalty = torch.mean(torque_squared, dim=1)  # (num_envs,)
        
        # Normalize by torque limits squared to make it scale-invariant
        # This ensures the penalty is comparable across different torque limits
        penalty = penalty / (self.torque_limits ** 2)
        
        return penalty

    def _penalty_action_rate(self) -> torch.Tensor:
        """Penalize first derivative of actions (a_t - a_{t-1}) for smoothness.
        Normalized by number of joints for scale-invariance.
        """
        # First derivative: current action - previous action
        action_diff = self._actions - self.last_actions[:, :, 0]  # (num_envs, action_dim)
        
        # Compute mean squared penalty per environment (normalized by number of joints)
        # This gives us the average squared change per joint
        num_joints = action_diff.shape[1]
        penalty = torch.mean(torch.square(action_diff), dim=1)  # (num_envs,)
        
        return penalty

    def _penalty_action_jerk(self) -> torch.Tensor:
        """Penalize second derivative of actions (a_t - 2a_{t-1} + a_{t-2}) to reduce jitter.
        Normalized by number of joints for scale-invariance.
        """
        # Second derivative: current - 2*previous + second_previous
        action_jerk = (
            self._actions 
            - 2 * self.last_actions[:, :, 0] 
            + self.last_actions[:, :, 1]
        )  # (num_envs, action_dim)
        
        # Compute mean squared penalty per environment (normalized by number of joints)
        # This gives us the average squared jerk per joint
        num_joints = action_jerk.shape[1]
        penalty = torch.mean(torch.square(action_jerk), dim=1)  # (num_envs,)
        
        return penalty

    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        """Sample new velocity commands for specified environments.
        Commands represent (v_x, v_y, yaw_rate) in body frame.
        """
        num_resets = len(env_ids)
        
        # Sample forward velocity (v_x)
        vx_min, vx_max = self.cfg.lin_vel_x_range
        self._commands[env_ids, 0] = sample_uniform(vx_min, vx_max, (num_resets,), device=self.device)
        
        # Sample lateral velocity (v_y)
        vy_min, vy_max = self.cfg.lin_vel_y_range
        self._commands[env_ids, 1] = sample_uniform(vy_min, vy_max, (num_resets,), device=self.device)
        
        # Sample yaw rate (yaw_dot)
        yaw_min, yaw_max = self.cfg.ang_vel_yaw_range
        self._commands[env_ids, 2] = sample_uniform(yaw_min, yaw_max, (num_resets,), device=self.device)

    def _update_commands(self) -> None:
        """Periodically resample commands during episodes for robust tracking.
        Resamples commands every command_resample_time seconds to test smooth transitions.
        """
        # Update command time
        self.command_time += self.step_dt
        
        # Find environments that need command resampling
        resample_mask = self.command_time >= self.cfg.command_resample_time
        
        if resample_mask.any():
            # Resample commands for those environments
            env_ids = torch.where(resample_mask)[0]
            self._sample_commands(env_ids)
            # Reset command time for resampled environments
            self.command_time[resample_mask] = 0.0

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        cstr_termination_contacts = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0
        # terminate if base is too low
        base_height = self.robot.data.root_pos_w[:, 2]
        cstr_base_height_min = base_height < self.cfg.base_height_min
        # apply all terminations
        died = cstr_termination_contacts | cstr_upsidedown | cstr_base_height_min
        return died, time_out

    def _step_contact_targets(self):
        """Defines contact plan and updates gait state."""
        frequencies = 3.0
        phases = 0.5
        offsets = 0.0
        bounds = 0.0
        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

        foot_indices = [
            self.gait_indices + phases + offsets + bounds,
            self.gait_indices + offsets,
            self.gait_indices + bounds,
            self.gait_indices + phases
        ]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                        0.5 / (1 - durations[swing_idxs]))

        self.clock_inputs[:, 0] = torch.sin(2 * math.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * math.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * math.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * math.pi * foot_indices[3])

        # von mises distribution
        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

        smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                        1 - smoothing_cdf_start(
                                    torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                        1 - smoothing_cdf_start(
                                    torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                        1 - smoothing_cdf_start(
                                    torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                        1 - smoothing_cdf_start(
                                    torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR

    def _reward_raibert_heuristic(self):
        """Compute Raibert heuristic reward based on foot placement error."""
        cur_footsteps_translated = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(
                math_utils.quat_conjugate(self.robot.data.root_quat_w),
                cur_footsteps_translated[:, i, :]
            )

        # nominal positions: [FL, FR, RL, RR] to match foot_names order
        desired_stance_width = 0.25
        # Order: FL (left=-), FR (right=+), RL (left=-), RR (right=+)
        desired_ys_nom = torch.tensor(
            [-desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2],
            device=self.device
        ).unsqueeze(0)

        desired_stance_length = 0.45
        # Order: FL (front=+), FR (front=+), RL (rear=-), RR (rear=-)
        desired_xs_nom = torch.tensor(
            [desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2],
            device=self.device
        ).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = torch.tensor([3.0], device=self.device)
        x_vel_des = self._commands[:, 0:1]
        yaw_vel_des = self._commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Reset last actions hist
        self.last_actions[env_ids] = 0.0
        # Reset applied torques
        self.applied_torques[env_ids] = 0.0
        # Reset raibert quantity
        self.gait_indices[env_ids] = 0
        # Reset gait shaping state
        self.contact_time[env_ids] = 0.0
        self.total_time[env_ids] = 0.0
        # Reset command time tracking
        self.command_time[env_ids] = 0.0
        # Sample new commands on reset
        self._sample_commands(env_ids)
        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat