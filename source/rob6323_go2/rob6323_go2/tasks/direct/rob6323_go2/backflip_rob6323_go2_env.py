# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Training Recipe for Stage-Wise Backflip (Based on arXiv:2409.15755):

PPO Hyperparameters (suggested):
- learning_rate: 3e-4
- num_steps_per_env: 24
- max_iterations: 5000
- minibatch_size: 4096
- num_learning_epochs: 5
- gamma: 0.99
- lam: 0.95
- clip_coef: 0.2

Curriculum Suggestions:
- Start with backflip_prob=0.2, gradually increase to 0.5
- Start with smaller rotation target (π radians) then increase to 2π
- Start with backflip_action_scale_mult=1.5, gradually increase to 2.0
- Reduce height targets initially (stand: 0.30, sit: 0.15) for easier learning

Domain Randomization (optional):
- Motor strength: ±20%
- Gravity: ±5%
- Friction: 0.7-1.3
- Joint position/velocity noise: ±0.05
- Base orientation noise: ±0.05 rad
"""

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

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp"
            ]
        }
        # Get specific body indices
        base_id_result, _ = self._contact_sensor.find_bodies("base")
        # Ensure base_id is a single integer
        if isinstance(base_id_result, (list, tuple)):
            self._base_id = int(base_id_result[0])
        else:
            self._base_id = int(base_id_result)
        
        # Find feet indices for contact detection
        # Go2 foot names: FL_foot, FR_foot, RL_foot, RR_foot
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self._feet_ids_sensor = []
        for foot_name in foot_names:
            foot_id, _ = self._contact_sensor.find_bodies(foot_name)
            if foot_id is not None:
                if isinstance(foot_id, (list, tuple)):
                    self._feet_ids_sensor.extend(foot_id)
                else:
                    self._feet_ids_sensor.append(int(foot_id))
        
        # Ensure we have 4 feet
        if len(self._feet_ids_sensor) < 4:
            # Fallback: try pattern matching
            all_feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
            if all_feet_ids is not None:
                if isinstance(all_feet_ids, (list, tuple)):
                    self._feet_ids_sensor = [int(id) for id in all_feet_ids[:4]]
                else:
                    self._feet_ids_sensor = [int(all_feet_ids)]
        
        # Separate front and rear feet (for foot contact cost in jump stage)
        # Front: FL_foot (0), FR_foot (1), Rear: RL_foot (2), RR_foot (3)
        self._front_feet_ids_sensor = self._feet_ids_sensor[:2] if len(self._feet_ids_sensor) >= 2 else []
        self._rear_feet_ids_sensor = self._feet_ids_sensor[2:4] if len(self._feet_ids_sensor) >= 4 else []
        
        # Find body/thigh indices for body contact cost
        # Go2 body names: FL_thigh, FR_thigh, RL_thigh, RR_thigh (and base)
        self._body_contact_ids = []
        body_names = [".*thigh"]  # Only use patterns that exist in Go2
        for body_name in body_names:
            try:
                body_id, _ = self._contact_sensor.find_bodies(body_name)
                if body_id is not None:
                    if isinstance(body_id, (list, tuple)):
                        self._body_contact_ids.extend([int(id) for id in body_id])
                    else:
                        self._body_contact_ids.append(int(body_id))
            except (ValueError, AttributeError):
                # Pattern doesn't match, skip
                pass
        
        # If no body contact ids found, use an empty list (will skip body contact cost)
        if len(self._body_contact_ids) == 0:
            self._body_contact_ids = []

        # ============ Backflip Stage Scheduler ============
        # Stage encoding: 0=Stand, 1=Sit, 2=Jump, 3=Air, 4=Land
        self._is_backflip_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 0-4
        self._stage_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._turn_progress = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # Integrated pitch rotation
        self._turn_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._land_stable_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # Counter for land success

        # Add backflip reward components to logging
        for stage_name in ["stand", "sit", "jump", "air", "land"]:
            for reward_type in ["height", "velocity", "balance", "style", "energy", "turn_progress"]:
                self._episode_sums[f"backflip_{stage_name}_{reward_type}"] = torch.zeros(
                    self.num_envs, dtype=torch.float, device=self.device
                )
            for cost_type in ["body_contact", "joint_position", "joint_velocity", "joint_torque", "foot_contact"]:
                self._episode_sums[f"backflip_{stage_name}_{cost_type}"] = torch.zeros(
                    self.num_envs, dtype=torch.float, device=self.device
                )

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
        
        # Stage-dependent action scaffolding (nominal joint offsets)
        q_nominal = self._get_stage_nominal_joint_positions()
        
        # Action scale multiplier for jump/air stages
        action_scale_mult = torch.where(
            (self._is_backflip_episode) & ((self._stage == 2) | (self._stage == 3)),
            self.cfg.backflip_action_scale_mult,
            torch.ones_like(self._stage, dtype=torch.float)
        )
        
        # Compute processed actions: default + nominal + scaled actions
        self._processed_actions = (
            self.robot.data.default_joint_pos
            + q_nominal
            + self.cfg.action_scale * action_scale_mult.unsqueeze(-1) * self._actions
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._processed_actions)

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
                )
                if tensor is not None
            ],
            dim=-1,
        )
        
        # Append backflip context if enabled
        if self.cfg.enable_backflip:
            # is_backflip_episode (0/1)
            is_backflip = self._is_backflip_episode.float().unsqueeze(-1)
            # stage normalized (stage/4)
            stage_norm = (self._stage.float() / 4.0).unsqueeze(-1)
            # stage_time normalized (divide by max expected stage duration)
            max_stage_duration = max(self.cfg.stage_durations_s.values())
            stage_time_norm = (self._stage_time / max_stage_duration).unsqueeze(-1)
            # turn_progress normalized (turn_progress / (2π), clipped to [-1, 1])
            turn_progress_norm = torch.clamp(self._turn_progress / (2 * math.pi), -1.0, 1.0).unsqueeze(-1)
            
            backflip_context = torch.cat([is_backflip, stage_norm, stage_time_norm, turn_progress_norm], dim=-1)
            obs = torch.cat([obs, backflip_context], dim=-1)
        
        observations = {"policy": obs}
        return observations

    def _update_stage_scheduler(self) -> None:
        """Update stage transitions based on Fig.3 from the paper."""
        # Update stage time
        self._stage_time += self.step_dt
        
        # Update turn progress (integrate pitch angular velocity)
        # For backflip, positive ωy (body frame y-axis angular velocity) = backward pitch rotation
        omega_y = self.robot.data.root_ang_vel_b[:, 1]  # Pitch angular velocity in body frame
        self._turn_progress += omega_y * self.step_dt
        
        # Check if turn is complete (360° = 2π radians)
        self._turn_complete = torch.abs(self._turn_progress) >= (2 * math.pi)
        
        # Get base height
        base_height = self.robot.data.root_pos_w[:, 2]
        
        # Get foot contact forces
        if len(self._feet_ids_sensor) > 0:
            foot_contact_forces = self._contact_sensor.data.net_forces_w_history[:, -1, self._feet_ids_sensor]
            foot_contact_norms = torch.norm(foot_contact_forces, dim=-1)
            foot_contact_thresh = 1.0  # N
            feet_in_contact = foot_contact_norms > foot_contact_thresh
            all_feet_airborne = ~torch.any(feet_in_contact, dim=-1)
            any_foot_contact = torch.any(feet_in_contact, dim=-1)
        else:
            # Fallback: assume no contact if feet not found
            all_feet_airborne = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            any_foot_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Stage transitions (only for backflip episodes)
        backflip_mask = self._is_backflip_episode
        
        # Stand → Sit: immediately on backflip episode start (handled in reset)
        # Sit → Jump: when base height drops below threshold
        sit_to_jump = (
            backflip_mask
            & (self._stage == 1)
            & (base_height < self.cfg.sit_to_jump_height_thresh)
        )
        self._stage[sit_to_jump] = 2
        self._stage_time[sit_to_jump] = 0.0
        
        # Jump → Air: when all feet detach from ground
        jump_to_air = (
            backflip_mask
            & (self._stage == 2)
            & all_feet_airborne
        )
        self._stage[jump_to_air] = 3
        self._stage_time[jump_to_air] = 0.0
        
        # Air → Land: when any foot makes contact
        air_to_land = (
            backflip_mask
            & (self._stage == 3)
            & any_foot_contact
        )
        self._stage[air_to_land] = 4
        self._stage_time[air_to_land] = 0.0

    def _get_stage_nominal_joint_positions(self) -> torch.Tensor:
        """Get stage-dependent nominal joint position offsets for action scaffolding."""
        num_joints = self.robot.data.default_joint_pos.shape[-1]
        q_nominal = torch.zeros(self.num_envs, num_joints, device=self.device)
        
        if not self.cfg.enable_backflip:
            return q_nominal
        
        # Define nominal poses for each stage (Go2 has 12 joints: 4 legs × 3 joints)
        # Joint order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
        # Approximate crouch offsets (in radians)
        crouch_offsets = torch.tensor([
            0.0, 0.5, -1.0,  # FL
            0.0, 0.5, -1.0,  # FR
            0.0, 0.5, -1.0,  # RL
            0.0, 0.5, -1.0,  # RR
        ], device=self.device) * 0.3  # Scale down
        
        # Tuck offsets for jump/air (more extreme)
        tuck_offsets = torch.tensor([
            0.0, 0.8, -1.5,  # FL
            0.0, 0.8, -1.5,  # FR
            0.0, 0.8, -1.5,  # RL
            0.0, 0.8, -1.5,  # RR
        ], device=self.device) * 0.3
        
        # Apply based on stage
        for env_idx in range(self.num_envs):
            if self._is_backflip_episode[env_idx]:
                stage = self._stage[env_idx].item()
                if stage == 1:  # Sit
                    q_nominal[env_idx] = crouch_offsets
                elif stage == 2:  # Jump
                    q_nominal[env_idx] = tuck_offsets
                elif stage == 3:  # Air
                    # Gradually untuck in late air (based on stage_time or turn_progress)
                    tuck_factor = torch.clamp(1.0 - self._stage_time[env_idx] / 1.0, 0.0, 1.0)
                    q_nominal[env_idx] = tuck_offsets * tuck_factor
                # Stand (0) and Land (4): no offset (return to default)
        
        return q_nominal

    def _get_rewards(self) -> torch.Tensor:
        # Update stage scheduler for backflip episodes
        if self.cfg.enable_backflip:
            self._update_stage_scheduler()
        
        # For non-backflip episodes, use original locomotion rewards
        rewards = {}
        
        # Standard locomotion rewards (for all episodes)
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        yaw_rate_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        
        rewards["track_lin_vel_xy_exp"] = lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt
        rewards["track_ang_vel_z_exp"] = yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt
        
        # For backflip episodes, compute stage-wise rewards (Table I)
        if self.cfg.enable_backflip:
            backflip_rewards = self._compute_stage_wise_rewards()
            rewards.update(backflip_rewards)
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            if key in self._episode_sums:
                self._episode_sums[key] += value
        
        return reward

    def _compute_stage_wise_rewards(self) -> dict:
        """Compute stage-wise rewards based on Table I from the paper."""
        rewards = {}
        backflip_mask = self._is_backflip_episode
        
        # Only compute for backflip episodes
        if not torch.any(backflip_mask):
            return rewards
        
        # Get robot state
        base_pos_w = self.robot.data.root_pos_w
        base_height = base_pos_w[:, 2]
        base_lin_vel_b = self.robot.data.root_lin_vel_b
        base_ang_vel_b = self.robot.data.root_ang_vel_b
        root_quat_w = self.robot.data.root_quat_w
        projected_gravity_b = self.robot.data.projected_gravity_b
        
        # Stage names for indexing
        stage_names = ["stand", "sit", "jump", "air", "land"]
        
        for stage_idx, stage_name in enumerate(stage_names):
            stage_mask = backflip_mask & (self._stage == stage_idx)
            if not torch.any(stage_mask):
                continue
            
            scales = self.cfg.reward_scales[stage_name]
            
            # (1) Base height reward
            if stage_name in ["stand", "land"]:
                height_target = self.cfg.stand_height_target
                height_reward = -torch.abs(base_height - height_target)
            elif stage_name == "sit":
                height_target = self.cfg.sit_height_target
                height_reward = -torch.abs(base_height - height_target)
            elif stage_name == "jump":
                # For jump stage, reward height gain (encourages upward motion)
                # Reward actual height (not capped) to encourage strong jump
                height_reward = base_height  # Direct reward for height
            else:  # air
                # 1_{pz<=0.5} * pz (per paper)
                height_reward = torch.where(
                    base_height <= self.cfg.jump_height_cap,
                    base_height,
                    torch.zeros_like(base_height)
                )
            
            rewards[f"backflip_{stage_name}_height"] = torch.where(
                stage_mask,
                height_reward * scales["height"] * self.step_dt,
                torch.zeros_like(height_reward)
            )
            
            # (2) Base velocity reward
            if stage_name in ["stand", "sit", "land"]:
                # -(vx^2 + vy^2 + wz^2)
                velocity_reward = -(
                    base_lin_vel_b[:, 0]**2
                    + base_lin_vel_b[:, 1]**2
                    + base_ang_vel_b[:, 2]**2
                )
            elif stage_name == "jump":
                # Paper Table I: -1_turn * ωy
                # 1_turn = 1 if turn complete, 0 otherwise
                # This means: reward = 0 before turn, -ωy after turn (penalize continued rotation)
                # However, to encourage backflip rotation, we also reward positive ωy before completion
                omega_y = base_ang_vel_b[:, 1]  # Pitch angular velocity (positive = backward rotation)
                # Encourage positive ωy before turn complete, penalize after
                velocity_reward = torch.where(
                    self._turn_complete,
                    -omega_y,  # After turn complete: penalize continued rotation (per paper)
                    omega_y    # Before turn complete: reward positive ωy (encourages backflip)
                )
            else:  # air
                # Paper Table I: -s * ωy where s changes sign
                # Strongly reward rotation before complete, penalize excessive rotation after
                omega_y = base_ang_vel_b[:, 1]
                # Before complete: strongly reward positive ωy, after: penalize excessive rotation
                velocity_reward = torch.where(
                    self._turn_complete,
                    -0.1 * omega_y,  # After turn complete: lightly penalize continued rotation
                    2.0 * omega_y    # Before turn complete: strongly reward positive ωy
                )
            
            rewards[f"backflip_{stage_name}_velocity"] = torch.where(
                stage_mask,
                velocity_reward * scales["velocity"] * self.step_dt,
                torch.zeros_like(velocity_reward)
            )
            
            # (3) Base balance reward
            if stage_name in ["stand", "sit", "land"]:
                # -angle(z_base, z_world) = -acos(projected_gravity_b[:, 2])
                # projected_gravity_b[2] is cos(angle) when upright
                # For upright: projected_gravity_b[2] ≈ -1 (gravity points down)
                # Angle from upright: angle ≈ acos(-projected_gravity_b[2])
                upright_score = -projected_gravity_b[:, 2]  # Should be close to 1 when upright
                balance_reward = -(1.0 - upright_score)  # Penalize deviation from 1.0
            else:  # jump, air
                # -|angle(y_base, z_world) - π/2|
                # Compute angle between base y-axis and world z-axis
                # Base y-axis in world frame: extract from quaternion
                base_y_axis_b = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float).unsqueeze(0).repeat(self.num_envs, 1)
                base_y_axis_w = math_utils.quat_rotate(root_quat_w, base_y_axis_b)
                world_z_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float).unsqueeze(0).repeat(self.num_envs, 1)
                # Dot product gives cos(angle)
                cos_angle = torch.sum(base_y_axis_w * world_z_axis, dim=-1)
                cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                angle = torch.acos(cos_angle)
                # Penalize deviation from π/2
                balance_reward = -torch.abs(angle - (math.pi / 2))
            
            rewards[f"backflip_{stage_name}_balance"] = torch.where(
                stage_mask,
                balance_reward * scales["balance"] * self.step_dt,
                torch.zeros_like(balance_reward)
            )
            
            # (4) Style reward: -Σ_j (q_j - q_default_j)^2
            joint_pos_error = self.robot.data.joint_pos - self.robot.data.default_joint_pos
            style_reward = -torch.sum(joint_pos_error**2, dim=-1)
            
            rewards[f"backflip_{stage_name}_style"] = torch.where(
                stage_mask,
                style_reward * scales["style"] * self.step_dt,
                torch.zeros_like(style_reward)
            )
            
            # (5) Energy reward: -Σ_j τ_j^2
            # Approximate with action magnitude and joint velocity penalty
            # Better approximation: use (q_target - q)^2 as proxy for torque
            joint_pos_target = self._processed_actions
            joint_pos_error_for_energy = joint_pos_target - self.robot.data.joint_pos
            energy_proxy = torch.sum(joint_pos_error_for_energy**2, dim=-1)
            # Also penalize joint velocities
            joint_vel_penalty = torch.sum(self.robot.data.joint_vel**2, dim=-1)
            energy_reward = -(energy_proxy + 0.1 * joint_vel_penalty)
            
            rewards[f"backflip_{stage_name}_energy"] = torch.where(
                stage_mask,
                energy_reward * scales["energy"] * self.step_dt,
                torch.zeros_like(energy_reward)
            )
            
            # (6) Turn progress reward (for jump and air stages)
            # Reward accumulating rotation progress, with bonus for completing full rotation
            if stage_name in ["jump", "air"] and "turn_progress" in scales and scales.get("turn_progress", 0) > 0:
                # Normalized turn progress (0 to 1 for full 360° rotation)
                turn_progress_norm = torch.clamp(torch.abs(self._turn_progress) / (2 * math.pi), 0.0, 1.0)
                # Reward: progress + bonus for completion
                turn_complete_bonus = self._turn_complete.float() * 2.0  # Extra bonus when complete
                turn_progress_reward = turn_progress_norm + turn_complete_bonus
                
                rewards[f"backflip_{stage_name}_turn_progress"] = torch.where(
                    stage_mask,
                    turn_progress_reward * scales["turn_progress"] * self.step_dt,
                    torch.zeros_like(turn_progress_reward)
                )
            elif stage_name in ["jump", "air"]:
                # Initialize to zero if not in scales
                rewards[f"backflip_{stage_name}_turn_progress"] = torch.zeros(
                    self.num_envs, device=self.device
                )
        
        # Compute costs as penalties (Table I)
        cost_penalties = self._compute_stage_wise_costs()
        rewards.update(cost_penalties)
        
        return rewards

    def _compute_stage_wise_costs(self) -> dict:
        """Compute cost penalties based on Table I from the paper."""
        penalties = {}
        backflip_mask = self._is_backflip_episode
        
        if not torch.any(backflip_mask):
            return penalties
        
        stage_names = ["stand", "sit", "jump", "air", "land"]
        net_contact_forces = self._contact_sensor.data.net_forces_w_history[:, -1, :]
        
        for stage_idx, stage_name in enumerate(stage_names):
            stage_mask = backflip_mask & (self._stage == stage_idx)
            if not torch.any(stage_mask):
                continue
            
            # Body contact cost: 1_{|I_body_C| > 0}
            if len(self._body_contact_ids) > 0:
                body_contact_force_norms = torch.norm(net_contact_forces[:, self._body_contact_ids], dim=-1)
                body_contact = torch.any(body_contact_force_norms > 1.0, dim=-1).float()
            else:
                body_contact = torch.zeros(self.num_envs, device=self.device)
            penalties[f"backflip_{stage_name}_body_contact"] = torch.where(
                stage_mask,
                -body_contact * self.cfg.cost_scales["body_contact"] * self.step_dt,
                torch.zeros_like(body_contact)
            )
            
            # Joint position cost: 1_{∃j: q_j > q_max_j || q_j < q_min_j}
            # Use approximate limits (Go2 typical ranges: ±1.5 rad)
            joint_pos = self.robot.data.joint_pos
            joint_limits_high = torch.ones_like(joint_pos) * 1.5
            joint_limits_low = -joint_limits_high
            joint_pos_violation = torch.any(
                (joint_pos > joint_limits_high) | (joint_pos < joint_limits_low),
                dim=-1
            ).float()
            penalties[f"backflip_{stage_name}_joint_position"] = torch.where(
                stage_mask,
                -joint_pos_violation * self.cfg.cost_scales["joint_position"] * self.step_dt,
                torch.zeros_like(joint_pos_violation)
            )
            
            # Joint velocity cost: 1_{∃j: |qdot_j| > qdot_max_j}
            joint_vel = torch.abs(self.robot.data.joint_vel)
            joint_vel_violation = torch.any(joint_vel > self.cfg.joint_velocity_limit, dim=-1).float()
            penalties[f"backflip_{stage_name}_joint_velocity"] = torch.where(
                stage_mask,
                -joint_vel_violation * self.cfg.cost_scales["joint_velocity"] * self.step_dt,
                torch.zeros_like(joint_vel_violation)
            )
            
            # Joint torque cost: 1_{∃j: |τ_j| > τ_max_j}
            # Approximate with action magnitude proxy
            action_magnitude = torch.abs(self._processed_actions - self.robot.data.default_joint_pos)
            torque_proxy = torch.sum(action_magnitude**2, dim=-1)
            torque_threshold = 2.0  # Approximate threshold
            joint_torque_violation = (torque_proxy > torque_threshold).float()
            penalties[f"backflip_{stage_name}_joint_torque"] = torch.where(
                stage_mask,
                -joint_torque_violation * self.cfg.cost_scales["joint_torque"] * self.step_dt,
                torch.zeros_like(joint_torque_violation)
            )
            
            # Foot contact cost: 1_{|I_foot,rear_C| == 0} (only in jump stage)
            if stage_name == "jump" and len(self._rear_feet_ids_sensor) > 0:
                rear_foot_forces = net_contact_forces[:, self._rear_feet_ids_sensor]
                rear_foot_force_norms = torch.norm(rear_foot_forces, dim=-1)
                # Penalize if rear feet detach before front feet
                # Check if rear feet are airborne while front feet might still be in contact
                front_foot_forces = net_contact_forces[:, self._front_feet_ids_sensor] if len(self._front_feet_ids_sensor) > 0 else torch.zeros(self.num_envs, 2, 3, device=self.device)
                front_foot_force_norms = torch.norm(front_foot_forces, dim=-1)
                rear_airborne = torch.all(rear_foot_force_norms < 1.0, dim=-1)
                front_in_contact = torch.any(front_foot_force_norms > 1.0, dim=-1)
                rear_detached_early = (rear_airborne & front_in_contact).float()
                penalties[f"backflip_{stage_name}_foot_contact"] = torch.where(
                    stage_mask,
                    -rear_detached_early * self.cfg.cost_scales["foot_contact"] * self.step_dt,
                    torch.zeros_like(rear_detached_early)
                )
            else:
                # Output threshold value (0.25) as cost (undefined in other stages)
                penalties[f"backflip_{stage_name}_foot_contact"] = torch.zeros(
                    self.num_envs, device=self.device
                )
        
        return penalties

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Separate termination logic for backflip vs normal episodes
        if self.cfg.enable_backflip:
            died = self._compute_backflip_terminations(time_out)
        else:
            # Original termination logic for non-backflip episodes
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            # net_forces_w_history shape: [num_envs, history_length, num_bodies, 3]
            # Get base contact forces: [num_envs, history_length, 3]
            base_contact_forces = net_contact_forces[:, :, self._base_id, :]
            # Compute norm: [num_envs, history_length]
            base_contact_norms = torch.norm(base_contact_forces, dim=-1)
            # Get max over history: [num_envs]
            max_base_contact = torch.max(base_contact_norms, dim=1)[0]
            cstr_termination_contacts = max_base_contact > 1.0
            cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0
            died = cstr_termination_contacts | cstr_upsidedown
        
        return died, time_out

    def _compute_backflip_terminations(self, time_out: torch.Tensor) -> torch.Tensor:
        """Compute terminations for backflip episodes (critical: don't terminate on upside-down mid-air)."""
        backflip_mask = self._is_backflip_episode
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # For non-backflip episodes, use original logic
        non_backflip_mask = ~backflip_mask
        if torch.any(non_backflip_mask):
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            # net_forces_w_history shape: [num_envs, history_length, num_bodies, 3]
            # Get base contact forces: [num_envs, history_length, 3]
            base_contact_forces = net_contact_forces[:, :, self._base_id, :]
            # Compute norm: [num_envs, history_length]
            base_contact_norms = torch.norm(base_contact_forces, dim=-1)
            # Get max over history: [num_envs]
            max_base_contact = torch.max(base_contact_norms, dim=1)[0]
            cstr_termination_contacts = max_base_contact > 1.0
            cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0
            died[non_backflip_mask] = cstr_termination_contacts[non_backflip_mask] | cstr_upsidedown[non_backflip_mask]
        
        if not torch.any(backflip_mask):
            return died
        
        # For backflip episodes:
        base_height = self.robot.data.root_pos_w[:, 2]  # Shape: [num_envs]
        base_ang_vel_norm = torch.norm(self.robot.data.root_ang_vel_b, dim=-1)  # Shape: [num_envs]
        net_contact_forces = self._contact_sensor.data.net_forces_w_history[:, -1, :]  # Shape: [num_envs, num_bodies, 3]
        
        # Get base contact force (base_id is now guaranteed to be an int)
        base_contact_force = net_contact_forces[:, self._base_id, :]  # Shape: [num_envs, 3]
        base_contact_force_norm = torch.norm(base_contact_force, dim=-1)  # Shape: [num_envs]
        
        # DO NOT terminate on upside-down during Jump/Air stages
        # Only terminate on true crash signals:
        
        # 1. Base contact hard impact (outside Land stage)
        base_hard_contact = (
            base_contact_force_norm > self.cfg.backflip_base_contact_force_thresh
        ) & (self._stage != 4)  # Not in land stage
        
        # 2. Height drops below min while still in Air (failed flip)
        air_crash = (
            (self._stage == 3)  # Air stage
            & (base_height < self.cfg.backflip_min_height_crash)
        )
        
        # 3. Extreme angular velocity safety clamp
        extreme_ang_vel = base_ang_vel_norm > self.cfg.backflip_max_ang_vel
        
        # 4. Early success termination in Land when recovered upright + stable
        self._update_land_success()
        land_success = (
            (self._stage == 4)
            & (self._land_stable_steps >= self.cfg.land_success_stable_steps)
        )
        
        # Combine termination conditions (all computed on full tensor, then filter)
        # Ensure all are 1D tensors of shape [num_envs]
        backflip_terminations = (
            base_hard_contact.bool()
            | air_crash.bool()
            | extreme_ang_vel.bool()
            | land_success.bool()  # Success is also a termination
        )
        
        # Only assign to backflip environments
        # backflip_terminations should be shape [num_envs], backflip_mask filters to subset
        died[backflip_mask] = backflip_terminations[backflip_mask]
        
        return died

    def _update_land_success(self) -> None:
        """Update land stability counter for success termination."""
        backflip_mask = self._is_backflip_episode
        land_mask = backflip_mask & (self._stage == 4)
        
        if not torch.any(land_mask):
            return
        
        # Check if upright and stable
        projected_gravity_b = self.robot.data.projected_gravity_b
        base_ang_vel_norm = torch.norm(self.robot.data.root_ang_vel_b, dim=-1)
        
        # Upright: projected_gravity_b[2] should be close to -1
        is_upright = -projected_gravity_b[:, 2] > (1.0 - self.cfg.land_success_upright_thresh)
        
        # Low angular velocity
        is_stable_ang_vel = base_ang_vel_norm < self.cfg.land_success_ang_vel_thresh
        
        # Feet in contact
        if len(self._feet_ids_sensor) > 0:
            foot_contact_forces = self._contact_sensor.data.net_forces_w_history[:, -1, self._feet_ids_sensor]
            foot_contact_norms = torch.norm(foot_contact_forces, dim=-1)
            feet_in_contact = torch.any(foot_contact_norms > 1.0, dim=-1)
        else:
            feet_in_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        is_stable = is_upright & is_stable_ang_vel & feet_in_contact
        
        # Increment counter if stable, reset if not
        self._land_stable_steps[land_mask & is_stable] += 1
        self._land_stable_steps[land_mask & ~is_stable] = 0

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
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Reset backflip stage scheduler
        if self.cfg.enable_backflip:
            # Convert env_ids to tensor if needed
            if isinstance(env_ids, (list, tuple)):
                env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
            elif isinstance(env_ids, torch.Tensor):
                env_ids_tensor = env_ids
            else:
                env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            
            # Sample backflip episodes based on probability
            backflip_sample = torch.rand(len(env_ids_tensor), device=self.device) < self.cfg.backflip_prob
            self._is_backflip_episode[env_ids_tensor] = backflip_sample
            
            # Initialize stages: Stand (0) for all, but transition to Sit (1) immediately for backflip
            self._stage[env_ids_tensor] = 0
            # Transition Stand → Sit immediately for backflip episodes (per Fig.3)
            backflip_env_ids = env_ids_tensor[backflip_sample]
            if len(backflip_env_ids) > 0:
                self._stage[backflip_env_ids] = 1
            
            # Reset stage-related variables
            self._stage_time[env_ids_tensor] = 0.0
            self._turn_progress[env_ids_tensor] = 0.0
            self._turn_complete[env_ids_tensor] = False
            self._land_stable_steps[env_ids_tensor] = 0
        
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