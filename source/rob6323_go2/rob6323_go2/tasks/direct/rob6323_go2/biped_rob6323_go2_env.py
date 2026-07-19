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
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from .biped_rob6323_go2_env_cfg import BipedRob6323Go2EnvCfg


class BipedRob6323Go2Env(DirectRLEnv):
    """Bipedal (hind-leg) locomotion for the Unitree Go2.

    Episodes start in the normal quadruped pose. Ungated uprightness/height shaping
    drives the rear-up transition; velocity tracking, front-leg tuck/contact and
    hind-leg gait rewards are gated by an uprightness measure so they only pay once
    the robot is standing on its hind legs.

    Frame convention when upright (nose-up ~90 deg pitch about body -y):
    world-up aligns with body +x, so projected_gravity_b goes from (0, 0, -1) in
    quadruped stance to (-1, 0, 0) fully upright.
    """

    cfg: BipedRob6323Go2EnvCfg

    def __init__(self, cfg: BipedRob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)

        # Commands: (vx, vy, yaw rate) — vy is always 0 for the biped task, kept for layout compatibility
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        # Track time since last command update for periodic resampling
        self.command_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "upright",
                "height_ramp",
                "height_fine",
                "track_lin_vel",
                "track_yaw_rate",
                "roll_penalty",
                "joint_limit_penalty",
                "front_tuck_penalty",
                "front_contact_penalty",
                "tracking_contacts_shaped_force",
                "undesired_contact_penalty",
                "hind_calf_contact_penalty",
                "vertical_velocity_penalty",
                "torque_magnitude_penalty",
                "action_rate_penalty",
                "action_jerk_penalty",
                "dof_vel_penalty",
            ]
        }
        # Action history for rate/jerk penalties. Shape: (num_envs, action_dim, history_length)
        self.last_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), 3,
            dtype=torch.float, device=self.device, requires_grad=False
        )
        # Store applied torques for regularization penalty
        self.applied_torques = torch.zeros(
            self.num_envs, self.robot.num_joints,
            dtype=torch.float, device=self.device, requires_grad=False
        )

        # Front-leg joint indices and tuck-pose targets (matching order)
        front_joint_names = [
            "FL_hip_joint", "FR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint",
        ]
        self._front_joint_ids = []
        front_tuck_targets = []
        tuck_by_type = {
            "hip": self.cfg.front_tuck_hip,
            "thigh": self.cfg.front_tuck_thigh,
            "calf": self.cfg.front_tuck_calf,
        }
        for name in front_joint_names:
            id_list, _ = self.robot.find_joints(name)
            self._front_joint_ids.append(id_list[0])
            front_tuck_targets.append(tuck_by_type[name.split("_")[1]])
        self._front_tuck_targets = torch.tensor(
            front_tuck_targets, dtype=torch.float, device=self.device
        ).unsqueeze(0)

        # Contact sensor body indices
        self._hind_feet_ids_sensor = []
        for name in ["RL_foot", "RR_foot"]:
            id_list, _ = self._contact_sensor.find_bodies(name)
            self._hind_feet_ids_sensor.append(id_list[0])
        self._front_feet_ids_sensor = []
        for name in ["FL_foot", "FR_foot"]:
            id_list, _ = self._contact_sensor.find_bodies(name)
            self._front_feet_ids_sensor.append(id_list[0])
        # Undesired contact bodies once upright: hind thighs/calves and the base
        self._undesired_contact_body_ids_sensor = []
        for pattern in ["R[LR]_thigh", "R[LR]_calf", "base"]:
            id_list, _ = self._contact_sensor.find_bodies(pattern)
            self._undesired_contact_body_ids_sensor.extend(id_list)
        # Hind calves separately for the ungated sitting penalty
        self._hind_calf_ids_sensor, _ = self._contact_sensor.find_bodies("R[LR]_calf")

        # Hind-leg gait clock state (2 feet: RL, RR)
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_indices = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)

        # Constant unit vectors for heading-frame math
        self._unit_x = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        self._unit_z = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)

        # PD control parameters
        self.Kp = torch.tensor([cfg.Kp] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.Kd = torch.tensor([cfg.Kd] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
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
        torques = (
            self.Kp * (
                self.desired_joint_pos
                - self.robot.data.joint_pos
            )
            - self.Kd * self.robot.data.joint_vel
        )

        # Clip torques to limits
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)

        # Store torques for regularization penalty
        self.applied_torques = torques.clone()

        # Apply torques to the robot
        self.robot.set_joint_effort_target(torques)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                self._commands,
                self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                self.robot.data.joint_vel,
                self._actions,
                self.clock_inputs,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    # -- posture / frame helpers ---------------------------------------------------

    def _get_up_proj(self) -> torch.Tensor:
        """Uprightness measure: 1.0 when the base is fully nose-up (body +x = world up)."""
        return (-self.robot.data.projected_gravity_b[:, 0]).clamp(min=0.0)

    def _get_gate(self, up_proj: torch.Tensor) -> torch.Tensor:
        """Smooth gate in [0, 1]: 0 below ~53 deg pitch, 1 above ~72 deg."""
        return ((up_proj - self.cfg.gate_lower) / (self.cfg.gate_upper - self.cfg.gate_lower)).clamp(0.0, 1.0)

    def _get_heading_velocities(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward/lateral planar velocity in a heading frame valid for any pitch in [0, 135] deg.

        Body x alone degenerates (zero horizontal projection) at 90 deg pitch and body -z
        at 0 deg; their difference keeps a horizontal norm >= 1 across the whole alive set.
        """
        q = self.robot.data.root_quat_w
        fwd_w = math_utils.quat_apply(q, self._unit_x)
        up_w = math_utils.quat_apply(q, self._unit_z)
        heading = (fwd_w - up_w)[:, :2]
        heading = heading / heading.norm(dim=1, keepdim=True).clamp(min=1e-6)
        v_xy = self.robot.data.root_lin_vel_w[:, :2]
        v_fwd = (v_xy * heading).sum(dim=1)
        v_lat = heading[:, 0] * v_xy[:, 1] - heading[:, 1] * v_xy[:, 0]
        return v_fwd, v_lat

    def _get_base_height(self) -> torch.Tensor:
        return self.robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]

    # -- rewards -------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        # Update gait state
        self._step_contact_targets()

        # Periodically resample commands for robust tracking
        self._update_commands()

        up_proj = self._get_up_proj()
        gate = self._get_gate(up_proj)
        base_height = self._get_base_height()
        v_fwd, v_lat = self._get_heading_velocities()

        # -- ungated shaping (drives the rear-up from frame 1)
        rew_upright = 0.5 * (1.0 + up_proj)
        # Linear ramp toward the biped stance height (exp kernel would be flat at start height),
        # gated by a soft pitch gate so a tip-toe quadruped stance earns nothing from height.
        height_pitch_gate = (
            (up_proj - self.cfg.height_ramp_gate_lower)
            / (self.cfg.height_ramp_gate_upper - self.cfg.height_ramp_gate_lower)
        ).clamp(0.0, 1.0)
        rew_height_ramp = height_pitch_gate * (
            (base_height - self.cfg.quad_start_height)
            / (self.cfg.biped_height_target - self.cfg.quad_start_height)
        ).clamp(0.0, 1.0)
        pen_roll = torch.square(self.robot.data.projected_gravity_b[:, 1])
        pen_joint_limits = self._penalty_joint_limits()
        pen_hind_calf = self._penalty_hind_calf_contact()

        # -- gated terms (pay only once upright)
        rew_height_fine = gate * torch.exp(
            -torch.square(base_height - self.cfg.biped_height_target) / self.cfg.height_fine_width
        )
        # Tracking additionally requires standing tall: crouch-walking with a vertical
        # trunk must not collect tracking reward (observed local optimum).
        height_gate = (
            (base_height - self.cfg.track_height_lower)
            / (self.cfg.track_height_upper - self.cfg.track_height_lower)
        ).clamp(0.0, 1.0)
        track_gate = gate * height_gate
        lin_err = torch.square(self._commands[:, 0] - v_fwd) + torch.square(v_lat)
        rew_track_lin = track_gate * torch.exp(-lin_err / 0.25)
        yaw_err = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_w[:, 2])
        rew_track_yaw = track_gate * torch.exp(-yaw_err / 0.25)
        pen_front_tuck = gate * self._penalty_front_tuck()
        pen_front_contact = gate * self._penalty_front_contact()
        rew_tracking_contacts = track_gate * self._reward_tracking_contacts_shaped_force()
        pen_undesired_contact = gate * self._penalty_undesired_contact()
        vertical_excess = torch.clamp(torch.abs(self.robot.data.root_lin_vel_w[:, 2]) - 0.3, min=0.0)
        pen_vertical_vel = gate * torch.square(vertical_excess)

        # -- regularization (compute BEFORE updating last_actions)
        penalty_torque_magnitude = self._penalty_torque_magnitude()
        penalty_action_rate = self._penalty_action_rate()
        penalty_action_jerk = self._penalty_action_jerk()
        pen_dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)

        # Update the prev action hist (roll buffer and insert new action)
        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions[:]

        rewards = {
            "upright": rew_upright * self.cfg.upright_reward_scale,
            "height_ramp": rew_height_ramp * self.cfg.height_ramp_reward_scale,
            "height_fine": rew_height_fine * self.cfg.height_fine_reward_scale,
            "track_lin_vel": rew_track_lin * self.cfg.lin_vel_reward_scale,
            "track_yaw_rate": rew_track_yaw * self.cfg.yaw_rate_reward_scale,
            "roll_penalty": pen_roll * self.cfg.roll_penalty_scale,
            "joint_limit_penalty": pen_joint_limits * self.cfg.joint_limit_penalty_scale,
            "front_tuck_penalty": pen_front_tuck * self.cfg.front_tuck_penalty_scale,
            "front_contact_penalty": pen_front_contact * self.cfg.front_contact_penalty_scale,
            "tracking_contacts_shaped_force": rew_tracking_contacts * self.cfg.tracking_contacts_shaped_force_reward_scale,
            "undesired_contact_penalty": pen_undesired_contact * self.cfg.undesired_contact_penalty_scale,
            "hind_calf_contact_penalty": pen_hind_calf * self.cfg.hind_calf_contact_penalty_scale,
            "vertical_velocity_penalty": pen_vertical_vel * self.cfg.vertical_velocity_penalty_scale,
            "torque_magnitude_penalty": penalty_torque_magnitude * self.cfg.torque_magnitude_penalty_scale,
            "action_rate_penalty": penalty_action_rate * self.cfg.action_rate_penalty_scale,
            "action_jerk_penalty": penalty_action_jerk * self.cfg.action_jerk_penalty_scale,
            "dof_vel_penalty": pen_dof_vel * self.cfg.dof_vel_penalty_scale,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _penalty_joint_limits(self) -> torch.Tensor:
        """Penalize joint positions beyond the soft limits (hind hips/thighs run near limits when upright)."""
        q = self.robot.data.joint_pos
        lim = self.robot.data.soft_joint_pos_limits  # (num_envs, 12, 2)
        out_of_limits = (lim[..., 0] - q).clamp(min=0.0) + (q - lim[..., 1]).clamp(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _penalty_front_tuck(self) -> torch.Tensor:
        """Penalize front-leg joints deviating from the tucked pose."""
        front_q_err = self.robot.data.joint_pos[:, self._front_joint_ids] - self._front_tuck_targets
        return torch.sum(torch.square(front_q_err), dim=1)

    def _penalty_front_contact(self) -> torch.Tensor:
        """Penalize front-foot ground contact (only meaningful once upright — gated by caller)."""
        front_forces = self._contact_sensor.data.net_forces_w_history[:, 0, self._front_feet_ids_sensor, :]
        force_magnitude = torch.norm(front_forces, dim=-1)
        return (force_magnitude > self.cfg.contact_force_threshold).float().sum(dim=1)

    def _reward_tracking_contacts_shaped_force(self) -> torch.Tensor:
        """Reward hind-foot contact forces that match the desired 2-foot gait timing."""
        hind_forces = self._contact_sensor.data.net_forces_w_history[:, 0, self._hind_feet_ids_sensor, :]
        force_magnitude = torch.norm(hind_forces, dim=-1)
        normalized_force = torch.clamp(
            force_magnitude / self.cfg.tracking_contact_force_norm, min=0.0, max=1.0
        )
        desired_contacts = torch.clamp(self.desired_contact_states, 0.0, 1.0)
        tracking_error = torch.abs(normalized_force - desired_contacts)
        per_foot_reward = 1.0 - torch.clamp(tracking_error, 0.0, 1.0)
        return torch.mean(per_foot_reward, dim=1)

    def _penalty_hind_calf_contact(self) -> torch.Tensor:
        """Ungated penalty for hind-calf ground contact — sitting back is never free."""
        calf_forces = self._contact_sensor.data.net_forces_w_history[:, 0, self._hind_calf_ids_sensor, :]
        force_magnitude = torch.norm(calf_forces, dim=-1)
        return (force_magnitude > self.cfg.contact_force_threshold).float().sum(dim=1)

    def _penalty_undesired_contact(self) -> torch.Tensor:
        """Penalize contact on hind thighs/calves and the base (gated: legal during rear-up)."""
        contact_forces = self._contact_sensor.data.net_forces_w_history[:, 0, self._undesired_contact_body_ids_sensor, :]
        force_magnitude = torch.norm(contact_forces, dim=-1)
        return (force_magnitude > self.cfg.contact_force_threshold).float().sum(dim=1)

    def _penalty_torque_magnitude(self) -> torch.Tensor:
        """Penalize mean squared applied torque, normalized by the torque limit."""
        penalty = torch.mean(torch.square(self.applied_torques), dim=1)
        return penalty / (self.torque_limits ** 2)

    def _penalty_action_rate(self) -> torch.Tensor:
        """Penalize first derivative of actions (a_t - a_{t-1}) for smoothness."""
        action_diff = self._actions - self.last_actions[:, :, 0]
        return torch.mean(torch.square(action_diff), dim=1)

    def _penalty_action_jerk(self) -> torch.Tensor:
        """Penalize second derivative of actions (a_t - 2a_{t-1} + a_{t-2}) to reduce jitter."""
        action_jerk = (
            self._actions
            - 2 * self.last_actions[:, :, 0]
            + self.last_actions[:, :, 1]
        )
        return torch.mean(torch.square(action_jerk), dim=1)

    # -- commands ------------------------------------------------------------------

    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        """Sample new (vx, yaw rate) commands; lateral vy is always 0 for the biped task."""
        num_resets = len(env_ids)
        vx_min, vx_max = self.cfg.lin_vel_x_range
        self._commands[env_ids, 0] = sample_uniform(vx_min, vx_max, (num_resets,), device=self.device)
        self._commands[env_ids, 1] = 0.0
        yaw_min, yaw_max = self.cfg.ang_vel_yaw_range
        self._commands[env_ids, 2] = sample_uniform(yaw_min, yaw_max, (num_resets,), device=self.device)

    def _update_commands(self) -> None:
        """Periodically resample commands during episodes for robust tracking."""
        self.command_time += self.step_dt
        resample_mask = self.command_time >= self.cfg.command_resample_time
        if resample_mask.any():
            env_ids = torch.where(resample_mask)[0]
            self._sample_commands(env_ids)
            self.command_time[resample_mask] = 0.0

    # -- terminations --------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        pg = self.robot.data.projected_gravity_b
        # Irrecoverable-fall checks only. NOTE: the quadruped upside-down check
        # (pg[:, 2] > 0) would fire the instant pitch passes 90 deg — the nominal
        # biped stance — so the backward check uses a wide margin instead.
        fell_backward = pg[:, 2] > self.cfg.fall_backward_threshold
        fell_forward = pg[:, 0] > self.cfg.fall_forward_threshold
        fell_sideways = torch.abs(pg[:, 1]) > self.cfg.fall_sideways_threshold
        self._fell_backward = fell_backward
        self._fell_forward = fell_forward
        self._fell_sideways = fell_sideways
        died = fell_backward | fell_forward | fell_sideways
        return died, time_out

    # -- gait clock ----------------------------------------------------------------

    def _step_contact_targets(self):
        """Two-foot (RL, RR) gait clock with 180 deg phase offset."""
        frequencies = self.cfg.gait_frequency
        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

        foot_indices = [
            self.gait_indices,
            torch.remainder(self.gait_indices + 0.5, 1.0),
        ]

        self.foot_indices = torch.remainder(
            torch.cat([foot_indices[i].unsqueeze(1) for i in range(2)], dim=1), 1.0
        )

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                        0.5 / (1 - durations[swing_idxs]))

        for i in range(2):
            self.clock_inputs[:, i] = torch.sin(2 * math.pi * foot_indices[i])

        # von mises distribution
        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

        for i in range(2):
            phase = torch.remainder(foot_indices[i], 1.0)
            smoothing_multiplier = (
                smoothing_cdf_start(phase) * (1 - smoothing_cdf_start(phase - 0.5))
                + smoothing_cdf_start(phase - 1) * (1 - smoothing_cdf_start(phase - 0.5 - 1))
            )
            self.desired_contact_states[:, i] = smoothing_multiplier

    # -- reset ---------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.applied_torques[env_ids] = 0.0
        self.gait_indices[env_ids] = 0
        self.desired_contact_states[env_ids] = 0.0
        self.command_time[env_ids] = 0.0
        # Sample new commands on reset
        self._sample_commands(env_ids)
        # Reset robot state to the quadruped default pose — the policy must learn the rear-up
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
        reset_terminated = self.reset_terminated[env_ids]
        extras["Episode_Termination/fell"] = torch.count_nonzero(reset_terminated).item()
        if hasattr(self, "_fell_backward"):
            extras["Episode_Termination/fell_backward"] = torch.count_nonzero(self._fell_backward[env_ids]).item()
            extras["Episode_Termination/fell_forward"] = torch.count_nonzero(self._fell_forward[env_ids]).item()
            extras["Episode_Termination/fell_sideways"] = torch.count_nonzero(self._fell_sideways[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    # -- debug visualization -------------------------------------------------------

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
