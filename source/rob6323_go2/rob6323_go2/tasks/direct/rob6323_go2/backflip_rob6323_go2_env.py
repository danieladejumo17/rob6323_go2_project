# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


import gymnasium as gym
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .backflip_rob6323_go2_env_cfg import BackflipRob6323Go2EnvCfg


class BackflipRob6323Go2Env(DirectRLEnv):
    """Single backflip with recovery to quadruped standing for the Unitree Go2.

    Timed phase schedule (policy observes normalized episode time and rotation
    progress): P0 stand -> P1 flip window (backward rotation rewarded only while
    airborne, plus a jump-height term) -> P2 recover and stand, with standing
    rewards gated by flip completion so skipping the flip never pays.

    Rotation bookkeeping: cum_pitch integrates body pitch rate about +y. In the
    FLU body frame a nose-up/backward rotation is negative about +y (right-hand
    rule), so a completed backflip reaches cum_pitch ~= -2*pi.
    """

    cfg: BackflipRob6323Go2EnvCfg

    def __init__(self, cfg: BackflipRob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "flip_rotation",
                "flip_height",
                "aerial_rotation",
                "flip_completion",
                "stand_upright",
                "stand_height",
                "stand_pose",
                "stand_feet_contact",
                "stand_still",
                "undesired_contact_penalty",
                "joint_limit_penalty",
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

        # Contact sensor body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids_sensor = []
        for name in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]:
            id_list, _ = self._contact_sensor.find_bodies(name)
            self._feet_ids_sensor.append(id_list[0])
        # Non-foot leg bodies: kneeling/sitting on these must never be free
        self._undesired_contact_body_ids_sensor = []
        for pattern in [".*thigh", ".*calf", ".*hip"]:
            id_list, _ = self._contact_sensor.find_bodies(pattern)
            self._undesired_contact_body_ids_sensor.extend(id_list)

        # Integrated body pitch (rad); a completed backflip reaches ~ -2*pi
        self.cum_pitch = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # Running max of flip progress — the rotation reward pays only for new progress
        self.max_flip_progress = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # Backward pitch accumulated while fully airborne (no body in contact) — a real
        # backflip must do at least aerial_rotation_threshold of its rotation ballistic
        self.aerial_pitch = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # PD control parameters
        self.Kp = torch.tensor([cfg.Kp] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.Kd = torch.tensor([cfg.Kd] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.torque_limits = cfg.torque_limits

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # NOTE: without this registration the scene never calls sensor.update() and
        # contact forces stay frozen at zero (bug found in the first training run)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
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
        episode_time_norm = self.episode_length_buf.float() * self.step_dt / self.max_episode_length_s
        rotation_progress = self.cum_pitch / self.cfg.flip_target_rotation
        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                self.robot.data.joint_vel,
                self._actions,
                episode_time_norm.unsqueeze(1),
                rotation_progress.unsqueeze(1),
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    # -- helpers -------------------------------------------------------------------

    def _get_base_height(self) -> torch.Tensor:
        return self.robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]

    def _get_feet_contacts(self) -> torch.Tensor:
        """Binary contact state per foot. Shape: (num_envs, 4), 1 = contact."""
        contact_forces = self._contact_sensor.data.net_forces_w_history[:, 0, self._feet_ids_sensor, :]
        force_magnitude = torch.norm(contact_forces, dim=-1)
        return (force_magnitude > self.cfg.contact_force_threshold).float()

    # -- rewards -------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        # Integrate body pitch rate (backflip direction is negative about +y)
        self.cum_pitch += self.robot.data.root_ang_vel_b[:, 1] * self.step_dt

        # Phase indicators from episode time
        t = self.episode_length_buf.float() * self.step_dt
        in_prepare = (t < self.cfg.flip_start_time).float()
        in_flip = ((t >= self.cfg.flip_start_time) & (t < self.cfg.flip_end_time)).float()
        in_recovery = (t >= self.cfg.flip_end_time).float()

        # Accumulate backward pitch while fully airborne (no contact on ANY body).
        # Gating completion on this kills the ground-tumble optimum (design iter3):
        # rolling over the legs rotates fully but never ballistically.
        all_forces = torch.norm(self._contact_sensor.data.net_forces_w_history[:, 0, :, :], dim=-1)
        fully_airborne = (torch.max(all_forces, dim=1)[0] < self.cfg.contact_force_threshold).float()
        backward_pitch_gain = (-self.robot.data.root_ang_vel_b[:, 1] * self.step_dt).clamp(min=0.0)
        # Credit aerial-rotation increments (capped at one full turn) for direct shaping
        aerial_prev = self.aerial_pitch.clamp(max=self.cfg.flip_target_rotation)
        self.aerial_pitch += fully_airborne * backward_pitch_gain
        aerial_gain = self.aerial_pitch.clamp(max=self.cfg.flip_target_rotation) - aerial_prev

        # Flip completion gate in [0, 1]: rotation complete AND enough of it aerial
        flip_progress = (-self.cum_pitch / self.cfg.flip_target_rotation).clamp(0.0, 1.0)
        aerial_gate = (self.aerial_pitch / self.cfg.aerial_rotation_threshold).clamp(0.0, 1.0)
        flip_done = aerial_gate * (
            (flip_progress - self.cfg.flip_done_gate_lower) / self.cfg.flip_done_gate_width
        ).clamp(0.0, 1.0)

        feet_contacts = self._get_feet_contacts()
        base_height = self._get_base_height()

        # -- flip window terms
        # Monotonic progress reward: pays only for NEW net backward rotation (running
        # max), so rate oscillation cannot farm it — total per episode is bounded by 1.0
        progress_gain = (flip_progress - self.max_flip_progress).clamp(min=0.0)
        self.max_flip_progress = torch.maximum(self.max_flip_progress, flip_progress)
        rew_flip_rotation = (in_flip + in_recovery) * progress_gain
        # Airborne-gated (design iter4): statically rearing up on the hind legs reaches
        # base heights > 0.35 m and farmed this term — only ballistic altitude pays
        rew_flip_height = (
            in_flip * fully_airborne
            * (base_height - self.cfg.flip_height_start).clamp(0.0, self.cfg.flip_height_clip)
        )
        # Direct shaping for airborne backward rotation — bridges "hop" to "aerial flip"
        # continuously instead of only paying at the compound completion gate
        rew_aerial_rotation = (in_flip + in_recovery) * aerial_gain
        rew_flip_completion = (in_flip + in_recovery) * flip_done

        # -- standing terms (P0 ungated; P2 gated by flip completion)
        stand_gate = in_prepare + in_recovery * flip_done
        rew_stand_upright = stand_gate * (-self.robot.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
        rew_stand_height = stand_gate * torch.exp(
            -torch.square(base_height - self.cfg.stand_height_target) / self.cfg.stand_height_width
        )
        pose_err = torch.sum(
            torch.square(self.robot.data.joint_pos - self.robot.data.default_joint_pos), dim=1
        )
        rew_stand_pose = stand_gate * torch.exp(-pose_err / self.cfg.stand_pose_width)
        rew_stand_feet = stand_gate * torch.mean(feet_contacts, dim=1)
        stillness = (
            torch.sum(torch.square(self.robot.data.root_lin_vel_b), dim=1)
            + 0.2 * torch.sum(torch.square(self.robot.data.root_ang_vel_b), dim=1)
        )
        pen_stand_still = stand_gate * stillness

        # -- always-on contact/regularization penalties (compute BEFORE updating last_actions)
        pen_undesired_contact = self._penalty_undesired_contact()
        pen_joint_limits = self._penalty_joint_limits()
        penalty_torque_magnitude = self._penalty_torque_magnitude()
        penalty_action_rate = self._penalty_action_rate()
        penalty_action_jerk = self._penalty_action_jerk()
        pen_dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)

        # Update the prev action hist (roll buffer and insert new action)
        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions[:]

        rewards = {
            "flip_rotation": rew_flip_rotation * self.cfg.flip_rotation_reward_scale,
            "flip_height": rew_flip_height * self.cfg.flip_height_reward_scale,
            "aerial_rotation": rew_aerial_rotation * self.cfg.aerial_rotation_reward_scale,
            "flip_completion": rew_flip_completion * self.cfg.flip_completion_reward_scale,
            "stand_upright": rew_stand_upright * self.cfg.stand_upright_reward_scale,
            "stand_height": rew_stand_height * self.cfg.stand_height_reward_scale,
            "stand_pose": rew_stand_pose * self.cfg.stand_pose_reward_scale,
            "stand_feet_contact": rew_stand_feet * self.cfg.stand_feet_contact_reward_scale,
            "stand_still": pen_stand_still * self.cfg.stand_still_penalty_scale,
            "undesired_contact_penalty": pen_undesired_contact * self.cfg.undesired_contact_penalty_scale,
            "joint_limit_penalty": pen_joint_limits * self.cfg.joint_limit_penalty_scale,
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

    def _penalty_undesired_contact(self) -> torch.Tensor:
        """Penalize ground contact on thighs/calves/hips (kneeling, sitting, ground-rolling)."""
        contact_forces = self._contact_sensor.data.net_forces_w_history[:, 0, self._undesired_contact_body_ids_sensor, :]
        force_magnitude = torch.norm(contact_forces, dim=-1)
        return (force_magnitude > self.cfg.contact_force_threshold).float().sum(dim=1)

    def _penalty_joint_limits(self) -> torch.Tensor:
        """Penalize joint positions beyond the soft limits."""
        q = self.robot.data.joint_pos
        lim = self.robot.data.soft_joint_pos_limits  # (num_envs, 12, 2)
        out_of_limits = (lim[..., 0] - q).clamp(min=0.0) + (q - lim[..., 1]).clamp(min=0.0)
        return torch.sum(out_of_limits, dim=1)

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

    # -- terminations --------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Crash: base touching the ground at any time (a failed flip lands on the back).
        # NOTE: no upside-down or min-height check — inverted mid-flip and a deep
        # crouch are both nominal states for this task.
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0]
            > self.cfg.base_contact_force_threshold,
            dim=1,
        )
        return died, time_out

    # -- reset ---------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        # NOTE: unlike the locomotion tasks, episode_length_buf is NOT randomized on
        # the initial full reset — the phase schedule is tied to episode time, so
        # staggered starts would drop envs mid-timeline without having flipped.
        self._actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.applied_torques[env_ids] = 0.0
        self.cum_pitch[env_ids] = 0.0
        self.max_flip_progress[env_ids] = 0.0
        # Log aerial rotation before clearing (diagnostic for the tumble-vs-flip question)
        self._last_aerial_pitch_log = torch.mean(self.aerial_pitch[env_ids]).item()
        self.aerial_pitch[env_ids] = 0.0
        # Reset robot state to the default quadruped pose
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
        extras["Episode_Termination/base_crash"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Episode_Metric/aerial_pitch"] = self._last_aerial_pitch_log
        self.extras["log"].update(extras)
