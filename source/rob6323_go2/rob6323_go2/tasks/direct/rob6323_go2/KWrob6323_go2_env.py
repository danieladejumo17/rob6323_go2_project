# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Licensed under the BSD-3-Clause License.

from __future__ import annotations
import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform, wrap_to_pi
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from .rob6323_go2_biped_env_cfg import Rob6323Go2BipedEnvCfg

class Rob6323Go2BipedEnv(DirectRLEnv):
    cfg: Rob6323Go2BipedEnvCfg

    def __init__(self, cfg: Rob6323Go2BipedEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action storage (6 DOF for training, but we need 12 DOF for the actual robot)
        self._actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._previous_previous_actions = torch.zeros(self.num_envs, 6, device=self.device)
        
        # Indices for rear legs [RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
        # In Go2, typically: FL=0-2, FR=3-5, RL=6-8, RR=9-11
        self.rear_leg_indices = [6, 7, 8, 9, 10, 11]
        self.front_leg_indices = [0, 1, 2, 3, 4, 5]

        # Commands: [x_vel, y_vel, yaw_rate]
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._command_timer = torch.zeros(self.num_envs, device=self.device)
        
        # Gait clock
        self._clock_phase = torch.zeros(self.num_envs, device=self.device)
        self._gait_freq = 1.5 # Hz

        # Scene assets
        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene["contact_forces"]
        
        # Markers
        self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer)
        self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        self.contact_sensor = ContactSensor(self.cfg.contact_forces)
        self.scene.sensors["contact_forces"] = self.contact_sensor
        super()._setup_scene()

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        
        # Full 12 DOF target
        full_actions = torch.zeros(self.num_envs, 12, device=self.device)
        
        # Keep front legs at default (0 deviation)
        # Apply learned actions to rear legs
        full_actions[:, self.rear_leg_indices] = self._actions * self.cfg.action_scale
        
        # Set targets
        targets = full_actions + self.robot.data.default_joint_pos
        self.robot.set_joint_position_target(targets)

    def _apply_action(self):
        pass # Handled in pre_physics_step for Articulation

    def _get_observations(self) -> dict:
        # Periodic clock inputs
        self._clock_phase = (self._clock_phase + self.cfg.sim.dt * self.cfg.decimation * self._gait_freq) % 1.0
        sin_clock = torch.sin(2 * torch.pi * self._clock_phase).unsqueeze(1)
        cos_clock = torch.cos(2 * torch.pi * self._clock_phase).unsqueeze(1)
        
        # Base state
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        projected_gravity = self.robot.data.projected_gravity_b
        
        # Rear Joint state
        joint_pos = self.robot.data.joint_pos[:, self.rear_leg_indices] - self.robot.data.default_joint_pos[:, self.rear_leg_indices]
        joint_vel = self.robot.data.joint_vel[:, self.rear_leg_indices]
        
        obs = torch.cat([
            base_lin_vel, base_ang_vel, projected_gravity,
            joint_pos, joint_vel,
            self._actions,
            self._commands,
            sin_clock, cos_clock
        ], dim=-1)
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # 1. Velocity Tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        reward_lin_vel = torch.exp(-lin_vel_error / 0.25) * self.cfg.lin_vel_reward_scale
        
        # 2. Stability (Bipedal specific)
        roll_pitch = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        reward_stability = roll_pitch * self.cfg.roll_pitch_penalty_scale
        
        # 3. Bipedal Gait (Alternating rear legs)
        # Left Rear (index 2 in sensor), Right Rear (index 3 in sensor)
        contact_forces = self.contact_sensor.data.net_forces_w[:, [2, 3], 2]
        left_contact = (contact_forces[:, 0] > 1.0).float()
        right_contact = (contact_forces[:, 1] > 1.0).float()
        
        # Desired contact phases (180 deg offset)
        desired_left_contact = (torch.sin(2 * torch.pi * self._clock_phase) > 0).float()
        desired_right_contact = (torch.sin(2 * torch.pi * (self._clock_phase + 0.5)) > 0).float()
        
        reward_phase = (
            (left_contact == desired_left_contact).float() + 
            (right_contact == desired_right_contact).float()
        ) * self.cfg.biped_phase_reward_scale

        return reward_lin_vel + reward_stability + reward_phase

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Termination: fall down or timeout
        height = self.robot.data.root_pos_w[:, 2]
        died = height < self.cfg.base_height_min
        time_out = self.episode_length_buf >= self.max_episode_length
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        # Randomize commands
        self._commands[env_ids, 0] = sample_uniform(*self.cfg.lin_vel_x_range, (len(env_ids),), self.device)
        self._commands[env_ids, 1] = sample_uniform(*self.cfg.lin_vel_y_range, (len(env_ids),), self.device)
        self._commands[env_ids, 2] = sample_uniform(*self.cfg.yaw_rate_range, (len(env_ids),), self.device)
        
        # Reset joint positions to default
        pos = self.robot.data.default_joint_pos[env_ids]
        vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])
        self.robot.write_joint_state_to_sim(pos, vel, env_ids)
        
        self._clock_phase[env_ids] = torch.rand(len(env_ids), device=self.device)