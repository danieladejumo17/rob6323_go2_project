# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

@configclass
class Rob6323Go2BipedEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    
    # - spaces definition (6 DOF: RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf)
    action_scale = 0.25
    action_space = 6 
    observation_space = 30 + 4  # Reduced joint space (6 pos + 6 vel) + base + clock
    debug_vis = True
    
    # Command following
    command_resample_time = 3.0
    lin_vel_x_range = (0.0, 1.5)
    lin_vel_y_range = (-0.5, 0.5)
    yaw_rate_range = (-1.0, 1.0)

    # Bipedal Termination (Higher min height than quadruped)
    base_height_min = 0.28  

    # Reward scales
    lin_vel_reward_scale = 1.5
    yaw_rate_reward_scale = 0.5
    
    # Penalties (Higher for bipeds to maintain balance)
    torque_magnitude_penalty_scale = -2e-4
    action_rate_penalty_scale = -0.01
    roll_pitch_penalty_scale = -2.0  # Crucial for bipeds
    
    # Bipedal Gait shaping
    # We use a 180 degree phase offset for alternating legs
    biped_symmetry_reward_scale = 1.0
    biped_phase_reward_scale = 1.0
    feet_clearance_reward_scale = -10.0
    
    # Robot configuration
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # Scene
    scene = sim_utils.InteractiveSceneCfg(num_envs=4096, env_spacing=2.5)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
    )

    # Sensors (Focused on Rear Feet)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot", 
        update_period=0.0, 
        history_length=3, 
        debug_vis=True
    )

    # Visualizers
    goal_vel_visualizer = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/CommandArrow")
    current_vel_visualizer = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/CurrentVelArrow")

    # Simulation
    sim = sim_utils.SimulationCfg(dt=0.005, render_interval=decimation)