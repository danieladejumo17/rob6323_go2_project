# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quantitative headless evaluation of a trained biped (hind-leg) locomotion policy.

Reports standing/tracking metrics instead of a video (the local WSL container has no
working renderer). Usage mirrors play.py:

    SCRIPT=/workspace/run/scripts/rsl_rl/eval_biped.py ./scripts/train_local.sh \
        --task Template-Rob6323-Go2-Biped-Direct-v0 --num_envs 64 --headless \
        --checkpoint /workspace/isaaclab/logs/rsl_rl/go2_biped_direct/<run>/model_2999.pt
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a biped RSL-RL policy quantitatively.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--eval_steps", type=int, default=1500, help="Total policy steps to simulate (50 Hz).")
parser.add_argument("--settle_steps", type=int, default=400, help="Steps to skip before collecting metrics (rear-up phase).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rob6323_go2.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw_env = env.unwrapped
    num_envs = raw_env.num_envs

    # accumulators (post-settle)
    n = 0
    sum_h = 0.0
    sum_up = 0.0
    standing_steps = 0
    sum_lin_err = 0.0
    sum_yaw_err = 0.0
    front_contact_steps = 0
    falls = 0

    obs = env.get_observations()
    for step in range(args_cli.eval_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            if step >= args_cli.settle_steps:
                n += 1
                up_proj = raw_env._get_up_proj()
                h = raw_env._get_base_height()
                v_fwd, v_lat = raw_env._get_heading_velocities()
                sum_up += up_proj.mean().item()
                sum_h += h.mean().item()
                standing_steps += ((up_proj > 0.9) & (h > 0.45)).float().mean().item()
                sum_lin_err += (raw_env._commands[:, 0] - v_fwd).abs().mean().item()
                sum_yaw_err += (raw_env._commands[:, 2] - raw_env.robot.data.root_ang_vel_w[:, 2]).abs().mean().item()
                front_forces = raw_env._contact_sensor.data.net_forces_w_history[:, 0, raw_env._front_feet_ids_sensor, :]
                front_contact_steps += (
                    (front_forces.norm(dim=-1) > raw_env.cfg.contact_force_threshold).any(dim=1).float().mean().item()
                )
                falls += int(raw_env.reset_terminated.sum().item())

    print("=" * 60)
    print("BIPED EVAL RESULTS")
    print(f"  envs: {num_envs}, steps: {args_cli.eval_steps} (metrics after step {args_cli.settle_steps})")
    print(f"  mean uprightness (up_proj, 1.0 = vertical trunk): {sum_up / n:.3f}")
    print(f"  mean base height [m] (target {raw_env.cfg.biped_height_target}): {sum_h / n:.3f}")
    print(f"  standing fraction (up_proj>0.9 & h>0.45): {standing_steps / n:.3f}")
    print(f"  mean |v_fwd error| [m/s]: {sum_lin_err / n:.3f}")
    print(f"  mean |yaw rate error| [rad/s]: {sum_yaw_err / n:.3f}")
    print(f"  front-foot contact fraction: {front_contact_steps / n:.3f}")
    print(f"  fall terminations (env-episodes): {falls}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
