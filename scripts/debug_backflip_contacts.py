# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnostic: step the backflip env with zero actions and print foot contact forces."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Backflip env contact diagnostic.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import rob6323_go2.tasks  # noqa: F401

TASK = "Template-Rob6323-Go2-Backflip-Direct-v0"


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()
    uenv = env.unwrapped
    for step in range(60):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=uenv.device)
            env.step(actions)
        if step % 10 == 0:
            forces = uenv._contact_sensor.data.net_forces_w_history[:, 0, uenv._feet_ids_sensor, :]
            mags = torch.norm(forces, dim=-1)
            contacts = uenv._get_feet_contacts()
            print(
                f"step {step:3d} | base_h {uenv._get_base_height()[0].item():.3f}"
                f" | foot forces env0 [{', '.join(f'{v:.1f}' for v in mags[0].tolist())}] N"
                f" | contacts env0 {contacts[0].tolist()}"
                f" | mean feet in contact {contacts.mean().item():.2f}"
                f" | cum_pitch env0 {uenv.cum_pitch[0].item():.3f}"
            )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
