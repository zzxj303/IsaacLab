# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for the winding-corridor quadcopter environment."""

import argparse

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test the winding-corridor quadcopter environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.direct.quadcopter_copy import QuadcopterWindingCorridorEnvCfg  # noqa: E402


def main():
    env_cfg = QuadcopterWindingCorridorEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make("Isaac-Quadcopter-WindingCorridor-Direct-v0", cfg=env_cfg)
    action_dim = env.unwrapped.single_action_space.shape[0]

    print("\n" + "=" * 80)
    print("Environment created successfully")
    print("=" * 80)
    print(f"Number of environments: {env.num_envs}")
    print(f"Policy observation space: {env.unwrapped.single_observation_space['policy']}")
    if env.unwrapped.single_observation_space.get("critic") is not None:
        print(f"Critic observation space: {env.unwrapped.single_observation_space['critic']}")
    print(f"Action space: {env.unwrapped.single_action_space}")
    print(f"Curriculum stage: {env_cfg.curriculum_stage}")
    print(f"Active pillars: {env_cfg.active_pillar_count}")
    print(f"Clear corridor half width: {env_cfg.clear_corridor_half_width:.2f}")
    print(f"Wind enabled: {env_cfg.active_wind_enabled}")
    print("=" * 80 + "\n")

    obs, _ = env.reset()
    print("Environment reset successfully")
    print(f"Policy observation shape: {obs['policy'].shape}")
    if "critic" in obs:
        print(f"Critic observation shape: {obs['critic'].shape}")

    print("\nRunning simulation for 100 steps...")
    for step in range(100):
        actions = 2.0 * torch.rand((env.num_envs, action_dim), device=env.device) - 1.0
        obs, rewards, terminated, truncated, info = env.step(actions)
        if step % 20 == 0:
            resets = torch.count_nonzero(terminated | truncated).item()
            print(f"Step {step:03d}: mean reward = {rewards.mean().item():.4f}, resets = {resets}")

    print("\nSimulation completed successfully")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
