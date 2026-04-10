# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Short reset-focused smoke test for quadcopter_copy environment."""

import argparse
import os
import sys

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

# Prefer local task sources over installed package variants.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_TASKS_SRC = os.path.join(REPO_ROOT, "source", "isaaclab_tasks")
if LOCAL_TASKS_SRC not in sys.path:
    sys.path.insert(0, LOCAL_TASKS_SRC)

parser = argparse.ArgumentParser(description="Short reset diagnostics for quadcopter_copy")
parser.add_argument("--num_envs", type=int, default=16, help="Number of vectorized envs")
parser.add_argument("--steps", type=int, default=40, help="How many simulation steps to run")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.direct.quadcopter_copy import QuadcopterWindingCorridorEnvCfg  # noqa: E402


def main() -> None:
    env_cfg = QuadcopterWindingCorridorEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make("Isaac-Quadcopter-WindingCorridor-Direct-v0", cfg=env_cfg)
    action_dim = env.unwrapped.single_action_space.shape[0]

    obs, _ = env.reset()
    _ = obs

    total_resets = 0
    print(f"[short-test] num_envs={env.num_envs}, steps={args_cli.steps}")

    for step in range(args_cli.steps):
        actions = 2.0 * torch.rand((env.num_envs, action_dim), device=env.device) - 1.0
        _, _, terminated, truncated, _ = env.step(actions)

        resets = torch.count_nonzero(terminated | truncated).item()
        total_resets += resets

        if (step + 1) % 10 == 0 or step == 0:
            reset_ratio = resets / max(env.num_envs, 1)
            print(f"step={step + 1:03d} resets={resets:3d} ratio={reset_ratio:.3f}")

    avg_resets_per_step = total_resets / max(args_cli.steps, 1)
    avg_reset_ratio_per_step = avg_resets_per_step / max(env.num_envs, 1)
    print("[short-test] summary")
    print(f"total_resets={total_resets}")
    print(f"avg_resets_per_step={avg_resets_per_step:.3f}")
    print(f"avg_reset_ratio_per_step={avg_reset_ratio_per_step:.3f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
