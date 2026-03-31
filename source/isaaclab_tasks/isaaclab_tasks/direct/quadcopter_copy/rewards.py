# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward helpers for the quadcopter winding-corridor task."""

from __future__ import annotations

import torch


def rew_goal_progress(env, current_goal_dist: torch.Tensor) -> torch.Tensor:
    """Reward progress towards the goal using distance delta."""

    delta = env._prev_goal_dist - current_goal_dist
    return torch.clamp(delta, min=-0.5, max=0.5)


def rew_goal_proximity(env, current_goal_dist: torch.Tensor) -> torch.Tensor:
    """Small dense shaping term that grows near the goal."""

    return 1.0 - torch.tanh(current_goal_dist / float(env.cfg.waypoint_tanh_std))


def rew_upright(env) -> torch.Tensor:
    """Reward staying close to upright flight."""

    gravity_z = env._robot.data.projected_gravity_b[:, 2]
    return ((gravity_z + 1.0) / 2.0) ** 2


def rew_height_tracking(env, env_origins: torch.Tensor) -> torch.Tensor:
    """Reward keeping the vehicle close to the nominal flight height."""

    current_height = (env._robot.data.root_pos_w - env_origins)[:, 2]
    height_error = torch.abs(current_height - float(env.cfg.flight_z))
    return torch.exp(-3.0 * height_error)


def rew_ang_vel_penalty(env) -> torch.Tensor:
    """Penalize excessive angular velocity."""

    return torch.sum(torch.square(env._robot.data.root_ang_vel_b), dim=1)


def rew_action_smoothness_penalty(env) -> torch.Tensor:
    """Penalize abrupt changes in the commanded action."""

    return torch.sum(torch.square(env._actions - env._previous_actions), dim=1)


def rew_height_violation_penalty(env, env_origins: torch.Tensor) -> torch.Tensor:
    """Penalize flying above the allowed pillar stage height."""

    current_height = (env._robot.data.root_pos_w - env_origins)[:, 2]
    excess = torch.clamp(current_height - float(env.cfg.max_height_pillar_stage), min=0.0)
    return excess**2


def rew_collision_penalty(env) -> torch.Tensor:
    """Apply a penalty when the quadcopter terminates from collision."""

    collision = env._term_hit_pillar | env._term_wall_contact | env._term_ground_contact
    return collision.float()


def rew_success_bonus(env) -> torch.Tensor:
    """Sparse bonus for successful goal completion."""

    return env._goal_reached.float()


def get_rewards(env) -> torch.Tensor:
    """Compute the step reward and update reward logging buffers."""

    env_origins = env._terrain.env_origins
    current_goal_dist = torch.linalg.norm(env._goal_pos_w - env._robot.data.root_pos_w, dim=1)

    components = {
        "goal_progress": rew_goal_progress(env, current_goal_dist) * env.cfg.goal_progress_reward_scale,
        "goal_proximity": rew_goal_proximity(env, current_goal_dist) * env.cfg.goal_proximity_reward_scale * env.step_dt,
        "upright": rew_upright(env) * env.cfg.upright_reward_scale * env.step_dt,
        "height": rew_height_tracking(env, env_origins) * env.cfg.height_reward_scale * env.step_dt,
        "ang_vel": rew_ang_vel_penalty(env) * env.cfg.ang_vel_penalty_scale * env.step_dt,
        "action_smoothness": rew_action_smoothness_penalty(env)
        * env.cfg.action_smoothness_penalty_scale
        * env.step_dt,
        "height_violation": rew_height_violation_penalty(env, env_origins)
        * env.cfg.height_violation_penalty_scale
        * env.step_dt,
        "collision": rew_collision_penalty(env) * env.cfg.collision_penalty,
        "success_bonus": rew_success_bonus(env) * env.cfg.success_bonus,
    }

    total = torch.zeros(env.num_envs, device=env.device)
    for key, value in components.items():
        env._episode_sums[key] += value
        total += value

    env._prev_goal_dist[:] = current_goal_dist
    return total
