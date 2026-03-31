# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation helpers for the quadcopter winding-corridor task."""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import quat_rotate_inverse, subtract_frame_transforms


def _wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angles + math.pi, 2.0 * math.pi) - math.pi


def compute_body_frame_pillar_vectors(env) -> torch.Tensor:
    """Return pillar center vectors in the quadcopter body frame."""

    env_origins = env._terrain.env_origins
    pos_local_xy = (env._robot.data.root_pos_w - env_origins)[:, :2]
    rel_local_xy = env._pillar_centers_xy - pos_local_xy.unsqueeze(1)

    rel_local = torch.zeros(env.num_envs, env.cfg.num_pillars, 3, device=env.device)
    rel_local[..., :2] = rel_local_xy

    root_quat = env._robot.data.root_quat_w[:, None, :].expand(-1, env.cfg.num_pillars, -1)
    rel_body = quat_rotate_inverse(root_quat.reshape(-1, 4), rel_local.reshape(-1, 3))
    return rel_body.reshape(env.num_envs, env.cfg.num_pillars, 3)


def compute_pillar_surface_distances(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return body-frame pillar vectors and surface distances."""

    rel_body = compute_body_frame_pillar_vectors(env)
    planar_vecs = rel_body[..., :2]
    surface_dist = torch.linalg.norm(planar_vecs, dim=-1) - float(env.cfg.pillar_radius)
    surface_dist = torch.clamp(surface_dist, min=0.0, max=float(env.cfg.obstacle_max_range))
    surface_dist = torch.where(
        env._active_pillar_mask,
        surface_dist,
        torch.full_like(surface_dist, float(env.cfg.obstacle_max_range)),
    )
    return planar_vecs, surface_dist


def compute_sector_obstacle_distances(env) -> torch.Tensor:
    """Return 8 sector-style obstacle distances in the body frame."""

    planar_vecs, surface_dist = compute_pillar_surface_distances(env)
    pillar_angles = torch.atan2(planar_vecs[..., 1], planar_vecs[..., 0])

    sector_angles = env._sector_angles.view(1, 1, -1)
    angle_diff = _wrap_to_pi(pillar_angles.unsqueeze(-1) - sector_angles)
    half_sector_width = math.pi / float(env.cfg.obstacle_sector_count)
    in_sector = torch.abs(angle_diff) <= half_sector_width
    valid = in_sector & env._active_pillar_mask.unsqueeze(-1)

    max_range = torch.full(
        (env.num_envs, env.cfg.num_pillars, env.cfg.obstacle_sector_count),
        float(env.cfg.obstacle_max_range),
        device=env.device,
    )
    sector_candidates = torch.where(valid, surface_dist.unsqueeze(-1), max_range)
    return torch.min(sector_candidates, dim=1).values


def compute_nearest_obstacle_vectors(env, count: int = 3) -> torch.Tensor:
    """Return the body-frame XY vectors to the nearest active pillars."""

    planar_vecs, surface_dist = compute_pillar_surface_distances(env)
    inf = torch.full_like(surface_dist, float("inf"))
    ranked_dist = torch.where(env._active_pillar_mask, surface_dist, inf)
    topk = torch.topk(ranked_dist, k=count, dim=1, largest=False).indices
    gather_ids = topk.unsqueeze(-1).expand(-1, -1, 2)
    nearest_vecs = torch.gather(planar_vecs, dim=1, index=gather_ids)
    return torch.nan_to_num(nearest_vecs, nan=0.0, posinf=0.0, neginf=0.0)


def compute_policy_observation(env) -> torch.Tensor:
    """Assemble the 24-dimensional actor observation."""

    goal_b, _ = subtract_frame_transforms(
        env._robot.data.root_pos_w,
        env._robot.data.root_quat_w,
        env._goal_pos_w,
    )
    obstacle_distances = compute_sector_obstacle_distances(env)

    env._waypoint_pos_w = env._goal_pos_w.clone()

    return torch.cat(
        [
            env._robot.data.root_lin_vel_b,
            env._robot.data.root_ang_vel_b,
            env._robot.data.projected_gravity_b,
            goal_b,
            obstacle_distances,
            env._actions,
        ],
        dim=-1,
    )


def compute_critic_observation(env, policy_obs: torch.Tensor) -> torch.Tensor:
    """Assemble the privileged critic observation."""

    env_origins = env._terrain.env_origins
    pos_local = env._robot.data.root_pos_w - env_origins
    nearest_vecs = compute_nearest_obstacle_vectors(env, count=3).reshape(env.num_envs, 6)
    return torch.cat(
        [
            policy_obs,
            pos_local,
            env._wind_force[:, 0, :],
            nearest_vecs,
        ],
        dim=-1,
    )


def get_observations(env) -> dict[str, torch.Tensor]:
    """Return policy observations and optional privileged critic observations."""

    policy_obs = compute_policy_observation(env)
    observations = {"policy": policy_obs}
    if env.cfg.use_privileged_critic:
        observations["critic"] = compute_critic_observation(env, policy_obs)
    return observations
