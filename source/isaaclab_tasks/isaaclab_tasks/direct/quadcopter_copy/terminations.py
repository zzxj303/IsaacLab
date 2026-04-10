# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Termination helpers for the quadcopter winding-corridor task."""

from __future__ import annotations

import math

import torch


def _local_position(env) -> torch.Tensor:
    return env._robot.data.root_pos_w - env._terrain.env_origins


def _nearest_pillar_surface_distance(env, pos_local: torch.Tensor) -> torch.Tensor:
    rel_xy = env._pillar_centers_xy - pos_local[:, None, :2]
    surface_dist = torch.linalg.norm(rel_xy, dim=-1) - float(env.cfg.pillar_radius)
    inactive_value = torch.full_like(surface_dist, float("inf"))
    surface_dist = torch.where(env._active_pillar_mask, surface_dist, inactive_value)
    return torch.min(surface_dist, dim=1).values


def term_out_of_bounds(env, pos_local: torch.Tensor) -> torch.Tensor:
    """True when the drone exits the task envelope."""

    return (
        (pos_local[:, 2] < 0.05)
        | (pos_local[:, 2] > float(env.cfg.max_flight_height))
        | (pos_local[:, 0] < -0.5)
        | (pos_local[:, 0] > float(env.cfg.field_length) + 0.5)
    )


def term_out_of_corridor(env, pos_local: torch.Tensor) -> torch.Tensor:
    """True when the drone exits the corridor width."""

    field_half_width = float(env.cfg.field_width) / 2.0
    return torch.abs(pos_local[:, 1]) > field_half_width


def term_flying_over_pillars(env, pos_local: torch.Tensor) -> torch.Tensor:
    """True when the drone exceeds the pillar stage ceiling."""

    max_allowed = float(env.cfg.max_height_pillar_stage) + float(env.cfg.height_violation_margin)
    return pos_local[:, 2] > max_allowed


def term_excessive_tilt(env) -> torch.Tensor:
    """True when the body tilt exceeds the configured limit."""

    tilt_limit = -math.cos(float(env.cfg.max_tilt_rad))
    return env._robot.data.projected_gravity_b[:, 2] > tilt_limit


def term_ground_contact(env, pos_local: torch.Tensor) -> torch.Tensor:
    """True when the body drops into the ground-contact band."""

    return pos_local[:, 2] <= float(env.cfg.ground_contact_height)


def term_wall_contact(env, pos_local: torch.Tensor) -> torch.Tensor:
    """True when the body enters the wall-contact margin near a boundary wall."""

    field_half_width = float(env.cfg.field_width) / 2.0
    margin = float(env.cfg.wall_contact_margin)
    near_side_wall = torch.abs(pos_local[:, 1]) >= (field_half_width - margin)
    near_back_wall = (pos_local[:, 0] >= 0.0) & (pos_local[:, 0] <= margin)
    near_front_wall = (pos_local[:, 0] <= float(env.cfg.field_length)) & (
        pos_local[:, 0] >= float(env.cfg.field_length) - margin
    )
    return near_side_wall | near_back_wall | near_front_wall


def term_hit_pillar(env, pos_local: torch.Tensor, ground: torch.Tensor, wall: torch.Tensor) -> torch.Tensor:
    """True when the body intersects the pillar collision radius."""

    nearest_surface = _nearest_pillar_surface_distance(env, pos_local)
    hit_pillar = nearest_surface <= float(env.cfg.pillar_collision_margin)
    return hit_pillar & ~(ground | wall)


def term_success(env) -> torch.Tensor:
    """True when the quadcopter reaches the goal region."""

    goal_dist = torch.linalg.norm(env._goal_pos_w - env._robot.data.root_pos_w, dim=1)
    return goal_dist < float(env.cfg.active_goal_reach_threshold)


def get_dones(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute environment termination and timeout signals."""

    pos_local = _local_position(env)

    time_out = env.episode_length_buf >= env.max_episode_length - 1
    out_of_bounds = term_out_of_bounds(env, pos_local)
    out_of_corridor = term_out_of_corridor(env, pos_local)
    over_height = term_flying_over_pillars(env, pos_local)
    excessive_tilt_raw = term_excessive_tilt(env)
    ground_contact = term_ground_contact(env, pos_local)
    wall_contact = term_wall_contact(env, pos_local)
    hit_pillar = term_hit_pillar(env, pos_local, ground_contact, wall_contact)
    success = term_success(env)

    grace_steps = int(getattr(env.cfg, "termination_grace_steps", 0))
    in_grace = env.episode_length_buf < grace_steps
    excessive_tilt = excessive_tilt_raw & ~in_grace

    env._goal_reached = success
    env._term_out_of_bounds = out_of_bounds | out_of_corridor
    env._term_success = success
    env._term_hit_pillar = hit_pillar
    env._term_wall_contact = wall_contact
    env._term_ground_contact = ground_contact
    env._term_flying_over_pillars = over_height
    env._term_excessive_tilt = excessive_tilt

    terminated = (
        out_of_bounds
        | out_of_corridor
        | over_height
        | excessive_tilt
        | ground_contact
        | wall_contact
        | hit_pillar
        | success
    )
    just_reset = env.episode_length_buf == 0
    terminated = terminated & ~just_reset
    return terminated, time_out
