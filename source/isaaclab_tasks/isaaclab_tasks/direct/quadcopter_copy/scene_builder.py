# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Scene construction helpers for the quadcopter winding-corridor task."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg

from .env_cfg import QuadcopterWindingCorridorEnvCfg


def generate_pillar_layout(
    cfg: QuadcopterWindingCorridorEnvCfg, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one obstacle layout with a guaranteed clear central corridor."""

    num_pillars = int(cfg.num_pillars)
    active_count = int(min(cfg.active_pillar_count, cfg.num_pillars))
    centers = torch.zeros(num_pillars, 2, device=device)
    centers[:, 0] = -(float(cfg.field_length) + 5.0)
    centers[:, 1] = 0.0
    active_mask = torch.zeros(num_pillars, dtype=torch.bool, device=device)

    field_half_width = float(cfg.field_width) / 2.0
    y_limit = max(field_half_width - float(cfg.pillar_radius) - 0.05, 0.1)
    x_min = float(cfg.start_zone_length)
    x_max = float(cfg.field_length) - float(cfg.goal_zone_length)
    corridor_half_width = float(cfg.clear_corridor_half_width)
    min_spacing = float(cfg.pillar_min_spacing)
    max_attempts = max(512, active_count * 32)

    placed = 0
    attempts = 0
    while placed < active_count and attempts < max_attempts:
        attempts += 1
        candidate_x = torch.rand(1, device=device) * (x_max - x_min) + x_min
        candidate_y = torch.rand(1, device=device) * (2.0 * y_limit) - y_limit

        if torch.abs(candidate_y - float(cfg.goal_y_position)) < corridor_half_width:
            continue

        candidate = torch.stack((candidate_x.squeeze(0), candidate_y.squeeze(0)))
        if placed > 0:
            distances = torch.linalg.norm(centers[:placed] - candidate.unsqueeze(0), dim=-1)
            if torch.any(distances < min_spacing):
                continue

        centers[placed] = candidate
        active_mask[placed] = True
        placed += 1

    if placed < active_count:
        fallback_x = torch.linspace(x_min, x_max, active_count - placed + 2, device=device)[1:-1]
        fallback_y = torch.empty(active_count - placed, device=device)
        fallback_y[0::2] = corridor_half_width + 0.35
        fallback_y[1::2] = -(corridor_half_width + 0.35)
        fallback_y = torch.clamp(fallback_y, min=-y_limit, max=y_limit)
        count = active_count - placed
        centers[placed:active_count, 0] = fallback_x[:count]
        centers[placed:active_count, 1] = fallback_y[:count]
        active_mask[placed:active_count] = True

    return centers, active_mask


def build_pillars_cfg(
    pillar_centers_xy: torch.Tensor,
    cfg: QuadcopterWindingCorridorEnvCfg,
) -> RigidObjectCollectionCfg:
    """Build the rigid-object collection used for the pillar set."""

    rigid_objects: dict[str, RigidObjectCfg] = {}
    z_center = float(cfg.pillar_height) / 2.0

    for i in range(pillar_centers_xy.shape[0]):
        px = float(pillar_centers_xy[i, 0].item())
        py = float(pillar_centers_xy[i, 1].item())
        rigid_objects[f"pillar_{i:02d}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Pillar_{i:02d}",
            spawn=sim_utils.CylinderCfg(
                radius=float(cfg.pillar_radius),
                height=float(cfg.pillar_height),
                axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(px, py, z_center),
                rot=(1.0, 0.0, 0.0, 0.0),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
            ),
        )

    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


def build_boundary_walls_cfg(cfg: QuadcopterWindingCorridorEnvCfg) -> RigidObjectCollectionCfg:
    """Build the boundary walls that enclose the field."""

    field_length = float(cfg.field_length)
    field_half_width = float(cfg.field_width) / 2.0
    wall_height = float(cfg.boundary_wall_height)
    wall_thickness = float(cfg.boundary_wall_thickness)
    z_center = wall_height / 2.0

    def _wall(prim_name: str, size: tuple[float, float, float], pos: tuple[float, float, float]) -> RigidObjectCfg:
        return RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/{prim_name}",
            spawn=sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.2, 0.2),
                    opacity=0.3,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=pos,
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

    rigid_objects = {
        "boundary_wall_left": _wall(
            "BoundaryWall_Left",
            size=(field_length, wall_thickness, wall_height),
            pos=(field_length / 2.0, -field_half_width, z_center),
        ),
        "boundary_wall_right": _wall(
            "BoundaryWall_Right",
            size=(field_length, wall_thickness, wall_height),
            pos=(field_length / 2.0, field_half_width, z_center),
        ),
        "boundary_wall_back": _wall(
            "BoundaryWall_Back",
            size=(wall_thickness, float(cfg.field_width), wall_height),
            pos=(0.0, 0.0, z_center),
        ),
        "boundary_wall_front": _wall(
            "BoundaryWall_Front",
            size=(wall_thickness, float(cfg.field_width), wall_height),
            pos=(field_length, 0.0, z_center),
        ),
    }
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


def reset_pillar_layouts(env, env_ids: torch.Tensor) -> None:
    """Sample new obstacle layouts for the selected environments."""

    num_envs_to_reset = len(env_ids)
    num_layouts = max(1, min(num_envs_to_reset, int(env.cfg.layout_batch_size)))

    layouts = []
    masks = []
    for _ in range(num_layouts):
        layout, active_mask = generate_pillar_layout(env.cfg, env.device)
        layouts.append(layout)
        masks.append(active_mask)

    layout_bank = torch.stack(layouts, dim=0)
    mask_bank = torch.stack(masks, dim=0)
    bank_ids = torch.arange(num_envs_to_reset, device=env.device) % num_layouts

    local_layouts = layout_bank[bank_ids]
    local_masks = mask_bank[bank_ids]

    env._pillar_centers_xy[env_ids] = local_layouts
    env._active_pillar_mask[env_ids] = local_masks

    pillar_pose = torch.zeros(num_envs_to_reset, env.cfg.num_pillars, 7, device=env.device)
    pillar_pose[..., 0:2] = local_layouts + env._terrain.env_origins[env_ids, None, 0:2]
    pillar_pose[..., 2] = torch.where(
        local_masks,
        torch.full_like(local_layouts[..., 0], float(env.cfg.pillar_height) / 2.0),
        torch.full_like(local_layouts[..., 0], float(env.cfg.inactive_pillar_drop_z)),
    )
    pillar_pose[..., 3] = 1.0
    env._pillars.write_object_pose_to_sim(pillar_pose, env_ids=env_ids)


def setup_scene(env) -> None:
    """Wire the robot, obstacles and terrain into the interactive scene."""

    cfg: QuadcopterWindingCorridorEnvCfg = env.cfg

    env._robot = Articulation(cfg.robot)
    env.scene.articulations["robot"] = env._robot

    initial_layout, initial_mask = generate_pillar_layout(cfg, env.device)
    env._pillar_centers_xy = initial_layout.unsqueeze(0).repeat(env.scene.cfg.num_envs, 1, 1)
    env._active_pillar_mask = initial_mask.unsqueeze(0).repeat(env.scene.cfg.num_envs, 1)

    pillars_cfg = build_pillars_cfg(initial_layout, cfg)
    env._pillars = RigidObjectCollection(pillars_cfg)
    env.scene.rigid_object_collections["pillars"] = env._pillars

    if cfg.boundary_wall_enabled:
        walls_cfg = build_boundary_walls_cfg(cfg)
        env._boundary_walls = RigidObjectCollection(walls_cfg)
        env.scene.rigid_object_collections["boundary_walls"] = env._boundary_walls

    cfg.terrain.num_envs = env.scene.cfg.num_envs
    cfg.terrain.env_spacing = env.scene.cfg.env_spacing
    env._terrain = cfg.terrain.class_type(cfg.terrain)

    env.scene.clone_environments(copy_from_source=False)
    if env.device == "cpu":
        env.scene.filter_collisions(global_prim_paths=[cfg.terrain.prim_path])

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
