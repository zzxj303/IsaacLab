# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate different terrains using trimesh."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh
from isaaclab.terrains.trimesh.utils import make_plane

if TYPE_CHECKING:
    import mesh_terrains_cfg


def custom_repeated_objects_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.CustomMeshRepeatedObjectsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    from .mesh_terrains_cfg import (
        CustomMeshRepeatedBoxesTerrainCfg,
        CustomMeshRepeatedCylindersTerrainCfg,
        CustomMeshRepeatedPyramidsTerrainCfg,
    )

    if isinstance(cfg.object_type, str):
        object_func = globals().get(f"make_{cfg.object_type}")
    else:
        object_func = cfg.object_type
    if not callable(object_func):
        raise ValueError(f"The attribute 'object_type' must be a string or a callable. Received: {object_func}")

    cp_0 = cfg.object_params_start
    cp_1 = cfg.object_params_end
    num_objects = cp_0.num_objects + int(difficulty * (cp_1.num_objects - cp_0.num_objects))
    height = cp_0.height + difficulty * (cp_1.height - cp_0.height)

    if isinstance(cfg, CustomMeshRepeatedBoxesTerrainCfg):
        object_kwargs = {
            "length": cp_0.size[0] + difficulty * (cp_1.size[0] - cp_0.size[0]),
            "width": cp_0.size[1] + difficulty * (cp_1.size[1] - cp_0.size[1]),
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
    elif isinstance(cfg, CustomMeshRepeatedPyramidsTerrainCfg):
        object_kwargs = {
            "radius": cp_0.radius + difficulty * (cp_1.radius - cp_0.radius),
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
    elif isinstance(cfg, CustomMeshRepeatedCylindersTerrainCfg):
        object_kwargs = {
            "radius": cp_0.radius + difficulty * (cp_1.radius - cp_0.radius),
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
    else:
        raise ValueError(f"Unknown terrain configuration: {cfg}")

    platform_clearance = 0.1

    meshes_list = []
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.5 * height))
    platform_corners = np.asarray(
        [
            [origin[0] - cfg.platform_width / 2, origin[1] - cfg.platform_width / 2],
            [origin[0] + cfg.platform_width / 2, origin[1] + cfg.platform_width / 2],
        ]
    )
    platform_corners[0, :] *= 1 - platform_clearance
    platform_corners[1, :] *= 1 + platform_clearance

    object_centers = np.zeros((num_objects, 3))
    masks_left = np.ones((num_objects,), dtype=bool)
    while np.any(masks_left):
        num_objects_left = masks_left.sum()
        object_centers[masks_left, 0] = np.random.uniform(cfg.border_width, cfg.size[0] - cfg.border_width, num_objects_left)
        object_centers[masks_left, 1] = np.random.uniform(cfg.border_width, cfg.size[1] - cfg.border_width, num_objects_left)

        is_within_platform_x = np.logical_and(
            object_centers[masks_left, 0] >= platform_corners[0, 0],
            object_centers[masks_left, 0] <= platform_corners[1, 0],
        )
        is_within_platform_y = np.logical_and(
            object_centers[masks_left, 1] >= platform_corners[0, 1],
            object_centers[masks_left, 1] <= platform_corners[1, 1],
        )
        masks_left[masks_left] = np.logical_and(is_within_platform_x, is_within_platform_y)

    for index in range(len(object_centers)):
        ob_height = height + np.random.uniform(-cfg.max_height_noise, cfg.max_height_noise)
        if ob_height > 0.0:
            object_mesh = object_func(center=object_centers[index], height=ob_height, **object_kwargs)
            meshes_list.append(object_mesh)

    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)

    platform_height = cfg.platform_height if cfg.platform_height is not None else 0.5 * height
    dim = (cfg.platform_width, cfg.platform_width, platform_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.5 * platform_height)
    platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(platform)

    return meshes_list, origin
