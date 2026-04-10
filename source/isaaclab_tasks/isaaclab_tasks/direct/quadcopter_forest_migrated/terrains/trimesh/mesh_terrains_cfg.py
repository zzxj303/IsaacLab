# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .mesh_terrains import custom_repeated_objects_terrain


@configclass
class CustomMeshRepeatedObjectsTerrainCfg(SubTerrainBaseCfg):
    @configclass
    class ObjectCfg:
        num_objects: int = MISSING
        height: float = MISSING

    function = custom_repeated_objects_terrain
    object_type: Literal["cylinder", "box", "cone"] | callable = MISSING
    object_params_start: ObjectCfg = MISSING
    object_params_end: ObjectCfg = MISSING
    max_height_noise: float = 0.0
    platform_width: float = 1.0
    platform_height: float | None = None
    border_width: float = 0.0


@configclass
class CustomMeshRepeatedPyramidsTerrainCfg(CustomMeshRepeatedObjectsTerrainCfg):
    @configclass
    class ObjectCfg(CustomMeshRepeatedObjectsTerrainCfg.ObjectCfg):
        radius: float = MISSING
        max_yx_angle: float = 0.0
        degrees: bool = True

    object_type = mesh_utils_terrains.make_cone
    object_params_start: ObjectCfg = MISSING
    object_params_end: ObjectCfg = MISSING


@configclass
class CustomMeshRepeatedBoxesTerrainCfg(CustomMeshRepeatedObjectsTerrainCfg):
    @configclass
    class ObjectCfg(CustomMeshRepeatedObjectsTerrainCfg.ObjectCfg):
        size: tuple[float, float] = MISSING
        max_yx_angle: float = 0.0
        degrees: bool = True

    object_type = mesh_utils_terrains.make_box
    object_params_start: ObjectCfg = MISSING
    object_params_end: ObjectCfg = MISSING


@configclass
class CustomMeshRepeatedCylindersTerrainCfg(CustomMeshRepeatedObjectsTerrainCfg):
    @configclass
    class ObjectCfg(CustomMeshRepeatedObjectsTerrainCfg.ObjectCfg):
        radius: float = MISSING
        max_yx_angle: float = 0.0
        degrees: bool = True

    object_type = mesh_utils_terrains.make_cylinder
    object_params_start: ObjectCfg = MISSING
    object_params_end: ObjectCfg = MISSING
