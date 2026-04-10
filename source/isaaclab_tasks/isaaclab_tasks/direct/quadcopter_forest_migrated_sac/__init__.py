# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Migrated quadcopter forest environment (IsaacLab 5.1 compatible)."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Experimental-Quadcopter-Forest-Pose-Direct-Migrated-SAC-v0",
    entry_point=f"{__name__}.quadcopter_forest_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_forest_env:QuadcopterEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Experimental-Quadcopter-Forest-Pose-Direct-Migrated-SAC-Play-v0",
    entry_point=f"{__name__}.quadcopter_forest_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_forest_env:QuadcopterEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
