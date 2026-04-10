# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Migrated quadcopter forest environment (IsaacLab 5.1 compatible)."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Experimental-Quadcopter-Forest-Pose-Direct-Migrated-v0",
    entry_point=f"{__name__}.quadcopter_forest_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_forest_env:QuadcopterEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterForestPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Experimental-Quadcopter-Forest-Pose-Direct-Migrated-Play-v0",
    entry_point=f"{__name__}.quadcopter_forest_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_forest_env:QuadcopterEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterForestPPORunnerCfg",
    },
)
