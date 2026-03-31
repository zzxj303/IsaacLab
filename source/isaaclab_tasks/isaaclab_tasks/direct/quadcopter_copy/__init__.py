# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quadcopter winding-corridor (obstacle forest) environment.

Module layout
-------------
* :mod:`.env_cfg`        — :class:`.QuadcopterWindingCorridorEnvCfg`
* :mod:`.quadcopter_env` — :class:`.QuadcopterWindingCorridorEnv` (main class)
* :mod:`.scene_builder`  — pillar / boundary wall construction
* :mod:`.observations`   — observation computation
* :mod:`.rewards`        — reward functions
* :mod:`.terminations`   — termination conditions
* :mod:`.metrics_logger` — CSV metrics logging
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Quadcopter-WindingCorridor-Direct-v0",
    entry_point=f"{__name__}.quadcopter_env:QuadcopterWindingCorridorEnv",
    disable_env_checker=True,
    kwargs={
        # Config class is defined in env_cfg.py
        "env_cfg_entry_point": f"{__name__}.env_cfg:QuadcopterWindingCorridorEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterWindingCorridorPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
