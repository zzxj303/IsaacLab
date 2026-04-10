# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the quadcopter winding-corridor task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip


class QuadcopterEnvWindow:
    """Placeholder imported by the environment to avoid a circular import."""

    pass


@configclass
class QuadcopterWindingCorridorEnvCfg(DirectRLEnvCfg):
    """Configuration for the direct quadcopter obstacle-avoidance task."""

    # -------------------------------------------------------------------------
    # Core env
    # -------------------------------------------------------------------------
    episode_length_s: float = 10.0
    decimation: int = 2
    action_space: int = 4
    observation_space: int = 24
    state_space: int = 0
    debug_vis: bool = True
    ui_window_class_type = None

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # -------------------------------------------------------------------------
    # Scene
    # -------------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=16.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # -------------------------------------------------------------------------
    # Robot and low-level control
    # -------------------------------------------------------------------------
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=CRAZYFLIE_CFG.spawn.replace(
            scale=(1.8, 1.8, 1.8),
        ),
    )
    thrust_to_weight: float = 2.2
    body_rate_scale: tuple[float, float, float] = (3.5, 3.5, 1.8)
    rate_kp: tuple[float, float, float] = (0.02, 0.02, 0.01)
    rate_kd: tuple[float, float, float] = (0.004, 0.004, 0.002)
    moment_limit: tuple[float, float, float] = (0.08, 0.08, 0.04)

    # -------------------------------------------------------------------------
    # Reward scales
    # -------------------------------------------------------------------------
    goal_progress_reward_scale: float = 15.0
    goal_proximity_reward_scale: float = 0.5
    success_bonus: float = 20.0
    upright_reward_scale: float = 4.0
    height_reward_scale: float = 0.75
    ang_vel_penalty_scale: float = -0.04
    action_smoothness_penalty_scale: float = -0.03
    height_violation_penalty_scale: float = -4.0
    collision_penalty: float = -5.0

    # -------------------------------------------------------------------------
    # Course and goal
    # -------------------------------------------------------------------------
    flight_z: float = 1.0
    waypoint_tanh_std: float = 0.60
    goal_reach_threshold: float = 0.35
    goal_margin_x: float = 0.5
    goal_y_position: float = 0.0

    # -------------------------------------------------------------------------
    # Field dimensions
    # -------------------------------------------------------------------------
    field_length: float = 12.0
    field_width: float = 6.0

    # -------------------------------------------------------------------------
    # Obstacles and local obstacle observations
    # -------------------------------------------------------------------------
    num_pillars: int = 60
    pillar_radius: float = 0.08
    pillar_height: float = 2.0
    pillar_min_spacing: float = 0.45
    pillar_collision_margin: float = 0.05
    start_zone_length: float = 1.5
    goal_zone_length: float = 1.5
    clear_corridor_half_width: float = 0.45
    active_pillar_count: int = 60
    inactive_pillar_drop_z: float = -5.0
    layout_batch_size: int = 32
    resample_pillars_on_reset: bool = False
    obstacle_sector_count: int = 8
    obstacle_max_range: float = 4.0

    # -------------------------------------------------------------------------
    # Boundary walls
    # -------------------------------------------------------------------------
    boundary_wall_enabled: bool = True
    boundary_wall_thickness: float = 0.1
    boundary_wall_height: float = 2.5

    # -------------------------------------------------------------------------
    # Safety limits and terminations
    # -------------------------------------------------------------------------
    max_height_pillar_stage: float = 2.3
    height_violation_margin: float = 0.2
    max_flight_height: float = 2.5
    max_tilt_rad: float = 1.22
    ground_contact_height: float = 0.16
    wall_contact_margin: float = 0.18
    collision_force_threshold: float = 0.5
    termination_grace_steps: int = 20

    # -------------------------------------------------------------------------
    # Reset noise
    # -------------------------------------------------------------------------
    reset_pos_noise_xy: float = 0.05
    reset_yaw_noise_rad: float = 0.35
    reset_height_noise: float = 0.2

    # -------------------------------------------------------------------------
    # Wind disturbance
    # -------------------------------------------------------------------------
    wind_enabled: bool = False
    wind_is_global: bool = False
    wind_xy_only: bool = False
    wind_mean: float = 0.3
    wind_variance: float = 0.15
    wind_update_interval: float = 0.2
    active_wind_enabled: bool = False
    active_wind_mean: float = 0.3
    active_wind_variance: float = 0.15

    # -------------------------------------------------------------------------
    # Curriculum and asymmetric critic
    # -------------------------------------------------------------------------
    curriculum_stage: int = 3
    active_goal_reach_threshold: float = 0.35
    use_privileged_critic: bool = False
    critic_observation_space: int = 36
