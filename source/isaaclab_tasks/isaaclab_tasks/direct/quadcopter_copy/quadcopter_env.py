# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Quadcopter winding-corridor direct environment."""

from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers  # isort: skip
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse

from . import observations as obs_module
from . import rewards as rew_module
from . import scene_builder
from . import terminations as term_module
from .env_cfg import QuadcopterWindingCorridorEnvCfg
from .metrics_logger import MetricsLogger


def _apply_curriculum_settings(cfg: QuadcopterWindingCorridorEnvCfg) -> None:
    """Map the chosen curriculum stage onto concrete task parameters."""

    stage = int(max(1, min(3, cfg.curriculum_stage)))
    cfg.active_goal_reach_threshold = float(cfg.goal_reach_threshold)
    cfg.active_wind_enabled = bool(cfg.wind_enabled)
    cfg.active_wind_mean = float(cfg.wind_mean)
    cfg.active_wind_variance = float(cfg.wind_variance)

    if stage == 1:
        cfg.active_pillar_count = min(int(cfg.num_pillars), 16)
        cfg.clear_corridor_half_width = 1.20
        cfg.active_goal_reach_threshold = max(float(cfg.goal_reach_threshold), 0.45)
        cfg.active_wind_enabled = False
        cfg.active_wind_mean = 0.0
        cfg.active_wind_variance = 0.0
    elif stage == 2:
        cfg.active_pillar_count = min(int(cfg.num_pillars), 36)
        cfg.clear_corridor_half_width = 0.75
        cfg.active_goal_reach_threshold = max(float(cfg.goal_reach_threshold), 0.40)
        cfg.active_wind_enabled = bool(cfg.wind_enabled)
        cfg.active_wind_mean = min(float(cfg.wind_mean), 0.15)
        cfg.active_wind_variance = min(float(cfg.wind_variance), 0.08)
    else:
        cfg.active_pillar_count = int(cfg.num_pillars)
        cfg.clear_corridor_half_width = 0.45


class QuadcopterEnvWindow(BaseEnvWindow):
    """Minimal window manager for the quadcopter environment."""

    def __init__(self, env: "QuadcopterWindingCorridorEnv", window_name: str = "IsaacLab"):
        super().__init__(env, window_name)


class QuadcopterWindingCorridorEnv(DirectRLEnv):
    """Quadcopter navigating a winding corridor with obstacle and wind disturbances."""

    cfg: QuadcopterWindingCorridorEnvCfg

    def __init__(self, cfg: QuadcopterWindingCorridorEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.ui_window_class_type = QuadcopterEnvWindow
        cfg.observation_space = 24
        cfg.state_space = cfg.critic_observation_space if cfg.use_privileged_critic else 0
        _apply_curriculum_settings(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._wind_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._wind_update_counter = torch.zeros(self.num_envs, device=self.device)
        self._goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._waypoint_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_goal_dist = torch.zeros(self.num_envs, device=self.device)
        self._prev_ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device)

        self._term_out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_hit_pillar = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_wall_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_ground_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_flying_over_pillars = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_excessive_tilt = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "goal_progress",
                "goal_proximity",
                "upright",
                "height",
                "ang_vel",
                "action_smoothness",
                "height_violation",
                "collision",
                "success_bonus",
            ]
        }

        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self._body_rate_scale = torch.tensor(self.cfg.body_rate_scale, device=self.device)
        self._rate_kp = torch.tensor(self.cfg.rate_kp, device=self.device)
        self._rate_kd = torch.tensor(self.cfg.rate_kd, device=self.device)
        self._moment_limit = torch.tensor(self.cfg.moment_limit, device=self.device)
        self._sector_angles = torch.linspace(
            -torch.pi,
            torch.pi,
            steps=self.cfg.obstacle_sector_count + 1,
            device=self.device,
        )[:-1]
        self._wind_steps_per_update = max(
            1, int(self.cfg.wind_update_interval / (self.cfg.sim.dt * self.cfg.decimation))
        )

        log_dir = getattr(self.cfg, "log_dir", "logs/metrics")
        self._metrics_logger = MetricsLogger(log_dir=log_dir)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        scene_builder.setup_scene(self)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Map 4D CTBR actions to collective thrust and body-rate control moments."""

        self._previous_actions.copy_(self._actions)
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust.zero_()
        self._moment.zero_()

        hover_thrust = self._robot_weight
        max_thrust = float(self.cfg.thrust_to_weight) * self._robot_weight
        thrust_action = self._actions[:, 0]
        thrust = torch.where(
            thrust_action < 0.0,
            hover_thrust * (thrust_action + 1.0),
            hover_thrust + (max_thrust - hover_thrust) * thrust_action,
        )
        self._thrust[:, 0, 2] = thrust

        current_ang_vel_b = self._robot.data.root_ang_vel_b
        ang_acc_b = (current_ang_vel_b - self._prev_ang_vel_b) / max(self.step_dt, 1.0e-6)
        target_body_rates = self._actions[:, 1:] * self._body_rate_scale
        commanded_moment = self._rate_kp * (target_body_rates - current_ang_vel_b) - self._rate_kd * ang_acc_b
        self._moment[:, 0, :] = torch.clamp(commanded_moment, min=-self._moment_limit, max=self._moment_limit)
        self._prev_ang_vel_b.copy_(current_ang_vel_b)

    def _apply_action(self):
        """Apply thrust, body-rate moments and world-frame wind to the robot."""

        if self.cfg.active_wind_enabled:
            self._update_wind()

        if self.cfg.wind_is_global:
            thrust_world = quat_rotate(self._robot.data.root_quat_w, self._thrust[:, 0, :]).unsqueeze(1)
            moment_world = quat_rotate(self._robot.data.root_quat_w, self._moment[:, 0, :]).unsqueeze(1)
            total_force_world = thrust_world + self._wind_force
            self._robot.set_external_force_and_torque(
                total_force_world,
                moment_world,
                body_ids=self._body_id,
                is_global=True,
            )
        else:
            wind_body = quat_rotate_inverse(self._robot.data.root_quat_w, self._wind_force[:, 0, :]).unsqueeze(1)
            self._robot.set_external_force_and_torque(
                self._thrust + wind_body,
                self._moment,
                body_ids=self._body_id,
                is_global=False,
            )

    def _update_wind(self):
        """Randomize a horizontal gust field in the world frame."""

        self._wind_update_counter += 1
        update_mask = self._wind_update_counter >= self._wind_steps_per_update
        if not torch.any(update_mask):
            return

        update_count = int(update_mask.sum().item())
        wind = torch.zeros(update_count, 3, device=self.device)
        direction_xy = torch.randn(update_count, 2, device=self.device)
        direction_xy = direction_xy / torch.linalg.norm(direction_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
        magnitude = torch.randn(update_count, device=self.device) * float(self.cfg.active_wind_variance)
        magnitude += float(self.cfg.active_wind_mean)
        magnitude = torch.clamp(magnitude, min=0.0)
        wind[:, :2] = direction_xy * magnitude.unsqueeze(-1)
        if not self.cfg.wind_xy_only:
            wind[:, 2] = 0.25 * magnitude

        self._wind_force[update_mask, 0, :] = wind
        self._wind_update_counter[update_mask] = 0.0

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return obs_module.get_observations(self)

    def _get_states(self) -> torch.Tensor | None:
        if not self.cfg.use_privileged_critic:
            return None
        policy_obs = obs_module.compute_policy_observation(self)
        return obs_module.compute_critic_observation(self, policy_obs)

    def _get_rewards(self) -> torch.Tensor:
        return rew_module.get_rewards(self)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return term_module.get_dones(self)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._log_episode_stats(env_ids)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._reset_goal(env_ids)
        self._reset_obstacle_layout(env_ids)
        self._reset_robot_state(env_ids)
        self._reset_episode_buffers(env_ids)
        self._prev_goal_dist[env_ids] = torch.linalg.norm(
            self._goal_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids],
            dim=1,
        )

    def _reset_goal(self, env_ids: torch.Tensor):
        """Place the goal slightly before the front wall."""

        env_origins = self._terrain.env_origins
        self._goal_pos_w[env_ids, 0] = env_origins[env_ids, 0] + float(self.cfg.field_length) - float(
            self.cfg.goal_margin_x
        )
        self._goal_pos_w[env_ids, 1] = env_origins[env_ids, 1] + float(self.cfg.goal_y_position)
        self._goal_pos_w[env_ids, 2] = float(self.cfg.flight_z)

    def _reset_obstacle_layout(self, env_ids: torch.Tensor):
        """Sample a fresh pillar layout for the reset environments."""

        scene_builder.reset_pillar_layouts(self, env_ids)

    def _reset_robot_state(self, env_ids: torch.Tensor):
        """Reset the robot near the start of the corridor."""

        env_origins = self._terrain.env_origins
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]

        noise = float(self.cfg.reset_pos_noise_xy)
        xy_noise = torch.empty((len(env_ids), 2), device=self.device).uniform_(-noise, noise)
        default_root_state[:, 0] = env_origins[env_ids, 0] + 0.5 + xy_noise[:, 0]
        default_root_state[:, 1] = env_origins[env_ids, 1] + float(self.cfg.goal_y_position) + xy_noise[:, 1]

        height_low = max(0.3, float(self.cfg.flight_z) - float(self.cfg.reset_height_noise))
        height_high = min(float(self.cfg.max_height_pillar_stage), float(self.cfg.flight_z) + float(self.cfg.reset_height_noise))
        default_root_state[:, 2] = torch.empty(len(env_ids), device=self.device).uniform_(height_low, height_high)

        yaw = torch.empty(len(env_ids), device=self.device).uniform_(
            -float(self.cfg.reset_yaw_noise_rad),
            float(self.cfg.reset_yaw_noise_rad),
        )
        default_root_state[:, 3] = torch.cos(0.5 * yaw)
        default_root_state[:, 4] = 0.0
        default_root_state[:, 5] = 0.0
        default_root_state[:, 6] = torch.sin(0.5 * yaw)
        default_root_state[:, 7:13] = 0.0

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_episode_buffers(self, env_ids: torch.Tensor):
        """Clear per-episode buffers for the reset environments."""

        self._goal_reached[env_ids] = False
        self._term_out_of_bounds[env_ids] = False
        self._term_success[env_ids] = False
        self._term_hit_pillar[env_ids] = False
        self._term_wall_contact[env_ids] = False
        self._term_ground_contact[env_ids] = False
        self._term_flying_over_pillars[env_ids] = False
        self._term_excessive_tilt[env_ids] = False
        self._wind_force[env_ids] = 0.0
        self._wind_update_counter[env_ids] = 0.0
        self._prev_ang_vel_b[env_ids] = 0.0

    def _log_episode_stats(self, env_ids: torch.Tensor):
        """Aggregate episode metrics and write them to extras and CSV."""

        env_origins = self._terrain.env_origins
        extras: dict[str, float] = {}

        for key in self._episode_sums:
            episodic_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = episodic_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = extras

        pos_local = self._robot.data.root_pos_w[env_ids] - env_origins[env_ids]
        collision_mask = (
            self._term_hit_pillar[env_ids] | self._term_wall_contact[env_ids] | self._term_ground_contact[env_ids]
        )
        action_change = torch.linalg.norm(self._actions[env_ids] - self._previous_actions[env_ids], dim=1)

        term_extras = {
            "Episode_Termination/time_out": torch.count_nonzero(self.reset_time_outs[env_ids]).item(),
            "Episode_Termination/success": torch.count_nonzero(self._term_success[env_ids]).item(),
            "Episode_Termination/hit_pillar": torch.count_nonzero(self._term_hit_pillar[env_ids]).item(),
            "Episode_Termination/wall_contact": torch.count_nonzero(self._term_wall_contact[env_ids]).item(),
            "Episode_Termination/ground_contact": torch.count_nonzero(self._term_ground_contact[env_ids]).item(),
            "Episode_Termination/flying_over_pillars": torch.count_nonzero(
                self._term_flying_over_pillars[env_ids]
            ).item(),
            "Episode_Termination/excessive_tilt": torch.count_nonzero(self._term_excessive_tilt[env_ids]).item(),
            "Episode_Termination/out_of_bounds": torch.count_nonzero(self._term_out_of_bounds[env_ids]).item(),
            "Metrics/final_distance_to_goal": torch.linalg.norm(
                self._goal_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids],
                dim=1,
            ).mean().item(),
            "Metrics/avg_height": pos_local[:, 2].mean().item(),
            "Metrics/avg_x_progress": pos_local[:, 0].mean().item(),
            "Metrics/avg_y_deviation": torch.abs(pos_local[:, 1]).mean().item(),
            "Metrics/avg_lin_vel": torch.linalg.norm(self._robot.data.root_lin_vel_w[env_ids], dim=1).mean().item(),
            "Metrics/avg_ang_vel": torch.linalg.norm(self._robot.data.root_ang_vel_w[env_ids], dim=1).mean().item(),
            "Metrics/avg_upright": ((self._robot.data.projected_gravity_b[env_ids, 2] + 1.0) / 2.0).mean().item(),
            "Metrics/avg_action_change": action_change.mean().item(),
        }
        self.extras["log"].update(term_extras)

        self._metrics_logger.log(
            step=self.common_step_counter,
            success_count=int(torch.count_nonzero(self._term_success[env_ids]).item()),
            collision_count=int(torch.count_nonzero(collision_mask).item()),
            num_episodes=len(env_ids),
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._goal_pos_w)

    def close(self):
        self._metrics_logger.close()
        super().close()
