# Quadcopter Forest Migrated Task (IsaacLab 5.1 Compatible)

This document summarizes the full technical design of the migrated direct-RL quadcopter forest task implemented in this folder.

## 1. Task Scope and Registration

This package defines two Gym tasks:

- `Isaac-Experimental-Quadcopter-Forest-Pose-Direct-Migrated-v0`
- `Isaac-Experimental-Quadcopter-Forest-Pose-Direct-Migrated-Play-v0`

Both are registered in `__init__.py` and point to the same environment class (`QuadcopterForestEnv`) with different config classes:

- Training config: `QuadcopterForestEnvCfg`
- Play config: `QuadcopterForestEnvCfgPlay`

The default RL runner config points to:

- `agents/rsl_rl_ppo_cfg.py:QuadcopterForestPPORunnerCfg`

## 2. Folder Structure and Responsibilities

- `__init__.py`
  - Registers Gym task IDs and entry points.
- `quadcopter_forest_env.py`
  - Main environment class (`QuadcopterForestEnv`) and task configs.
  - Defines dynamics control, observation assembly, rewards, reset logic, done conditions, debug visualization.
- `agents/rsl_rl_ppo_cfg.py`
  - RSL-RL PPO runner and network/optimizer hyperparameters.
- `terrains/config/quadcopter_rough.py`
  - Forest terrain generator factory and object density mapping.
- `terrains/trimesh/mesh_terrains_cfg.py`
  - Custom trimesh subterrain config classes.
- `terrains/trimesh/mesh_terrains.py`
  - Custom repeated-object mesh terrain generation implementation.

## 3. Environment Configuration (Training Variant)

Main config class: `QuadcopterForestEnvCfg`.

### 3.1 Timing and rollout setup

- `episode_length_s = 7.5`
- `dt = 1 / 50 = 0.02 s`
- `decimation = 2`
- Effective RL step time: `step_dt = dt * decimation = 0.04 s`
- Approx max RL steps per episode: `7.5 / 0.04 = 187.5` (buffer-based limit in framework)

### 3.2 Spaces

- Action dim: `4`
- Observation dim: `59`
- State dim: `0`

### 3.3 Parallel environments and scene

- `num_envs = 4096`
- `env_spacing = 2.5`
- Physics replicated per env.

### 3.4 Robot and sensors

- Robot asset: Crazyflie (`CRAZYFLIE_CFG`) with contact sensors activated.
- Contact sensor:
  - Prim regex over robot links.
  - History length: `2`.
- Lidar-like ray caster (`simple_lidar`):
  - Mounted on robot body.
  - Horizontal FOV: `[-90, 90]` degrees.
  - Horizontal resolution: `3.99` degrees.
  - Channels: `1` (single scan plane).
  - Max distance: `15`.
  - Rays collide against `"/World/ground"` terrain mesh.

### 3.5 Terrain

The terrain uses a generator (`TerrainImporterCfg`) with custom subterrain configs from `QUADCOPTER_ROUGH_TERRAINS_CFG_FACTORY(...)`.

Default training terrain parameters:

- Terrain tile size: `25 x 25`
- Generator grid: `num_rows=15`, `num_cols=15`
- Subterrain border width (inside tile): `2.0`
- Central platform width: `0.5`
- Object density range parameters:
  - `objects_density_min = 0.17`
  - `objects_density_max = 0.7`

Object counts are computed from occupied area:

- `occupied_area = (size - 2*border)^2 - platform^2`
- `num_objects_min = int(density_min * occupied_area)`
- `num_objects_max = int(density_max * occupied_area)`

With size `25`, this yields approximately:

- occupied area: `440.75`
- min objects: `74`
- max objects: `308`

Subterrain mixture:

- `repeated_cylinders`: proportion `0.975`
- `empty` plane: proportion `0.025`

The repeated cylinders interpolate with difficulty from sparse/easy to dense/harder:

- Start: `height=10.0`, `radius=0.3`, `max_yx_angle=0.0`, `num_objects=min`
- End: `height=12.0`, `radius=0.06`, `max_yx_angle=30.0`, `num_objects=max`

A central platform is always carved/reserved so spawn/goal region has a guaranteed gap.

## 4. Action Processing and Applied Forces

Input action per env: `a in R^4`, clamped to `[-2, 2]`.

Mapping:

- Thrust command uses only `a[0]`:
  - `thrust_z = thrust_to_weight * robot_weight * (thrust_scale * a0 + 1) / 2`
- Body moments use `a[1:4]`:
  - `moment_xyz = moment_scale * a[1:4]`

Default scales:

- `thrust_to_weight = 1.9`
- `thrust_scale = 0.75`
- `moment_scale = 0.01`

The final applied external force is:

- `total_force = thrust_force + wind_force`

## 5. Wind Model

Wind is enabled by default and acts in XY (no direct Z wind term).

Per episode/environment:

1. Sample a random XY direction uniformly in `[0, 2*pi)`.
2. Sample background speed in `[0.0, 2.0]`.
3. Initialize gust delta speed to `0`.

During simulation:

- A per-env countdown (`_wind_steps_until_update`) triggers gust updates.
- Update interval is resampled from `[0.2, 1.0]` seconds.
- At update:
  - Add delta sampled from `[-0.5, 0.5]`.
  - Clamp gust delta to `[-1.0, 1.0]`.
- Effective speed:
  - `wind_speed = clamp(background + gust_delta, min=0)`
- Force magnitude:
  - `wind_force_mag = robot_weight * wind_force_ratio_per_speed * wind_speed`
- Default ratio:
  - `wind_force_ratio_per_speed = 0.03`

## 6. Observation Vector (59-D)

Observation is policy-only and concatenates:

1. Root COM linear velocity in body frame: `3`
2. Root COM angular velocity in body frame: `3`
3. Projected gravity in body frame: `3`
4. Height scalar (normalized): `1`
5. Goal position in body frame (clipped/normalized): `3`
6. Lidar ray distances (scaled/clipped): remaining dims

Total = `59`.

Noise is injected directly in observation assembly:

- Goal position noise: uniform `[-0.02, 0.02]`
- Height noise: uniform `[-0.02, 0.02]`
- Lidar distance noise: uniform `[-0.02, 0.02]`
- Linear velocity noise: uniform `[-0.2, 0.2]`
- Angular velocity noise: uniform `[-0.1, 0.1]`
- Projected gravity noise: uniform `[-0.03, 0.03]`

Normalization/clipping highlights:

- Goal body-frame vector scaled by `desired_pos_b_obs_clip = 4.0` and clipped to `[-1, 1]`.
- Height divided by upper height limit (`4.5`).
- Lidar distances scaled by `1/6` and clipped to `[-1, 1]`.

## 7. Goal Sampling and Spawn Logic

### 7.1 Goal XY sampling with central hole

Goals are sampled from a square region excluding a central square hole (implemented by `_sample_points_square_hole`).

- Outer half-width: `desired_pos_b_xy_limits[1]`
- Inner half-width: `desired_pos_b_xy_limits[0]`

Default values (training, size=25):

- inner half-width: `11.5`
- outer half-width: `12.5`

This creates a thin ring-like valid goal region around the center exclusion zone.

### 7.2 Goal Z sampling

- Goal height sampled in `[2.0, 3.0]`.

### 7.3 Robot spawn

- XY spawn around terrain origin.
- Z spawn randomized within a center band of allowed height range.
- Yaw randomized uniformly (via `random_yaw_orientation`).

### 7.4 Optional random respawn terrain level

If `random_respawn=True`, env origins may move up/down terrain levels on reset. Disabled in training cfg, enabled in play cfg.

## 8. Reward Design

The reward is a weighted sum of terms (each multiplied by `step_dt`).

Let `r = sum_i r_i`.

### 8.1 Penalty terms

- `lin_vel`: penalizes processed linear velocity magnitude.
  - Scale: `-0.018`
  - Additional directional shaping on x/y is applied before norm.
- `ang_vel`: penalizes angular velocity.
  - Scale: `-0.06`
- `ang_vel_final`: extra angular velocity penalty near goal.
  - Active when distance to goal `< 0.3`.
  - Scale: `-0.2`
- `actions`: penalizes action magnitude (`L1`).
  - Scale: `-0.2`
- `undesired_contacts`: penalizes contact events from contact sensor history.
  - Scale: `-4.0`
- `flat_orientation`: penalizes non-flat attitude via projected gravity XY.
  - Scale: `-1.0`
- `obstacle_proximity`: penalizes lidar distances below threshold.
  - Threshold: `0.4`
  - Scale: `-6.0`
- `height_bounds_proximity`: penalizes proximity beyond soft height limits.
  - Soft margin from bounds: `0.3`
  - Scale: `-4.0`
- `terminated`: terminal penalty when reset due to termination.
  - Scale: `-200.0`

### 8.2 Goal progress and distance terms

- `progress_to_goal`:
  - Uses current vs previous distance delta.
  - Mapped by `-tanh((d_t - d_{t-1}) / sigma^2)`.
  - `sigma = sqrt(0.1)`.
  - Scale: `+2.0`.
- `distance_to_goal`:
  - `1 - tanh(d / sigma^2)`, `sigma = sqrt(1.25)`.
  - Scale: `+1.0`.
- `distance_to_goal_fine`:
  - Similar with tighter `sigma = sqrt(0.3)`.
  - Scale: `+0.7`.

Distance-based rewards are gated by a clipped-goal-window condition so they activate in a bounded region around the agent.

### 8.3 Logging

Every component has episodic accumulators in `_episode_sums`, exported at reset as:

- `Episode_Reward/<term>`

Additional reset metrics:

- `Metrics/final_distance_to_goal`
- `Metrics/final_distance_to_goal_10_percentile`
- `Metrics/final_distance_to_goal_90_percentile`
- `Metrics/final_ang_vel_goal_10_percentile`
- `Episode_Termination/died`
- `Episode_Termination/time_out`

## 9. Done Conditions

Per env done tuple: `(died, time_out)`.

- `time_out`: episode length reached.
- `died` if either:
  - Height out of hard bounds `[0.5, 4.5]`, or
  - Any contact norm on undesired contact bodies exceeds threshold (`0.0001` in done check).

## 10. Play Config Differences

`QuadcopterForestEnvCfgPlay` modifies training cfg:

- `episode_length_s *= 4`
- `size_terrain *= 4` (larger area)
- `objects_density_max *= 0.8` (slightly reduced top density)
- Rebuild terrain generator with `num_rows=5`, `num_cols=5`
- `random_respawn = True`
- `avoid_reset_spikes_in_training = False`

This is aimed at interactive/visual evaluation rather than stable PPO training throughput.

## 11. PPO Runner Config

Defined in `agents/rsl_rl_ppo_cfg.py`.

Runner:

- `num_steps_per_env = 50`
- `max_iterations = 100000`
- `save_interval = 500`
- `experiment_name = quadcopter_forest_pose_direct_migrated`
- `empirical_normalization = False`

Policy network:

- Actor hidden dims: `[128, 64]`
- Critic hidden dims: `[128, 64]`
- Activation: `elu`
- Init noise std: `1.0`

PPO algorithm:

- `value_loss_coef = 1.0`
- `use_clipped_value_loss = True`
- `clip_param = 0.2`
- `entropy_coef = 0.01`
- `num_learning_epochs = 5`
- `num_mini_batches = 4`
- `learning_rate = 1e-3`
- `schedule = adaptive`
- `gamma = 0.99`
- `lam = 0.95`
- `desired_kl = 0.01`
- `max_grad_norm = 1.0`

## 12. UI and Debug Visualization

- Custom window class: `QuadcopterForestEnvWindow`
- Debug visualization toggle controlled by cfg `debug_vis`.
- Goal marker rendered as a small cuboid at current desired position.

Viewer defaults are configured, and scene setup uses third-person follow-camera settings tied to robot root.

## 13. Related Runtime Scripts (Outside This Folder)

While not defined in this folder, the repository includes helper scripts:

- `scripts/train_quadcopter_forest_migrated.sh`
- `scripts/play_quadcopter_forest_migrated.sh`
- `scripts/plot_quadcopter_forest_migrated_metrics.py`

Notable detail in training script:

- It prepends source package paths via `PYTHONPATH` to avoid importing a mismatched installed package variant that can change lidar discretization and thus observation dimensions.

## 14. Practical Notes and Tuning Handles

High-impact knobs in this task:

- Difficulty and clutter:
  - `size_terrain`, `objects_density_min/max`, terrain grid (`num_rows/num_cols`).
- Flight aggressiveness/stability:
  - `thrust_to_weight`, `thrust_scale`, `moment_scale`.
- Disturbance profile:
  - wind ranges and update intervals.
- Safety corridor:
  - `height_w_limits`, `threshold_obstacle_proximity`, `threshold_height_bounds_proximity`.
- Goal convergence behavior:
  - progress and distance std terms (`progress_to_goal_std`, `distance_to_goal_std`, `distance_to_goal_fine_std`).
- Throughput vs fidelity:
  - `num_envs`, `decimation`, episode length.

## 15. Quick Checklist for Reproducibility

1. Use task ID exactly as registered in this package.
2. Ensure source paths are preferred over stale installed package versions.
3. Keep observation dimension at `59` unless intentionally changing lidar/obs composition.
4. If changing terrain density/size, re-check reward balance and collision rate.
5. Track `Episode_Termination/died` and distance percentiles to diagnose training regressions quickly.

---

If you want, this README can be extended with:

- an explicit observation index table (dim-by-dim mapping),
- a reward sanity test protocol (expected value ranges for random policy),
- recommended curriculum stages for density/wind,
- and comparison notes versus the SAC variant in `quadcopter_forest_migrated_sac`.
