#!/usr/bin/env python3

"""Plot training metrics for quadcopter_forest_migrated from TensorBoard event logs.

Usage examples:
  python scripts/plot_quadcopter_forest_migrated_metrics.py --step 20000
  python scripts/plot_quadcopter_forest_migrated_metrics.py --run-dir logs/rsl_rl/quadcopter_forest_pose_direct_migrated/2026-04-03_16-13-09 --step 20000
  python scripts/plot_quadcopter_forest_migrated_metrics.py --event-file logs/rsl_rl/.../events.out.tfevents... --step 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


DEFAULT_EXPERIMENT_DIR = Path("logs/rsl_rl/quadcopter_forest_pose_direct_migrated")


def find_latest_event_file(experiment_dir: Path) -> tuple[Path, Path]:
    """Return (run_dir, latest_event_file) from an experiment directory."""
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    run_dirs = [p for p in experiment_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {experiment_dir}")

    latest_run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
    event_files = sorted(latest_run_dir.glob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {latest_run_dir}")

    return latest_run_dir, event_files[-1]


def find_latest_event_in_run(run_dir: Path) -> Path:
    """Return latest event file in a specific run directory."""
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    event_files = sorted(run_dir.glob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {run_dir}")
    return event_files[-1]


def load_scalars(event_file: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all scalar tags from an event file."""
    accumulator = event_accumulator.EventAccumulator(str(event_file))
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])

    data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag in tags:
        events = accumulator.Scalars(tag)
        steps = np.asarray([e.step for e in events], dtype=np.int64)
        values = np.asarray([e.value for e in events], dtype=np.float64)
        data[tag] = (steps, values)
    return data


def _filter_by_step(steps: np.ndarray, values: np.ndarray, step_max: int) -> tuple[np.ndarray, np.ndarray]:
    mask = steps <= step_max
    return steps[mask], values[mask]


def plot_series(ax, data: dict[str, tuple[np.ndarray, np.ndarray]], tags: list[str], title: str, step_max: int):
    """Plot one panel with one or multiple scalar tags in [0, step_max]."""
    plotted = False
    for tag in tags:
        if tag not in data:
            continue
        steps, values = data[tag]
        steps, values = _filter_by_step(steps, values, step_max)
        if steps.size == 0:
            continue
        plotted = True
        ax.plot(steps, values, linewidth=2.0, label=tag)

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_xlim(left=0, right=step_max)
    ax.grid(True, alpha=0.35)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)


def make_summary_figure(data: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path, run_name: str, step_max: int):
    """Create and save a summary figure for main training metrics in [0, step_max]."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

    panel_defs = [
        (["Train/mean_reward"], "Train Mean Reward"),
        (["Train/mean_episode_length"], "Train Mean Episode Length"),
        (
            [
                "Metrics/final_distance_to_goal",
                "Metrics/final_distance_to_goal_10_percentile",
                "Metrics/final_distance_to_goal_90_percentile",
            ],
            "Final Distance to Goal",
        ),
        (["Episode_Termination/died", "Episode_Termination/time_out"], "Episode Terminations"),
        (["Loss/surrogate", "Loss/value_function", "Loss/entropy"], "PPO Losses"),
        (
            [
                "Episode_Reward/progress_to_goal",
                "Episode_Reward/distance_to_goal",
                "Episode_Reward/undesired_contacts",
                "Episode_Reward/terminated",
            ],
            "Reward Components",
        ),
    ]

    for ax, (tags, title) in zip(axes.flatten(), panel_defs):
        plot_series(ax, data, tags, title, step_max=step_max)

    fig.suptitle(f"Quadcopter Forest Migrated - Training Metrics ({run_name}) [0, {step_max}]", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot quadcopter_forest_migrated training metrics in [0, step].")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Experiment directory that contains run subdirectories.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific run directory (if omitted, latest run under --experiment-dir is used).",
    )
    parser.add_argument("--event-file", type=Path, default=None, help="Specific event file to read.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: <run_dir>/metrics_summary.png",
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Plot only data from step 0 to this step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.step < 0:
        raise ValueError("--step must be >= 0")

    if args.event_file is not None:
        event_file = args.event_file
        if not event_file.exists():
            raise FileNotFoundError(f"Event file not found: {event_file}")
        run_dir = event_file.parent
    elif args.run_dir is not None:
        run_dir = args.run_dir
        event_file = find_latest_event_in_run(run_dir)
    else:
        run_dir, event_file = find_latest_event_file(args.experiment_dir)

    data = load_scalars(event_file)
    if not data:
        raise RuntimeError(f"No scalar data found in: {event_file}")

    output_path = args.output if args.output is not None else run_dir / "metrics_summary.png"
    make_summary_figure(data=data, out_path=output_path, run_name=run_dir.name, step_max=args.step)

    available_tags = sorted(data.keys())
    print(f"Loaded event file: {event_file}")
    print(f"Found scalar tags: {len(available_tags)}")
    print(f"Plot range: [0, {args.step}]")
    print(f"Saved summary image: {output_path}")


if __name__ == "__main__":
    main()
