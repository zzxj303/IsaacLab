# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""CSV metrics logger for the quadcopter winding-corridor environment."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


class MetricsLogger:
    """Persistent CSV logger for success and collision metrics."""

    def __init__(self, log_dir: str = "logs/metrics") -> None:
        metrics_dir = Path(log_dir) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = metrics_dir / f"success_collision_metrics_{timestamp}.csv"

        self._file = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "timestep",
                "success_count",
                "collision_count",
                "success_rate",
                "collision_rate",
                "total_episodes",
            ]
        )
        self._file.flush()

        self._cumulative_success = 0
        self._cumulative_collisions = 0
        self._cumulative_episodes = 0

        print(f"[INFO] MetricsLogger: writing to {csv_path}")

    def log(
        self,
        step: int,
        success_count: int,
        collision_count: int,
        num_episodes: int,
    ) -> None:
        """Append one aggregated row to the CSV file."""

        self._cumulative_success += success_count
        self._cumulative_collisions += collision_count
        self._cumulative_episodes += num_episodes

        total_episodes = self._cumulative_episodes
        success_rate = self._cumulative_success / total_episodes * 100 if total_episodes > 0 else 0.0
        collision_rate = self._cumulative_collisions / total_episodes * 100 if total_episodes > 0 else 0.0

        self._writer.writerow(
            [
                step,
                success_count,
                collision_count,
                f"{success_rate:.2f}",
                f"{collision_rate:.2f}",
                num_episodes,
            ]
        )
        self._file.flush()

    def close(self) -> None:
        """Flush and close the underlying CSV file."""

        try:
            self._file.close()
            print("[INFO] MetricsLogger: CSV file closed.")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] MetricsLogger: error closing file: {exc}")
