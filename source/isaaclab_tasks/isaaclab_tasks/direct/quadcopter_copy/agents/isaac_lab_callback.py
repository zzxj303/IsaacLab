# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom skrl Callback for logging Isaac Lab environment metrics to TensorBoard.

This callback reads environment-specific metrics from extras["log"] and writes them 
to TensorBoard for better training observability.
"""

from typing import Any
import torch
from skrl.utils.callbacks import Callback


class IsaacLabLogCallback(Callback):
    """Custom callback to log Isaac Lab environment extras to TensorBoard.
    
    This callback intercepts the extras["log"] dictionary from the environment
    and logs all metrics to TensorBoard using the trainer's writer.
    
    Features:
    - Logs Episode_Reward metrics (all reward components)
    - Logs Episode_Termination counts (success, collision, timeout, etc.)
    - Logs custom Metrics (avg_height, avg_x_progress, etc.)
    - Handles tensor-to-scalar conversion automatically
    - Supports wrapped environments (accesses env.unwrapped)
    
    Example usage:
        ```python
        from isaac_lab_callback import IsaacLabLogCallback
        
        # Create environment and runner
        env = gym.make("Isaac-Quadcopter-Direct-v0", cfg=env_cfg)
        env = SkrlVecEnvWrapper(env)
        runner = Runner(env, agent_cfg)
        
        # Add callback
        callback = IsaacLabLogCallback()
        runner.agent.set_mode("train")
        runner.agent.set_running_mode("train")
        
        # Manually add callback to trainer
        runner.trainer.callbacks.append(callback)
        
        # Run training
        runner.run()
        ```
    """
    
    def __init__(self):
        """Initialize the Isaac Lab logging callback."""
        super().__init__()
        self._last_logged_step = -1
    
    def on_timestep_end(self, trainer: Any, timestep: int, timesteps: int) -> None:
        """Called at the end of each timestep.
        
        Reads the extras["log"] dictionary from the environment and logs all
        metrics to TensorBoard.
        
        Args:
            trainer: The skrl trainer instance
            timestep: Current timestep
            timesteps: Total timesteps for training
        """
        # Only log once per timestep to avoid duplicates
        if timestep == self._last_logged_step:
            return
        self._last_logged_step = timestep
        
        # Access the unwrapped environment to get extras
        # Handle both wrapped and unwrapped environments
        env = trainer.env
        if hasattr(env, "unwrapped"):
            unwrapped_env = env.unwrapped
        else:
            unwrapped_env = env
        
        # Check if extras["log"] exists and has data
        if not hasattr(unwrapped_env, "extras"):
            return
        
        extras_log = unwrapped_env.extras.get("log", {})
        if not extras_log:
            return
        
        # Log all metrics from extras["log"] to TensorBoard
        for metric_name, metric_value in extras_log.items():
            # Convert tensor to scalar if needed
            if isinstance(metric_value, torch.Tensor):
                # Handle both single values and batched tensors
                if metric_value.numel() == 1:
                    scalar_value = metric_value.item()
                else:
                    # If it's a batch, take the mean
                    scalar_value = metric_value.mean().item()
            elif isinstance(metric_value, (int, float)):
                scalar_value = float(metric_value)
            else:
                # Skip unsupported types
                continue
            
            # Write to TensorBoard
            # Use the trainer's writer (TensorBoard SummaryWriter)
            if hasattr(trainer, "writer") and trainer.writer is not None:
                trainer.writer.add_scalar(
                    tag=f"IsaacLab/{metric_name}",
                    scalar_value=scalar_value,
                    global_step=timestep
                )
    
    def on_episode_end(self, trainer: Any, timestep: int, timesteps: int) -> None:
        """Called at the end of each episode.
        
        This is an alternative hook point. Currently not used, but available
        for episode-level logging if needed.
        
        Args:
            trainer: The skrl trainer instance
            timestep: Current timestep
            timesteps: Total timesteps for training
        """
        pass  # Could be used for episode-level aggregations

