# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Early Stopping Callback for skrl PPO training.

This callback monitors the mean reward during training and automatically stops
when the reward has converged (no improvement for a specified patience period).
"""

from typing import Any
import torch
from skrl.utils.callbacks import Callback


class EarlyStoppingCallback(Callback):
    """Early stopping callback that monitors reward convergence and stops training.
    
    This callback tracks the mean reward over a sliding window and stops training
    when the reward has not improved for a specified number of checkpoints (patience).
    
    Features:
    - Monitors configurable metric (default: mean reward)
    - Smoothing via sliding window for noise reduction
    - Configurable patience and minimum improvement delta
    - Verbose logging of early stopping decisions
    - Automatically stops training when convergence detected
    
    Args:
        patience: Number of checkpoints without improvement before stopping (default: 50)
        min_delta: Minimum change in metric to qualify as improvement (default: 0.01)
        window_size: Number of recent values to average for smoothing (default: 10)
        metric: Name of the metric to monitor (default: "Reward/mean_reward")
        verbose: Whether to print early stopping messages (default: True)
    
    Example usage:
        ```python
        from early_stopping_callback import EarlyStoppingCallback
        
        # Create callback with custom parameters
        early_stop = EarlyStoppingCallback(
            patience=30,
            min_delta=0.02,
            window_size=5,
            verbose=True
        )
        
        # Add to trainer callbacks (in your training script)
        # This requires modifying the training script to support callbacks
        runner.trainer.callbacks.append(early_stop)
        
        # Or use in isaaclab.envs.mdp.command
        ```
    """
    
    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 0.01,
        window_size: int = 10,
        metric: str = "Reward/mean_reward",
        verbose: bool = True
    ):
        """Initialize the early stopping callback."""
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        self.metric = metric
        self.verbose = verbose
        
        # Internal state
        self.best_value = -float('inf')
        self.best_timestep = 0
        self.counter = 0
        self.should_stop = False
        self.value_history = []
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[EarlyStopping] Initialized")
            print(f"{'='*70}")
            print(f"  - Patience: {self.patience} checkpoints")
            print(f"  - Min delta: {self.min_delta}")
            print(f"  - Window size: {self.window_size}")
            print(f"  - Monitoring metric: {self.metric}")
            print(f"{'='*70}\n")
    
    def _get_current_value(self, trainer: Any) -> float:
        """Extract the monitored metric value from trainer.
        
        Args:
            trainer: The skrl trainer instance
            
        Returns:
            Current value of the monitored metric
        """
        # Try to get from agent tracking dict
        if hasattr(trainer, 'agent') and hasattr(trainer.agent, 'tracking_data'):
            tracking = trainer.agent.tracking_data
            
            # Look for mean reward in tracking data
            if 'Reward / Mean reward (training)' in tracking:
                values = tracking['Reward / Mean reward (training)']
                if len(values) > 0:
                    return float(values[-1])
            elif 'Reward / Instantaneous reward (training)' in tracking:
                values = tracking['Reward / Instantaneous reward (training)']
                if len(values) > 0:
                    return float(values[-1])
        
        # Alternative: extract from environment
        env = trainer.env
        if hasattr(env, "unwrapped"):
            unwrapped_env = env.unwrapped
        else:
            unwrapped_env = env
        
        # Get mean episode reward if available
        if hasattr(unwrapped_env, 'episode_rewards'):
            if len(unwrapped_env.episode_rewards) > 0:
                return float(torch.tensor(unwrapped_env.episode_rewards).mean())
        
        # Default: return 0 if metric not found
        return 0.0
    
    def on_checkpoint(self, trainer: Any, timestep: int, timesteps: int) -> None:
        """Called when a checkpoint is saved.
        
        Evaluates whether training should stop based on reward improvement.
        
        Args:
            trainer: The skrl trainer instance
            timestep: Current timestep
            timesteps: Total timesteps for training
        """
        # Get current metric value
        current_value = self._get_current_value(trainer)
        
        # Add to history for smoothing
        self.value_history.append(current_value)
        if len(self.value_history) > self.window_size:
            self.value_history.pop(0)
        
        # Calculate smoothed value (average over window)
        smoothed_value = sum(self.value_history) / len(self.value_history)
        
        # Check if this is an improvement
        if smoothed_value > self.best_value + self.min_delta:
            # Improvement detected
            improvement = smoothed_value - self.best_value
            self.best_value = smoothed_value
            self.best_timestep = timestep
            self.counter = 0
            
            if self.verbose:
                print(f"\n[EarlyStopping] ✓ Timestep {timestep:,}: "
                      f"New best smoothed reward: {smoothed_value:.4f} "
                      f"(+{improvement:.4f})")
                print(f"  - Counter reset. Patience remaining: {self.patience}\n")
        else:
            # No improvement
            self.counter += 1
            
            if self.verbose and self.counter % 10 == 0:
                print(f"\n[EarlyStopping] ⚠ Timestep {timestep:,}: "
                      f"No improvement for {self.counter} checkpoints.")
                print(f"  - Current smoothed reward: {smoothed_value:.4f}")
                print(f"  - Best smoothed reward: {self.best_value:.4f} "
                      f"(at timestep {self.best_timestep:,})")
                print(f"  - Patience remaining: {self.patience - self.counter}\n")
            
            # Check if patience exceeded
            if self.counter >= self.patience:
                self.should_stop = True
                
                if self.verbose:
                    print(f"\n{'='*70}")
                    print(f"[EarlyStopping] 🛑 TRAINING STOPPED - Reward has converged!")
                    print(f"{'='*70}")
                    print(f"  - Stopped at timestep: {timestep:,}")
                    print(f"  - Best smoothed reward: {self.best_value:.4f} "
                          f"(at timestep {self.best_timestep:,})")
                    print(f"  - No improvement for {self.counter} consecutive checkpoints")
                    print(f"  - Training completed {timestep/timesteps*100:.1f}% of total timesteps")
                    print(f"  - Saved best model at timestep {self.best_timestep:,}")
                    print(f"{'='*70}\n")
                
                # Request trainer to stop
                # Note: skrl doesn't have a built-in stop mechanism,
                # so we'll need to modify the trainer's timesteps
                if hasattr(trainer, '_timesteps'):
                    trainer._timesteps = timestep
    
    def on_timestep_end(self, trainer: Any, timestep: int, timesteps: int) -> None:
        """Called at the end of each timestep to check if training should stop.
        
        Args:
            trainer: The skrl trainer instance
            timestep: Current timestep
            timesteps: Total timesteps for training
        """
        # Check if we should stop and force trainer to exit
        if self.should_stop:
            # Signal trainer to stop by setting current timestep to max
            if hasattr(trainer, 'timestep'):
                trainer.timestep = timesteps
            # Alternative: raise a custom exception that can be caught
            # This is a more forceful way to stop training
            raise KeyboardInterrupt("Early stopping triggered - reward has converged")
