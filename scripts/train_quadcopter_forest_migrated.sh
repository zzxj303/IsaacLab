#!/usr/bin/env bash

set -e
# Use workspace source packages explicitly. The installed isaaclab package may have
# different lidar discretization behavior, which changes observation dimension.
export PYTHONPATH="$PWD/source/isaaclab:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_rl:${PYTHONPATH}"
  # --video \
  # --video_length 400 \
  # --video_interval 50000
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Experimental-Quadcopter-Forest-Pose-Direct-Migrated-v0 \
  --num_envs 2048 \
  --headless

