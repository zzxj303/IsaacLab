#!/usr/bin/env bash

set -e

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Experimental-Quadcopter-Forest-Pose-Direct-Migrated-Play-v0 \
  --num_envs 15
