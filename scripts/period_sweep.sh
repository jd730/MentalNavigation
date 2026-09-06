#!/bin/bash
#
# VHA-D period sweep: vary the middle grid-module period to test
# how grid periodicity affects the path-integration code. Matches the
# canonical paper period sweep (scripts/run_period.sh in the legacy
# tree). VHA-D is hardcoded since this analysis only makes sense
# with the grid-cell wrapper, which is auto-on for VHA / VHA-D.
#
# Slurm array layout: 6 period configs x 50 seeds = 300 tasks (0..299).
#   period_idx = SLURM_ARRAY_TASK_ID / 50
#   seed_idx   = SLURM_ARRAY_TASK_ID % 50

#SBATCH -c 8
#SBATCH -n 1
#SBATCH -p ou_bcs_low,ou_bcs_normal
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-299

set -euo pipefail

# Middle-period sweep; lambdas[0] and lambdas[2] are fixed at 11 and 13.
LAMBDAS_LIST=(
    "11 4 13"
    "11 6 13"
    "11 8 13"
    "11 12 13"
    "11 16 13"
    "11 20 13"
)
SEEDS=({0..49})

period_idx=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
seed_idx=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
LAMBDAS=${LAMBDAS_LIST[$period_idx]}
SEED=${SEEDS[$seed_idx]}

SAVE_DIR="results_new/period_sweep"

python train.py \
    -model VHA-D \
    -lambdas $LAMBDAS \
    -seed "$SEED" \
    -trial_length 200 \
    -save_dir "$SAVE_DIR" \
    -project_name 'mental_navigation_publish_period_sweep'
