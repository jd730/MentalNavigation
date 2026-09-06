#!/bin/bash
#
# Scaling-factor experiments: VHA-D with the internal_corr_suc_3
# scaling-factor head enabled (-scaling_factor) and the internal
# velocity multiplied (-resolution > 1). Matches the canonical
# paper scaling sweep (scripts/run_scaling.sh in the legacy tree).
# Lambda triples are chosen so the grid-wrap consistency check
# `lambda / resolution == (input_dim + interval_dim) / step_size`
# (train.py:period_match) still holds:
#
#   resolution 2 + lambdas (22, 24, 26)  -> 11, 12, 13 effective
#   resolution 3 + lambdas (33, 36, 39)  -> 11, 12, 13 effective
#
# Slurm array layout: 2 resolution variants x 50 seeds = 100 tasks (0..99).
#   variant_idx = SLURM_ARRAY_TASK_ID / 50
#   seed_idx    = SLURM_ARRAY_TASK_ID % 50

#SBATCH -c 8
#SBATCH -n 1
#SBATCH -p ou_bcs_low,ou_bcs_normal
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-99

set -euo pipefail

RESOLUTIONS=(2 3)
LAMBDAS_LIST=("22 24 26" "33 36 39")
SEEDS=({0..49})

variant_idx=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
seed_idx=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
RESOLUTION=${RESOLUTIONS[$variant_idx]}
LAMBDAS=${LAMBDAS_LIST[$variant_idx]}
SEED=${SEEDS[$seed_idx]}

SAVE_DIR="results_new/scaling_factor"

python train.py \
    -model VHA-D \
    -scaling_factor \
    -lambdas $LAMBDAS \
    -resolution "$RESOLUTION" \
    -seed "$SEED" \
    -epochs 10000 \
    -save_dir "$SAVE_DIR" \
    -project_name 'mental_navigation_publish_scaling_factor'
