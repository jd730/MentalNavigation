#!/bin/bash
#
# Train one row of the baseline grid (model x seed).
#
# CLI defaults (lib/arguments.py) already encode the canonical
# paper config -- optimizer adam, lr 1e-3, feature_dim 256,
# trial_length 100, input_dim 384, interval_dim 384, step_size 64,
# epochs 2000, num_envs 3, alpha 0.9, log_interval 100, etc.
# This script only needs to set -model and -seed (grid cells are
# auto-enabled in train.py for VHA / VHA-D).
#
# Slurm array layout: 9 models x 50 seeds = 450 tasks (0..449).
#   model_idx = SLURM_ARRAY_TASK_ID / 50
#   seed_idx  = SLURM_ARRAY_TASK_ID % 50

#SBATCH -c 8
#SBATCH -n 1
#SBATCH -p ou_bcs_low,ou_bcs_normal
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-449

set -euo pipefail

MODELS=("RNN" "RNN-D" "RNN_FF" "RNN-D_FF" "RNN_action" "RNN_autoreg" "RNN-D_autoreg" "VHA" "VHA-D")
SEEDS=({0..49})

model_idx=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
seed_idx=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
MODEL=${MODELS[$model_idx]}
SEED=${SEEDS[$seed_idx]}

SAVE_DIR="results_new/baselines"

python train.py \
    -model "$MODEL" \
    -seed "$SEED" \
    -save_dir "$SAVE_DIR" \
    -project_name 'mental_navigation_publish_baselines'
