#!/bin/bash
#
# Baseline sweep: RNN and RNN-D at 2000 epochs, 50 seeds each.
# These are the paper's non-grid-cell baselines against which the
# VHA-ReLU / VHA-ReLU-D generalization plots compare. Save layout matches
# the ReLU sweep so analyze_relu.py can stitch them into one figure.
#
# Slurm array layout: 2 models x 50 seeds = 100 tasks (0..99).
#   model_idx = SLURM_ARRAY_TASK_ID / 50
#   seed_idx  = SLURM_ARRAY_TASK_ID % 50

#SBATCH -c 8
#SBATCH -n 1
#SBATCH -p ou_bcs_low,ou_bcs_normal
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-99
#SBATCH -o slurm_logs/rnn_baseline_%A_%a.out
#SBATCH -e slurm_logs/rnn_baseline_%A_%a.err

set -euo pipefail

mkdir -p slurm_logs

# Explicitly activate the mental conda env. The default `python` on the
# compute nodes points at /home/software/anaconda3/2023.07/bin/python whose
# typing_extensions is too old for the pinned torch build.
source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate mental
echo "python: $(which python)"

MODELS=("RNN" "RNN-D")
SEEDS=({0..49})

model_idx=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
seed_idx=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
MODEL=${MODELS[$model_idx]}
SEED=${SEEDS[$seed_idx]}

SAVE_DIR="decoder_dir_mental/RNN_baseline"

python train.py \
    -model "$MODEL" \
    -seed "$SEED" \
    -save_dir "$SAVE_DIR" \
    -project_name 'mental_navigation_publish_rnn_baseline'
