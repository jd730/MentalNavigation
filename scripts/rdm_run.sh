#!/bin/bash
#SBATCH -c 8
#SBATCH -n 1
#SBATCH -p ou_bcs_low,ou_bcs_normal
#SBATCH -t 1:00:00
#SBATCH --array=0-49
#SBATCH -o slurm_logs/rdm_%A_%a.out
#SBATCH -e slurm_logs/rdm_%A_%a.err
#SBATCH --mem=32G

# RDM analysis batch script - one slurm task per training seed.
#
# Each task runs rdm_analysis.py for one seed (0-49). All tasks share the
# same monkey-RDM cache directory, so the first task to reach the monkey
# step does the heavy gaussian-filter pass and writes the cache; the other
# 49 tasks pick the cached pattern up. Each task writes its own
# pair_scores_<mode>_S<seed>_T<t_common>_<sm>_<metric>.npz so the aggregator
# can later read all 50 in one pass.
#
# Configuration knobs (override on the sbatch command line with --export if
# you want):

METRIC=${METRIC:-correlation}   # one of: correlation, euclidean, cosine
SMOOTHING=${SMOOTHING:-0}       # 1 -> pass --smoothing, 0 -> raw model
SAVE_DIR=${SAVE_DIR:-results/RDM}
CONDITIONS=${CONDITIONS:-pair}  # distance (10) or pair (30 curr->target)
PCA_COMPONENTS=${PCA_COMPONENTS:-256}   # PCA basis size, fit once per system
# Choose ONE of:
#   PCA_USE        -- fixed top-K (0 disables PCA)
#   PCA_EVR        -- cumulative explained-variance threshold (e.g. 0.8)
# If PCA_EVR is set non-empty it takes precedence over PCA_USE.
PCA_USE=${PCA_USE:-0}
PCA_EVR=${PCA_EVR:-0.8}

mkdir -p slurm_logs

SEED=${SLURM_ARRAY_TASK_ID}
echo "seed=${SEED} t_common=${T_COMMON} metric=${METRIC} smoothing=${SMOOTHING}"

SMOOTH_FLAG=""
[ "${SMOOTHING}" = "1" ] && SMOOTH_FLAG="--smoothing"

PCA_FLAGS="--pca_components ${PCA_COMPONENTS}"
if [ -n "${PCA_EVR}" ]; then
    PCA_FLAGS="${PCA_FLAGS} --pca_evr ${PCA_EVR}"
else
    PCA_FLAGS="${PCA_FLAGS} --pca_use ${PCA_USE}"
fi

python3 rdm_analysis.py \
    --seed       "${SEED}" \
    --t_common   12 \
    --metric     "${METRIC}" \
    --save_dir   "${SAVE_DIR}" \
    --conditions "${CONDITIONS}" --pca_use 3 # \
#    ${PCA_FLAGS}  \
#    --smoothing
