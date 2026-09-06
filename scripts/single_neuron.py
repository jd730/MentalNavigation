"""Regenerate per-seed single-neuron firing-rate dumps at the canonical
paper protocol (sigma=2, truncate=2, ignore_last=True) for VHA-ReLU and
VHA-ReLU-D. This is the setting used for the final single-neuron
visualization figures; the sigma=4 dumps used elsewhere (e.g. PCA)
coexist under separate subdirs and are unaffected.

Output layout per seed:
    <run>/001999/visual__0_<target>/indiv_ignore_lastleftSig2Truncate2/

Runs analyses=['firing_rate', 'neurons'] on all 50 seeds. Options
``{'ignore_last':True, 'smoothing_sigma':2, 'smoothing_truncate':2}``
route into the sigma=2 path in the firing-rate / neurons handlers.

Historical note: this script replaces three earlier variants
(single_neuron_sig2, _noignorelast, _sig4_ignorelast) that existed
during the (sigma, ignore_last) sweep; that comparison is complete
and only the canonical config is kept.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.analysis import ANALYSES, run_dir
from analyze_relu import (
    ANALYSIS_EPOCH,
    RELU_MODELS,
    SWEEP_DIR,
    SAVE_DIR,
    _model_constraint,
)


LEGACY_SIGMA_OPTIONS = {
    'firing_rate': {'ignore_last': True, 'smoothing_sigma': 2, 'smoothing_truncate': 2,
                    'separate_direction': True},
    'neurons':     {'smoothing_sigma': 2, 'smoothing_truncate': 2},
}


def _seed_target_done(dynamics_dir, run_name, epoch, target):
    """Skip runs whose indiv_ignore_lastleftSig2Truncate2/ already has 256 PDFs
    -- one per neuron. Cheap resume gate for the sigma=2 batch."""
    p = os.path.join(dynamics_dir, run_name, epoch,
                     f'visual__0_{target}',
                     'indiv_ignore_lastleftSig2Truncate2')
    if not os.path.isdir(p):
        return False
    return len([f for f in os.listdir(p) if f.endswith('.pdf')]) >= 256


PRIORITY_SEEDS = [19, 28]


def _all_seeds_for(constraint):
    seeds = []
    for run_name in sorted(os.listdir(SWEEP_DIR)):
        if not all(c in run_name for c in constraint):
            continue
        try:
            seeds.append(int(run_name.split('Ep')[0].split('S')[-1]))
        except ValueError:
            continue
    return seeds


def main():
    dynamics_dir = os.path.join(SAVE_DIR, 'dynamics')
    # Union of seeds across models -- priority seeds get done end-to-end
    # (both models, both targets) before the rest even start.
    per_model = {m: _all_seeds_for([_model_constraint(m)]) for m in RELU_MODELS}
    union = sorted(set().union(*per_model.values()))
    ordered = ([s for s in PRIORITY_SEEDS if s in union]
               + [s for s in union if s not in PRIORITY_SEEDS])
    for seed in ordered:
        for m in RELU_MODELS:
            if seed not in per_model[m]:
                continue
            targets = ('hidden', 'base_hidden') if m.endswith('-D') else ('hidden',)
            constraint = [_model_constraint(m)]
            for target in targets:
                run_name = next((r for r in os.listdir(SWEEP_DIR)
                                 if all(c in r for c in constraint) and f'S{seed}Ep' in r), None)
                if run_name is None:
                    continue
                if _seed_target_done(dynamics_dir, run_name, ANALYSIS_EPOCH, target):
                    continue
                print(f'\n=== {m} S{seed} target={target} (sigma=2, ignore_last=True) ===')
                run_dir(dirname=SWEEP_DIR, save_dir=dynamics_dir, target=target,
                        analyses=['firing_rate', 'neurons'],
                        force=set(ANALYSES.keys()),
                        constraints=constraint,
                        seeds=[seed],
                        visual_only=True, save_indiv=True,
                        use_cuda=True, verbose=True,
                        epoch=ANALYSIS_EPOCH,
                        options=LEGACY_SIGMA_OPTIONS)
    print('\nall done')


if __name__ == '__main__':
    main()
