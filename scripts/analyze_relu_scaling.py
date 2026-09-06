"""End-to-end analysis for the VHA-ReLU-D velocity-scaling sweep.

Consolidates the previous three variants (analyze_relu_scaling.py for
the WITH-SF sweep at ReLU_scaling, plus _no_sf.py and _with_sf.py for
the split control + backup) into one script parameterized via CLI.

The KDE loads periodicity npys directly (not the dataframes), so it
naturally includes 'failed' seeds -- any seed whose
run_info_<epoch>.pth exists shows up regardless of whether its
accuracy hit the training threshold. (Seeds with zero valid trials
after suc_only+uni_only filtering do drop out earlier in
per_seed_periodicity because there's literally nothing to plot.)

Usage
-----
    # WITH-SF sweep (original)
    python scripts/analyze_relu_scaling.py \\
        --sweep_dir decoder_dir_mental/ReLU_scaling \\
        --save_dir  results/ReLU_scaling

    # NO-SF control
    python scripts/analyze_relu_scaling.py \\
        --sweep_dir decoder_dir_mental/ReLU_scaling_no_sf \\
        --save_dir  results/ReLU_scaling_no_sf

    # WITH-SF archived backup (log_interval=100 preservation)
    python scripts/analyze_relu_scaling.py \\
        --sweep_dir decoder_dir_mental/ReLU_scaling_log100_backup \\
        --save_dir  results/ReLU_scaling_with_sf

    # Skip expensive per-seed periodicity when the npys already exist
    python scripts/analyze_relu_scaling.py --sweep_dir ... --save_dir ... \\
        --skip_periodicity
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.analyze_utils.collect import (
    read_results,
    draw_generalization,
    draw_aggregated_periodicity,
)
from analyze_relu import (
    ANALYSIS_EPOCH,
    per_seed_periodicity,
    _model_constraint,
)


MODEL = 'VHA-ReLU-D'
RESOLUTIONS = [2, 3]
# Scaling runs train for 10000 epochs/env (vs canonical 2000); first-env
# checkpoint is 009999, matching the "end-of-env0" paper convention.
SCALING_EPOCH = '009999'


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--sweep_dir', required=True,
                   help='Directory of per-seed run subdirs '
                        '(e.g. decoder_dir_mental/ReLU_scaling_no_sf).')
    p.add_argument('--save_dir', required=True,
                   help='Output root '
                        '(e.g. results/ReLU_scaling_no_sf).')
    p.add_argument('--epoch', default=SCALING_EPOCH,
                   help='Checkpoint suffix to analyze; default %(default)s.')
    p.add_argument('--skip_periodicity', action='store_true',
                   help='Assume periodicity npys already on disk under '
                        'save_dir/dynamics/. Useful for a fast rebuild of '
                        'the generalization figures once the periodicity '
                        'npys are already there.')
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    dfs = read_results(args.sweep_dir, [_model_constraint(MODEL)])
    if len(dfs) == 0:
        print(f'no runs found under {args.sweep_dir}')
        return
    dfs['model'] = MODEL
    seeds = sorted(dfs['seed'].unique().tolist())
    print(f'{len(seeds)} seeds available '
          f'(resolutions in df: {sorted(dfs["resolution"].unique())})')

    if not args.skip_periodicity:
        print('\n=== per-seed periodicity (autocorr + FR + regression) ===')
        per_seed_periodicity(args.sweep_dir, args.save_dir, seeds=seeds,
                             focus_seed=None,
                             targets=('hidden', 'base_hidden'),
                             force=False,
                             epoch=args.epoch)

    # generalization1 expects a 'visual/all/0/accuracy' column that
    # read_results doesn't emit for scaling runs; derive it as the
    # seen/unseen mean per env.
    for env in (0, 1, 2):
        seen = f'visual/seen/{env}/accuracy'
        unseen = f'visual/unseen/{env}/accuracy'
        if seen in dfs.columns and unseen in dfs.columns:
            dfs[f'visual/all/{env}/accuracy'] = dfs[[seen, unseen]].mean(axis=1)
        seen_m = f'mental/seen/{env}/accuracy'
        unseen_m = f'mental/unseen/{env}/accuracy'
        if seen_m in dfs.columns and unseen_m in dfs.columns:
            dfs[f'mental/all/{env}/accuracy'] = dfs[[seen_m, unseen_m]].mean(axis=1)

    for res in RESOLUTIONS:
        sub = dfs[dfs['resolution'] == res]
        if len(sub) == 0:
            print(f'\nresolution={res}: no rows, skipping')
            continue
        res_seeds = sorted(sub['seed'].unique().tolist())
        print(f'\n=== resolution={res} ({len(res_seeds)} seeds) ===')

        # 3a. generalization 1/2/3-1/3-2 under save_dir/generalization/Res<N>/.
        # Scaling sweep uses 10000 epochs/env (vs canonical 2000); pass
        # through so the Env1/Env2/Env3 dividers land in the right places.
        gen_out = os.path.join(args.save_dir, 'generalization', f'Res{res}')
        os.makedirs(gen_out, exist_ok=True)
        draw_generalization(
            None, gen_out, epoch=10000, _format='pdf',
            data=[sub],
            linestyles=['solid'],
            colors=None,
        )
        print(f'  wrote generalization plots -> {gen_out}')

        # 3b. Aggregated periodicity KDE (both layers), per-resolution
        #     subdir so Res2/Res3 don't collide.
        agg_out = os.path.join(args.save_dir, 'agg_periodicity', f'Res{res}')
        os.makedirs(agg_out, exist_ok=True)
        res_constraint = [_model_constraint(MODEL), f'_Res{res}_']
        for target in ('visual__0_hidden', 'visual__0_base_hidden'):
            try:
                draw_aggregated_periodicity(
                    res_seeds, args.save_dir, constraint=res_constraint,
                    target=target, save_dir=agg_out, _format='pdf',
                    resolutions=[], mode='kde', target_seeds=[],
                )
                print(f'  wrote KDE {target} -> {agg_out}')
            except Exception as e:  # noqa: BLE001
                print(f'  KDE {target} failed: {e}')


if __name__ == '__main__':
    main()
