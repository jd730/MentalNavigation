"""
Aggregate per-seed RDM pairing scores into a single table and summary plot.

Reads every ``pair_scores_<mode>_S<seed>_T<t_common>_<sm>_<metric>.npz``
written by ``rdm_analysis.py`` that matches the user's filter and produces:

    - aggregated_pair_scores_<tag>.csv : one row per (seed, mode); columns are
      the four monkey-region <-> model-layer Spearman scores plus the
      configuration knobs.
    - aggregated_pair_scores_<tag>.pdf/.png : box+swarm plot of the four
      scores per mode, showing the across-seed distribution.

The CSV makes downstream stats (paired tests, noise ceiling) easy.

This file is the launching code only. The library helpers (`collect`,
`add_specificity_columns`, `wilcoxon_table`, `plot_specificity`,
`plot_distributions`) live in `lib.analyze_utils.rdm_aggregate`.
"""
import os
import argparse

from lib.analyze_utils.rdm import variant_subdir
from lib.analyze_utils.rdm_aggregate import (
    PAIR_KEYS,
    collect, add_specificity_columns, wilcoxon_table,
    plot_specificity, plot_distributions,
)


def main():
    """CLI entry point: aggregate per-seed RDM scores into one table + figures.

    Globs every ``pair_scores_<mode>_S*_T<t_common>_<sm>_<metric>.npz``
    written by rdm_analysis.py that matches the ``--save_dir`` /
    ``--t_common`` / ``--metric`` / ``--smoothing`` / ``--pca_use`` (or
    ``--pca_evr``) / ``--conditions`` filter, then writes:

      - ``aggregated_pair_scores_<tag>.csv``      one row per (seed, mode)
      - ``aggregated_pair_scores_<tag>.{pdf,png}`` box+swarm of the four
                                                  monkey<->model pairings
      - ``specificity_wilcoxon_<tag>.csv``        one-sided Wilcoxon stats
      - ``specificity_box_<tag>.{pdf,png}``       box+swarm of the three
                                                  specificity contrasts
    into ``<save_dir>/<conditions>/<pca>/``.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--save_dir', type=str,
                        default='results/RDM',
                        help='Directory containing the pair_scores npz files.')
    parser.add_argument('--t_common', type=int, default=12,
                        help='Filter for this T_common value (default 12).')
    parser.add_argument('--metric', type=str, default='correlation',
                        help='Filter for this metric (default correlation).')
    parser.add_argument('--smoothing', action='store_true',
                        help='Aggregate the smoothed-model runs instead of '
                             'the no-smoothing runs.')
    parser.add_argument('--pca_use', type=int, default=0,
                        help='Match runs that used --pca_use=K (0 = raw '
                             'neuron-space runs, default).')
    parser.add_argument('--pca_evr', type=float, default=None,
                        help='Match runs that used --pca_evr=THRESH instead '
                             'of a fixed K (e.g. 0.8). Overrides --pca_use.')
    parser.add_argument('--conditions', type=str, default='distance',
                        choices=('distance', 'pair'),
                        help='Match the --conditions value used in '
                             'rdm_analysis.py (default distance).')
    parser.add_argument('--tag', type=str, default=None,
                        help='Tag for output file names (default auto: '
                             'T<t_common>_<sm|nosm>_<metric>[_pca<K> or '
                             '_pcaEVR<pct>][_pair]).')
    args = parser.parse_args()

    sm_tag = 'sm' if args.smoothing else 'nosm'
    if args.pca_evr is not None:
        pca_tag = f'_pcaEVR{int(round(args.pca_evr*100))}'
    elif args.pca_use > 0:
        pca_tag = f'_pca{args.pca_use}'
    else:
        pca_tag = ''
    cond_tag = '' if args.conditions == 'distance' else f'_{args.conditions}'
    tag = (args.tag if args.tag is not None
           else f'T{args.t_common}_{sm_tag}_{args.metric}{pca_tag}{cond_tag}')

    # Outputs and inputs both live in <save_dir>/<cond>/<pca>/ - matching
    # the layout written by rdm_analysis.py.
    variant_dir = os.path.join(args.save_dir,
                               variant_subdir(args.conditions,
                                              pca_use=args.pca_use,
                                              pca_evr=args.pca_evr))
    os.makedirs(variant_dir, exist_ok=True)
    print(f'variant_dir = {variant_dir}')

    pattern = os.path.join(
        variant_dir,
        f'pair_scores_*_S*_T{args.t_common}_{sm_tag}_{args.metric}'
        f'{pca_tag}{cond_tag}.npz',
    )
    print(f'glob: {pattern}')
    df = collect(pattern)
    print(f'collected {len(df)} rows over '
          f'{df["seed"].nunique()} seeds, '
          f'{df["mode"].nunique()} modes')

    df = add_specificity_columns(df)
    csv_path = os.path.join(variant_dir, f'aggregated_pair_scores_{tag}.csv')
    df.to_csv(csv_path, index=False)
    print(f'wrote {csv_path}')

    print('\nMean +/- std across seeds (Spearman):')
    grp = df.groupby('mode')[list(PAIR_KEYS)]
    for mode, sub in grp:
        print(f'  mode = {mode}')
        for k in PAIR_KEYS:
            v = sub[k]
            print(f'    {k:14s}  mean={v.mean():+.3f}  '
                  f'std={v.std():.3f}  n={len(v)}')

    # Per-system K distribution. Monkey K is constant across seeds (one PCA
    # basis); the two model-side K's vary per seed.
    k_cols = [c for c in ('pca_k_7a', 'pca_k_ec',
                          'pca_k_hidden', 'pca_k_base_hidden')
              if c in df.columns]
    if k_cols:
        print('\nPer-system K across seeds (mean / min / max):')
        for c in k_cols:
            v = df[c]
            print(f'  {c:20s}  mean={v.mean():.2f}  '
                  f'min={v.min()}  max={v.max()}  '
                  f'unique={sorted(v.unique())}')

    wdf = wilcoxon_table(df)
    print('\nSpecificity Wilcoxon (one-sided, H1: median > 0):')
    print(wdf.to_string(index=False,
                        float_format=lambda x: f'{x:+.3g}'))
    w_path = os.path.join(variant_dir,
                          f'specificity_wilcoxon_{tag}.csv')
    wdf.to_csv(w_path, index=False)
    print(f'wrote {w_path}')

    for ext in ('pdf', 'png'):
        out = os.path.join(variant_dir,
                           f'aggregated_pair_scores_{tag}.{ext}')
        plot_distributions(df, out,
                           title=f'RDM Spearman vs monkey  '
                                 f'(T_common={args.t_common}, '
                                 f'{sm_tag}, {args.metric})')
        print(f'wrote {out}')

        out = os.path.join(variant_dir,
                           f'specificity_box_{tag}.{ext}')
        plot_specificity(df, out,
                         title=f'Pairing specificity  '
                               f'(T_common={args.t_common}, '
                               f'{sm_tag}, {args.metric})')
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
