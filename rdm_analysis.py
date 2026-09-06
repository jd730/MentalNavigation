"""
Representational Dissimilarity Matrix (RDM) analysis comparing monkey neural
recordings to a trained RNN model performing the mental-navigation task.

Background
----------
RDMs are a standard tool for comparing the geometry of representations across
brains and models without requiring matched units (Kriegeskorte 2008,
https://pmc.ncbi.nlm.nih.gov/articles/PMC2605405/; Khaligh-Razavi &
Kriegeskorte 2014, https://www.pnas.org/doi/10.1073/pnas.1403112111). For each
"system" (a recorded brain region or a model layer), we:
    1. compute a single response *pattern* per experimental condition,
    2. compute pairwise dissimilarity between condition patterns -> the RDM,
    3. compare RDMs across systems by Spearman-correlating their
       upper triangles.

This script does that for four systems:
    monkey 7a            -- "PPC"      (file 7a_*_a_neur_tensor_joyon.mat)
    monkey EC            -- "EC"       (file ec_*_a_neur_tensor_joyon.mat)
    model `hidden`       -- PPC-like   ("action RNN", in REV_ABS/vHMN2 seed 43)
    model `base_hidden`  -- EC-like    ("distance RNN", same model file)

Conditions
----------
Signed integer distances [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]. The monkey
records distance directly in `pm.dist_conditions`. The model stores it in
units of 12 (`raw_dist = trajs[:,1] - trajs[:,0] in {-60,-48,...,+60}`), so we
divide by 12 to align scales.

Two ways of building the per-condition pattern
----------------------------------------------
    'time_mean' : average activity over the navigation window per trial, then
                  across trials -> a single (n_neurons,) vector per condition.
    'flat'      : trial-average, then linearly time-stretch the navigation
                  window to T_COMMON bins, then flatten -> a
                  (T_COMMON * n_neurons,) vector per condition. Captures
                  temporal structure that 'time_mean' collapses.

Three dissimilarity metrics
---------------------------
    correlation  : 1 - Pearson r           (mean / scale invariant)
    euclidean    : L2                      (sensitive to scale)
    cosine       : 1 - cos(theta)          (scale invariant, mean sensitive)

Cross-system RDM similarity is the Spearman rank correlation of the upper
triangles of the two RDMs (the standard 2nd-order RSA statistic).

Notes
-----
- Monkey preprocessing copies pca.py: spike counts / 1000, then
  gaussian_filter1d along time with sigma=400 ms (binwidth = 1 ms).
- Model preprocessing copies pca.py: gaussian_filter1d with sigma=4,
  truncate=2 along the time axis. Each model trial has its first
  (T - (|raw_d|+1)) timesteps padded with NaN, so we slice each trial down
  to its valid navigation window *before* smoothing - otherwise the
  gaussian propagates NaN through the whole trial.
"""
import os
import glob
import argparse

import numpy as np

from lib.analyze_utils.rdm import (
    DISTANCE_CONDITIONS, PAIR_CONDITIONS,
    MONKEY_FILES, METRIC_CHOICES, T_COMMON,
    compute_cond_idx, label_distance, variant_subdir,
    load_model, condition_patterns,
    fit_pca_basis, pick_k_evr, project_activity,
    compute_rdm, rdm_spearman,
    MonkeyRDMCache,
)

def main():
    """CLI: compute monkey-vs-model RDMs for one seed and save the per-seed npz.

    For the seed chosen via ``--seed``, loads both monkey neural tensors
    (7a + EC) and the trained model's ``hidden`` / ``base_hidden`` layers,
    builds the per-condition pattern matrices in both ``time_mean`` and
    ``flat`` modes, computes the four RDMs, and the four monkey<->model
    Spearman pairings.

    Writes per-seed outputs under ``<save_dir>/<conditions>/<pca>/``:

      - ``rdms_<mode>_S<seed>_T<T>_<sm>_<metric>...npz``  per-mode RDM tensors
      - ``pair_scores_<mode>_S<seed>_T<T>_<sm>_<metric>...npz``  four
                                                                cross-system
                                                                Spearman scores
      - ``rdm_summary_<tag>.txt``  human-readable summary

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--seed', type=int, default=43,
                        help='Model training seed to analyze (default 43). '
                             'Selects the matching *S<seed>Ep* .mat file.')
    parser.add_argument('--t_common', type=int, default=T_COMMON,
                        help='Common time length for the "flat" RDM mode '
                             '(default %(default)s).')
    parser.add_argument('--save_dir', type=str,
                        default='results/RDM',
                        help='Base output directory.')
    parser.add_argument('--tag', type=str, default=None,
                        help='Optional tag appended to file names. Default is '
                             '"S<seed>_T<t_common>" so parallel runs do not '
                             'overwrite each other.')
    parser.add_argument('--model_glob', type=str, default=None,
                        help='Override the glob used to locate the model .mat '
                             'file. {seed} is substituted with the seed.')
    parser.add_argument('--monkey_cache_dir', type=str, default=None,
                        help='Directory holding the cached per-(region, mode, '
                             'T) monkey RDMs. Default: <save_dir>/_monkey_cache '
                             '(shared across analysis variants).')
    parser.add_argument('--rebuild_monkey_cache', action='store_true',
                        help='Force recomputing the monkey RDMs even if cache '
                             'files exist.')
    parser.add_argument('--smoothing', action='store_true',
                        help='Apply gaussian smoothing to the *model* activity '
                             'along time (sigma=4, truncate=2) before building '
                             'patterns. Default: off. Monkey is always smoothed '
                             'inside load_monkey() regardless of this flag.')
    parser.add_argument('--metric', type=str, default='correlation',
                        choices=METRIC_CHOICES,
                        help='Single dissimilarity metric to use when building '
                             'RDMs (default %(default)s = 1 - Pearson r, the '
                             'Yamins/Kriegeskorte standard).')
    parser.add_argument('--pca_components', type=int, default=256,
                        help='Number of PCA components to fit per system '
                             '(default 256). Cached on disk per system. Only '
                             'used when --pca_use > 0.')
    parser.add_argument('--pca_use', type=int, default=0,
                        help='Number of top PCs to keep when projecting '
                             'activity before building RDMs. 0 disables PCA '
                             '(use raw neuron/unit space). Typical: 3.')
    parser.add_argument('--pca_evr', type=float, default=None,
                        help='If set (e.g. 0.8), choose K per system as the '
                             'smallest K with cumulative explained-variance '
                             'ratio >= this threshold. Overrides --pca_use.')
    parser.add_argument('--conditions', type=str, default='distance',
                        choices=('distance', 'pair'),
                        help='Condition labelling. "distance" -> 10 signed '
                             'distances (default). "pair" -> 30 (curr, '
                             'target) pairs over positions {1..6}.')
    parser.add_argument('--precompute_only', action='store_true',
                        help='Build the monkey smoothed-activity cache (and, '
                             'if pca_use>0, the PCA basis + projected '
                             'patterns) for both regions, then exit. Use this '
                             'as a setup step before launching the 50-seed '
                             'SLURM array to avoid 50 workers all racing on '
                             'the same load+smooth.')
    args = parser.parse_args()

    seed     = args.seed
    t_common = args.t_common
    metric   = args.metric
    # Auto-tag encodes seed, common-time length, smoothing flag and metric so
    # a batch run over many configurations stays unambiguous on disk.
    sm_tag = 'sm' if args.smoothing else 'nosm'
    if args.pca_evr is not None:
        # Encode threshold as integer percent for filename hygiene.
        pca_tag = f'_pcaEVR{int(round(args.pca_evr*100))}'
    elif args.pca_use > 0:
        pca_tag = f'_pca{args.pca_use}'
    else:
        pca_tag = ''
    cond_tag = '' if args.conditions == 'distance' else f'_{args.conditions}'
    tag = (args.tag if args.tag is not None
           else f'S{seed}_T{t_common}_{sm_tag}_{metric}{pca_tag}{cond_tag}')

    # Materialise the conditions list and pull in helper names that depend
    # on it.
    conditions = (PAIR_CONDITIONS if args.conditions == 'pair'
                  else DISTANCE_CONDITIONS)
    n_cond = len(conditions)
    print(f'conditions: {args.conditions} ({n_cond} conditions)')

    # Base save_dir holds the shared monkey cache; per-variant outputs live
    # under <save_dir>/<cond>/<pca>/ via variant_subdir(). This keeps a single
    # base dir tidy across many configurations.
    base_dir = args.save_dir
    out_dir = os.path.join(base_dir,
                           variant_subdir(args.conditions,
                                          pca_use=args.pca_use,
                                          pca_evr=args.pca_evr))
    os.makedirs(out_dir, exist_ok=True)
    monkey_cache_dir = (args.monkey_cache_dir
                        if args.monkey_cache_dir is not None
                        else os.path.join(base_dir, '_monkey_cache'))
    print(f'seed = {seed},  T_COMMON = {t_common},  tag = {tag}')
    print(f'base_dir = {base_dir}')
    print(f'out_dir  = {out_dir}')
    print(f'monkey_cache_dir = {monkey_cache_dir}')
    save_dir = out_dir  # downstream file writes target the variant subdir

    # -----------------------------------------------------------------------
    # 1. Locate the seed-<seed> model file. Prefer the rev_abs (norm5) folder;
    #    fall back to abs/ if the dataset has not been re-collected yet.
    # -----------------------------------------------------------------------
    if args.model_glob is not None:
        patterns = [args.model_glob.format(seed=seed)]
    else:
        patterns = [
            f'norm5_vHMN2_rev_abs/random*S{seed}Ep2000*.mat',
            f'abs/random*S{seed}Ep2000*.mat',
        ]
    model_candidates = []
    for p in patterns:
        model_candidates = sorted(glob.glob(p))
        if model_candidates:
            break
    assert model_candidates, f"no model .mat found for seed {seed}"
    model_path = model_candidates[0]
    print('Model file:', model_path)

    # -----------------------------------------------------------------------
    # 2. Set up monkey RDM caches (loaded lazily) and load the model.
    # -----------------------------------------------------------------------
    # When -rebuild_monkey_cache is set, wipe any matching cache files so the
    # next get_rdms() call is forced to recompute.
    if args.rebuild_monkey_cache and os.path.isdir(monkey_cache_dir):
        for f in glob.glob(os.path.join(monkey_cache_dir, 'monkey_*.npz')):
            os.remove(f)

    monk_7a_cache = MonkeyRDMCache('7a', MONKEY_FILES['7a'],
                                   monkey_cache_dir,
                                   pca_components=args.pca_components,
                                   cond_mode=args.conditions)
    monk_ec_cache = MonkeyRDMCache('EC', MONKEY_FILES['EC'],
                                   monkey_cache_dir,
                                   pca_components=args.pca_components,
                                   cond_mode=args.conditions)

    # PCA is "on" if either a fixed K is set or an EVR threshold is set.
    pca_on = (args.pca_use > 0) or (args.pca_evr is not None)

    def resolve_k(pca_obj, label=''):
        """Return K for one system, by either EVR threshold or fixed --pca_use."""
        if args.pca_evr is not None:
            K = pick_k_evr(pca_obj.explained_variance_ratio_, args.pca_evr)
            cum = pca_obj.explained_variance_ratio_[:K].sum()
            print(f'  {label}: K={K} (cum EVR={cum*100:.1f}%, '
                  f'threshold={args.pca_evr*100:.0f}%)')
            return K
        K = args.pca_use
        cum = pca_obj.explained_variance_ratio_[:K].sum()
        print(f'  {label}: K={K} (fixed; cum EVR={cum*100:.1f}%)')
        return K

    # Precompute step: warm the monkey smoothed-activity cache (and PCA basis
    # + projected pattern caches if PCA is on) for both regions, then exit
    # before touching the model. Use this as a one-off setup before launching
    # the 50-seed SLURM array, so workers all hit cache instead of racing on
    # the expensive sigma=400 smoothing.
    if args.precompute_only:
        for cache in (monk_7a_cache, monk_ec_cache):
            cache._load()
            if pca_on:
                pca = cache.get_pca()
                K = resolve_k(pca, label=f'monkey {cache.region}')
                for mode in ('time_mean', 'flat'):
                    cache._get_pattern(mode, t_common, pca_k=K)
            else:
                for mode in ('time_mean', 'flat'):
                    cache._get_pattern(mode, t_common, pca_k=0)
        print(f'Precompute done; smoothed-activity/PCA/pattern caches under '
              f'{monkey_cache_dir}/')
        return

    print(f'Loading model seed {seed} ...')
    model = load_model(model_path)
    print('  hidden shape:', model['hidden'].shape,
          ' base_hidden shape:', model['base_hidden'].shape,
          ' suc trials:', int(model['keep'].sum()))

    # -----------------------------------------------------------------------
    # 3. Build the per-condition slicing window for the model (the monkey
    #    window is internal to MonkeyRDMCache).
    #
    #    Model: the last (|raw_d|+1) timesteps of the trial are valid; the
    #           preceding timesteps are NaN padding.
    # -----------------------------------------------------------------------
    T_model = model['hidden'].shape[1]
    def model_window(c):
        """Return the (start, end) timestep window for condition ``c`` in model time.

        ``c`` is either a signed int (distance mode) or a ``(curr, target)``
        tuple (pair mode). Either way the window length is set by the
        absolute distance the agent has to navigate; the window is
        right-aligned to ``T_model``.

        Args:
            c: Condition key (signed int or ``(curr, target)`` tuple).

        Returns:
            (int, int): Inclusive ``(start, end)`` timestep indices.
        """
        n_valid = label_distance(c) * model['dist_unit'] + 1
        return (T_model - n_valid, T_model - 1)

    # Per-trial condition index for the model (matches `conditions` list).
    model_cond_idx = compute_cond_idx(model['curr'], model['target'],
                                      conditions)
    valid_trials = (model_cond_idx >= 0) & model['keep']
    print(f'model trials with valid condition: {int(valid_trials.sum())} '
          f'/ {len(model_cond_idx)}')

    # Model is smoothed *after* slicing because the raw tensor has NaN-padded
    # leading timesteps that gaussian_filter1d would propagate. Smoothing is
    # opt-in via --smoothing; without it the model RDMs are built from raw
    # (unsmoothed) hidden states.
    if args.smoothing:
        model_smooth = {'sigma':    model['smooth_sigma'],
                        'truncate': model['smooth_truncate'],
                        'mode':     'nearest'}
    else:
        model_smooth = None
    print(f'model smoothing: '
          f'{"on (sigma=" + str(model["smooth_sigma"]) + ")" if args.smoothing else "off"}')

    # Model-side systems only; monkey systems come from the cache.
    model_systems = {
        'model hidden (PPC-like)': dict(
            activity=model['hidden'], cond_idx=model_cond_idx,
            keep=valid_trials,
            window=model_window, smooth=model_smooth),
        'model base_hidden (EC-like)': dict(
            activity=model['base_hidden'], cond_idx=model_cond_idx,
            keep=valid_trials,
            window=model_window, smooth=model_smooth),
    }

    # Resolve per-system K for the monkey RDMs by pulling the cached PCA
    # basis (no-op if PCA is off). Threshold-driven K varies across systems
    # (7a vs EC vs each model layer), so we store K per system rather than
    # one global value.
    monk_k = {'7a': 0, 'EC': 0}
    if pca_on:
        print('Resolving monkey-side K from cached PCA bases:')
        monk_k['7a'] = resolve_k(monk_7a_cache.get_pca(), label='monkey 7a')
        monk_k['EC'] = resolve_k(monk_ec_cache.get_pca(), label='monkey EC')

    # Optional PCA projection of the model activity. Per-system: each model
    # layer fits its own basis (the two layers can live in very different
    # subspaces). We follow the same recipe used for the monkey side: fit
    # PCA(n=pca_components) on the trial-averaged time-resolved window-sliced
    # activity, pick K per-system (fixed or via EVR threshold), project the
    # full activity tensor to top-K, and replace `activity` for downstream
    # pattern construction.
    model_k = {name: 0 for name in model_systems}
    if pca_on:
        print(f'PCA: fitting per-system basis ({args.pca_components} comps).')
        for name, info in model_systems.items():
            pca = fit_pca_basis(
                info['activity'], info['cond_idx'], info['keep'],
                conditions=conditions,
                valid_window=info['window'], smooth=info['smooth'],
                n_components=args.pca_components)
            K = resolve_k(pca, label=name)
            model_k[name] = K
            info['activity'] = project_activity(info['activity'], pca, K)
            # Note: smoothing is left in info['smooth']. Smoothing and PCA
            # projection are both linear, so smooth(project(x)) ==
            # project(smooth(x)); we project the raw activity (preserves
            # NaN padding for slicing) and let condition_patterns smooth
            # the K-dim trace per condition.

    # -----------------------------------------------------------------------
    # 4. For each pattern-construction mode ('time_mean' / 'flat'), grab the
    #    monkey RDMs from cache, compute the model RDMs fresh, then save the
    #    plots / summary.
    # -----------------------------------------------------------------------
    summary_lines = []
    for mode in ('time_mean', 'flat'):
        # rdms[name] = (n_cond, n_cond) RDM under the single chosen metric.
        rdms = {
            'monkey 7a (PPC)': monk_7a_cache.get_rdm(
                mode, t_common, metric, pca_k=monk_k['7a']),
            'monkey EC':       monk_ec_cache.get_rdm(
                mode, t_common, metric, pca_k=monk_k['EC']),
        }
        # Model: depends on the seed, so always (re)computed.
        for name, info in model_systems.items():
            P = condition_patterns(info['activity'], info['cond_idx'],
                                   info['keep'],
                                   conditions=conditions,
                                   mode=mode,
                                   T_common=t_common,
                                   valid_window=info['window'],
                                   smooth=info['smooth'])
            rdms[name] = compute_rdm(P, metric=metric)
            print(f"[{mode}] {name}: pattern shape = {P.shape}, "
                  f"any-nan-pat={np.isnan(P).any()}")

        # File-name suffix that disambiguates this run on disk.
        suffix = f'_{mode}_{tag}'

        # 4a/4b. Per-seed RDM tensor npz + rdm_grid / rdm_cross heatmap PDFs
        # used to land here. Dropped: only the aggregate perspective plots
        # (EC/PPC + specificity boxes) get read downstream, and 50-seed sweeps
        # were dumping ~250 unused per-seed artifacts per RDM tag.

        # 4c. Headline numbers for the four PPC/EC pairings under the chosen
        #     metric. Also saved as a small npz that the aggregator over many
        #     seeds reads back.
        summary_lines.append(f'\n=== mode = {mode} ===')
        ppc    = rdm_spearman(rdms['monkey 7a (PPC)'],
                              rdms['model hidden (PPC-like)'])
        ec     = rdm_spearman(rdms['monkey EC'],
                              rdms['model base_hidden (EC-like)'])
        cross1 = rdm_spearman(rdms['monkey 7a (PPC)'],
                              rdms['model base_hidden (EC-like)'])
        cross2 = rdm_spearman(rdms['monkey EC'],
                              rdms['model hidden (PPC-like)'])
        summary_lines.append(
            f"  metric={metric:11s}  "
            f"7a~hidden={ppc:+.3f}  ec~base={ec:+.3f}  "
            f"7a~base={cross1:+.3f}  ec~hidden={cross2:+.3f}"
        )
        np.savez(
            os.path.join(save_dir, f'pair_scores{suffix}.npz'),
            seed=np.array(seed), t_common=np.array(t_common),
            mode=np.array(mode), metric=np.array(metric),
            smoothing=np.array(int(args.smoothing)),
            cond_mode=np.array(args.conditions),
            n_cond=np.array(len(conditions)),
            pca_use=np.array(int(args.pca_use)),
            pca_evr=np.array(args.pca_evr if args.pca_evr is not None else 0.0),
            pca_components=np.array(int(args.pca_components)),
            pca_k_7a=np.array(int(monk_k['7a'])),
            pca_k_ec=np.array(int(monk_k['EC'])),
            pca_k_hidden=np.array(
                int(model_k['model hidden (PPC-like)'])),
            pca_k_base_hidden=np.array(
                int(model_k['model base_hidden (EC-like)'])),
            ppc_ppc=np.array(ppc), ec_ec=np.array(ec),
            ppc_ec_cross=np.array(cross1), ec_ppc_cross=np.array(cross2),
        )

    # -----------------------------------------------------------------------
    # 5. Print and save the headline summary.
    # -----------------------------------------------------------------------
    summary = (f'T_common = {t_common}\n' + '\n'.join(summary_lines))
    print(summary)
    with open(os.path.join(save_dir, f'rdm_summary_{tag}.txt'), 'w') as f:
        f.write(summary + '\n')
    print(f'\nSaved to {save_dir}/  (tag={tag})')


if __name__ == '__main__':
    main()
