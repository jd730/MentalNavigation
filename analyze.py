"""General analysis driver for any Vector-HaSH sweep.

Broadened copy of analyze_relu.py: same per-paper analysis suite, but
model-agnostic. Point it at any sweep directory (default
``decoder_dir_mental/vHMN2_period``) and it auto-detects which models
are present from the run subdir prefixes (VHA-D, VHA-ReLU, VHA-ReLU-D,
RNN, RNN-D, ...).

Analyses (reusing existing modules):

    * :func:`lib.analysis.run_dir`        -- per-seed autocorr / firing-rate /
                                             regression / neurons figures
    * :func:`lib.analyze_utils.collect.post_analysis4periodicity`
                                          -- non_zeros / periodicity_at_LM /
                                             ratio bar plots, split by model
    * :func:`lib.analyze_utils.collect.draw_aggregated_periodicity`
                                          -- aggregated periodicity KDEs
    * ``rdm_analysis.py``                 -- monkey EC / PPC vs model RDMs

Speed-of-learning: computed here from ``log.csv`` (first epoch to reach
0.8 accuracy per seed).

Usage:
    # Auto-detect models under a sweep dir
    python analyze.py -dirname decoder_dir_mental/vHMN2_period

    # Restrict to a specific subset of models
    python analyze.py -dirname decoder_dir_mental/ReLU -models VHA-ReLU-D

    # Skip heavy blocks
    python analyze.py -dirname ... -skip rdm xcorr pca
"""
import argparse
import glob
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

from lib.analysis import ANALYSES, run_dir
from lib.analyze_utils.collect import (
    read_results,
    calculate_average,
    export_run_info_to_mat,
    draw_generalization,
    post_analysis4periodicity,
    draw_aggregated_periodicity,
)
from lib.analyze_utils.pca import plot_pca_for_sweep


SWEEP_DIR = 'decoder_dir_mental/vHMN2_period'
SAVE_DIR = 'results/analyze'
# Legacy (pre-refactor) sweeps that already exist on disk. Their arguments.json
# has no `model` key, so we tag them here by source directory.
LEGACY_BASELINE_DIRS = {
    'RNN':   'decoder_dir_mental/jaed/vRNN_log100',   # single CTRNN, no grid cells (51 seeds)
    'RNN-D': 'decoder_dir_mental/vRNN2',              # double CTRNN, no grid cells (50 seeds)
}
BASELINE_MODELS = list(LEGACY_BASELINE_DIRS.keys())
# Models the newer -model dispatch can emit. Auto-detection at runtime
# still narrows this to whatever prefixes are actually present under the
# sweep dir; this list only bounds what we recognize.
KNOWN_MODELS = ['VHA-D', 'VHA-ReLU', 'VHA-ReLU-D', 'RNN-D']
# MODELS gets rebound in main() to the subset actually present (or the
# user's -models arg). Kept as a module-level placeholder so per_seed
# helpers that iterate MODELS keep working when imported.
MODELS = KNOWN_MODELS
ALL_MODELS = BASELINE_MODELS + KNOWN_MODELS
ACC_KEY = 'visual/all/0/accuracy'
THRESHOLD = 0.8
# Paper convention: analyze the first-env checkpoint (before adaptation).
# export_run_info_to_mat also targets 001999 for RDM .mat exports.
ANALYSIS_EPOCH = '001999'
# Paper smoothing convention (rdm.load_model): sigma=4, truncate=2 for model activity.
# ``separate_direction`` for firing_rate matches _firing_rate_run in registry.py:
# writes indivleft/ and indivright/ per-neuron PDFs when save_indiv=True.
SMOOTHING_OPTIONS = {
    'firing_rate': {
        'smoothing_sigma': 4, 'smoothing_truncate': 2,
        'separate_direction': True,
    },
    'neurons':     {'smoothing_sigma': 4, 'smoothing_truncate': 2},
}


def _model_from_name(run_name):
    """Return the model prefix (e.g. 'VHA-ReLU-D', 'VHA-D') from a run-dir name.

    Newer save-dir names follow ``<model>_N<num_images>L<lambdas>...``, so the
    model is everything before the first ``_N``. Falls back to the leading
    ``_``-separated token when the ``_N`` marker isn't present.
    """
    m = re.match(r'^([A-Za-z0-9-]+)_N\d', run_name)
    if m:
        return m.group(1)
    return run_name.split('_', 1)[0]


def _model_constraint(model):
    """Substring that matches ``<model>`` and excludes longer prefixes.

    We match ``<model>_`` so ``VHA-D`` doesn't spuriously match VHA-ReLU-D
    runs; the trailing underscore forces the boundary.
    """
    return f'{model}_'


def _detect_models(dirname, allowed=None):
    """Return the KNOWN_MODELS present under ``dirname`` (sorted).

    ``allowed`` optionally restricts detection to a caller-supplied subset
    (typically from a ``-models`` CLI arg).
    """
    universe = set(allowed) if allowed else set(KNOWN_MODELS)
    present = set()
    if os.path.isdir(dirname):
        for name in os.listdir(dirname):
            m = _model_from_name(name)
            if m in universe:
                present.add(m)
    return sorted(present)


def load_all_logs(dirname, seeds=None):
    """Combined log DataFrame across ReLU + legacy RNN/RNN-D baselines.

    Adds a `model` column. RNN/RNN-D come from the pre-refactor sweep dirs
    listed in LEGACY_BASELINE_DIRS; ReLU variants come from `dirname`. If a
    legacy dir is missing we silently skip that model.
    """
    dfs = []
    # Models from the current sweep dir. MODELS is set at CLI time via
    # _detect_models(); at import time it falls back to KNOWN_MODELS.
    for m in MODELS:
        d = read_results(dirname, [_model_constraint(m)], seeds=seeds)
        if len(d) == 0:
            print(f'no runs found for {m} under {dirname}; skipping')
            continue
        d['model'] = m
        dfs.append(d)
    # Legacy baselines from their pre-refactor sweep dirs.
    for m, legacy_dir in LEGACY_BASELINE_DIRS.items():
        if not os.path.isdir(legacy_dir):
            print(f'legacy baseline dir {legacy_dir} missing — skipping {m}')
            continue
        d = read_results(legacy_dir, [], seeds=seeds)
        if len(d) == 0:
            print(f'no runs found for {m} under {legacy_dir}; skipping')
            continue
        # Drop stray non-canonical epoch checkpoints. vRNN_log100/S12 was
        # rerun at Ep5000 (all other seeds use Ep2000); the extra rows land
        # at epoch=5000, 6000, ... with only one seed contributing and
        # create sharp mean spikes in the paired 3-2 plot near Env3.
        if 'name' in d.columns:
            bad = d['name'].str.contains('Ep5000E256', regex=False, na=False)
            if bad.any():
                n = int(bad.sum())
                print(f'{m}: dropping {n} rows from non-canonical Ep5000 checkpoint(s)')
                d = d[~bad]
        d['model'] = m
        dfs.append(d)
    if not dfs:
        raise SystemExit(f'no runs found under {dirname}')
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# 1. Aggregate log.csv: final success rate + speed-to-threshold + learning curve
# ---------------------------------------------------------------------------

GEN_CONDS = [('visual', 'seen'), ('visual', 'unseen'),
             ('mental', 'seen'), ('mental', 'unseen')]
GEN_ENVS = [0, 1, 2]
MODEL_COLORS = {
    'RNN':        '#7f7f7f',   # neutral grey
    'RNN-D':      '#4a4a4a',   # darker grey
    'VHA':        '#33669a',   # blue  (canonical single-CTRNN)
    'VHA-D':      '#d95527',   # orange (canonical double-CTRNN, paper hero)
    'VHA-ReLU':   '#33669a',
    'VHA-ReLU-D': '#d95527',
}
MODEL_LINESTYLES = {
    'RNN':        '--',
    'RNN-D':      '--',
    'VHA':        'solid',
    'VHA-D':      'solid',
    'VHA-ReLU':   'solid',
    'VHA-ReLU-D': 'solid',
}


def _acc_col(mode, cond, env):
    return f'{mode}/{cond}/{env}/accuracy'


def aggregate_accuracy(dfs, save_dir):
    """Write final-success + per-condition learning curves + speed-to-threshold.

    The four generalization axes are drawn separately (mode x seen/unseen)
    for each of the three environments, matching the paper's convention.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---- Final success at last epoch: per (mode, seen/unseen, env) x model ----
    last_rows = dfs.sort_values('epoch').groupby(['model', 'seed']).tail(1)
    final_records = []
    for _, row in last_rows.iterrows():
        for mode, cond in GEN_CONDS:
            for env in GEN_ENVS:
                col = _acc_col(mode, cond, env)
                if col in row.index:
                    final_records.append({
                        'model': row['model'], 'seed': row['seed'],
                        'mode': mode, 'cond': cond, 'env': env,
                        'final_acc': row[col],
                    })
    final_df = pd.DataFrame(final_records)
    final_df.to_csv(os.path.join(save_dir, 'final_success.csv'), index=False)

    summary = (final_df.groupby(['model', 'mode', 'cond', 'env'])['final_acc']
                       .agg(['mean', 'std', 'count'])
                       .reset_index())
    summary.to_csv(os.path.join(save_dir, 'final_success_summary.csv'), index=False)
    print('\n=== Final success rate (per condition) ===')
    print(summary.to_string(index=False))

    # ---- Learning curves: paper's canonical draw_generalization output ----
    # For each model, produces generalization{1, 2, 3-1, 3-2}.pdf under
    # <save_dir>/generalization/<model>/ using the exact code the paper uses.
    #     generalization1 : Visual vs Mental (env 0)
    #     generalization2 : Seen  vs Unseen  (env 0)
    #     generalization3-1 : all 3 envs
    #     generalization3-2 : env 0 only variant
    # epoch=2000 matches the per-env epochs in lib/arguments.py; the full run
    # is 3 envs x 2000 trials so draw_generalization3 uses this as the segment
    # width for the tick / vline placement.
    for m in ALL_MODELS:
        sub = dfs[dfs['model'] == m]
        if len(sub) == 0:
            continue
        out = os.path.join(save_dir, 'generalization', m)
        os.makedirs(out, exist_ok=True)
        draw_generalization(None, out, epoch=2000, _format='pdf',
                            data=[sub],
                            linestyles=[MODEL_LINESTYLES.get(m, 'solid')],
                            colors=[MODEL_COLORS.get(m)])

    # Paper-style pairwise baseline-vs-ours comparisons. linestyles=
    # ['dashed', 'solid'] with baseline drawn first. Two pair-ups:
    #   * RNN   (single-CTRNN baseline) vs VHA-ReLU   (single-CTRNN, ours)
    #   * RNN-D (double-CTRNN baseline) vs VHA-ReLU-D (double-CTRNN, ours)
    BASELINE_PAIRS = [('RNN', 'VHA-ReLU'), ('RNN-D', 'VHA-ReLU-D')]
    for baseline, ours in BASELINE_PAIRS:
        b = dfs[dfs['model'] == baseline]
        o = dfs[dfs['model'] == ours]
        if len(b) == 0 or len(o) == 0:
            continue
        # Legacy RNN sweeps ran to 15000 epochs (5x the 3x2000 window our
        # Env1/Env2/Env3 labels assume). Truncate both datasets to the
        # canonical 3-env window so gen3's dashed line doesn't run past
        # Env3. Don't pass options={'xlim': ...}: generalization1/2 default
        # to plt.xlim(0, epochs) = (0, 2000) so those panels stay cropped
        # to env1; overriding here would show all three envs, which the
        # gen1/gen2 axis labels don't support.
        max_epoch = 3 * 2000
        b = b[b['epoch'] <= max_epoch]
        o = o[o['epoch'] <= max_epoch]
        out = os.path.join(save_dir, 'generalization', f'{baseline}_vs_{ours}')
        os.makedirs(out, exist_ok=True)
        draw_generalization(
            None, out, epoch=2000, _format='pdf',
            data=[b, o],
            linestyles=['dashed', 'solid'],
            colors=None,  # None -> draw_generalization uses black+red per dataset
        )

    # ---- Speed-to-threshold: env-0, seen/unseen aggregated ----
    # For each (model, seed) row group in dfs, pick the earliest epoch
    # whose visual/all/0/accuracy (calculate_average's 0.8*seen + 0.2*unseen
    # mean) >= THRESHOLD. One row per seed per model; env 1/2 are dominated
    # by log-interval jitter after the env boundary so aren't worth
    # reporting.
    def _first_iter(group):
        reached = group[group[ACC_KEY] >= THRESHOLD]
        return reached['epoch'].iloc[0] if not reached.empty else np.inf

    speed_wide = (dfs.groupby(['model', 'seed'])
                     .apply(_first_iter)
                     .reset_index(name='trial_to_thresh')
                     .pivot(index='seed', columns='model', values='trial_to_thresh'))
    speed_wide.to_csv(os.path.join(save_dir, 'speed_to_threshold.csv'))

    speed_summary = (speed_wide.replace(np.inf, np.nan)
                               .agg(['mean', 'std', 'count'])
                               .T
                               .reset_index()
                               .rename(columns={'index': 'model'}))
    speed_summary.to_csv(os.path.join(save_dir, 'speed_to_threshold_summary.csv'),
                         index=False)
    print(f'\n=== Trials to reach visual/all/0/accuracy >= {THRESHOLD:.2f} ===')
    print(speed_summary.to_string(index=False))

    # Paired Wilcoxon: RNN vs VHA-ReLU and RNN-D vs VHA-ReLU-D on env-0 speed.
    BASELINE_PAIRS = [('RNN', 'VHA-ReLU'), ('RNN-D', 'VHA-ReLU-D')]
    wilcox_rows = []
    for baseline, ours in BASELINE_PAIRS:
        if baseline not in speed_wide or ours not in speed_wide:
            continue
        pair = speed_wide[[baseline, ours]].replace(np.inf, np.nan).dropna()
        if len(pair) < 5 or (pair[baseline] == pair[ours]).all():
            continue
        stat, p = wilcoxon(pair[baseline], pair[ours])
        wilcox_rows.append({
            'baseline': baseline, 'ours': ours,
            'stat': stat, 'p': p, 'N': len(pair),
            f'{baseline}_mean': pair[baseline].mean(),
            f'{ours}_mean': pair[ours].mean(),
        })
    wilcox_df = pd.DataFrame(wilcox_rows)
    wilcox_df.to_csv(os.path.join(save_dir, 'speed_wilcoxon.csv'), index=False)
    if len(wilcox_df):
        print('\n=== Wilcoxon paired speed comparisons ===')
        print(wilcox_df.to_string(index=False))

    # ---- Box plot per model (env-0 speed, seen/unseen aggregated) ----
    speed_long = (speed_wide.replace(np.inf, np.nan)
                            .reset_index()
                            .melt(id_vars='seed', var_name='model',
                                  value_name='trial_to_thresh')
                            .dropna())
    ylabel = f'Trials to reach {int(THRESHOLD*100)}% accuracy (visual/all/env0)'

    # Box plot: non-baseline models only (drop RNN / RNN-D baselines); boxes
    # in neutral grey since this panel is about the model variants, not the
    # baseline comparison.
    non_baseline = [m for m in KNOWN_MODELS if m in speed_long['model'].unique()]
    box_order = non_baseline
    box_data = speed_long[speed_long['model'].isin(box_order)]
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=200)
    sns.boxplot(data=box_data, x='model', y='trial_to_thresh', order=box_order,
                color='lightgray', width=0.4, ax=ax)
    sns.stripplot(data=box_data, x='model', y='trial_to_thresh', order=box_order,
                  color='k', size=3, alpha=0.5, ax=ax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    plt.xticks(rotation=0, ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'speed_box.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Per-seed periodicity analyses (autocorr writes the periodicity/*.npy files
#    that post_analysis4periodicity aggregates in step 3)
# ---------------------------------------------------------------------------

DEFAULT_TARGETS = ('hidden', 'base_hidden')


def per_seed_periodicity(dirname, save_dir, seeds, focus_seed, targets=DEFAULT_TARGETS,
                        force=False, epoch=None):
    """Call run_dir(...) per target for the periodicity npy dumps.

    Only the two CTRNN layers are analyzed by default -- the grid-cell code
    ('gs') is not periodicity-relevant. .mat exports still include gs via
    export_run_info_to_mat for downstream RDM / decoding work.
    For every seed we compute autocorr (produces the periodicity/*.npy that
    post_analysis4periodicity reads) + firing_rate + regression. Single-neuron
    figures ('neurons') are heavy so we only produce them for the focus seed.
    """
    force_set = set(ANALYSES.keys()) if force else set()
    # post_analysis4periodicity / draw_aggregated_periodicity glob under
    # <save_dir>/dynamics/... so we anchor the per-seed run_dir outputs there.
    dynamics_dir = os.path.join(save_dir, 'dynamics')
    for target in targets:
        # Cross-seed: cheap analyses only (drops 'neurons' since it writes one
        # figure per neuron). Smoothing enabled on firing_rate.
        print(f'\n--- run_dir target={target} (all seeds, epoch={ANALYSIS_EPOCH}) ---')
        run_dir(dirname=dirname, save_dir=dynamics_dir, target=target,
                analyses=['autocorr', 'firing_rate', 'regression'],
                force=force_set,
                seeds=seeds, visual_only=True, use_cuda=True, verbose=True,
                epoch=epoch or ANALYSIS_EPOCH,
                options=SMOOTHING_OPTIONS)

        # Focus-seed heavies: neurons (NxN grid) + firing_rate with
        # save_indiv=True (per-neuron aggregate PDFs at indivleft/, indivright/
        # subdirs, both onset+offset). firing_rate_each_trial (the 6x6 grid
        # with inferno colors) is intentionally skipped -- not used in the paper.
        if focus_seed is not None:
            print(f'\n--- run_dir target={target} focus-seed (seed={focus_seed}, '
                  f'epoch={ANALYSIS_EPOCH}) ---')
            run_dir(dirname=dirname, save_dir=dynamics_dir, target=target,
                    analyses=['firing_rate', 'neurons'],
                    force=force_set,
                    seeds=[focus_seed], visual_only=True,
                    save_indiv=True, use_cuda=True, verbose=True,
                    epoch=ANALYSIS_EPOCH,
                    options=SMOOTHING_OPTIONS)


# ---------------------------------------------------------------------------
# 3. Post-analysis: periodicity_at_LM / non_zeros / ratio bars, per model
# ---------------------------------------------------------------------------

_BAR_KEEP_NAMES = [
    'non_zeros_box.pdf', 'periodicity_at_LM_box.pdf',
]
_BAR_DROP_NAMES = [
    'non_zeros.pdf', 'periodicity_at_LM.pdf', 'ratio.pdf',
    'non_zeros_with_EC.pdf', 'periodicity_at_LM_with_EC.pdf', 'ratio_with_EC.pdf',
    'non_zeros_violin.pdf', 'periodicity_at_LM_violin.pdf', 'ratio_violin.pdf',
]


def post_periodicity_bars(dfs, save_dir):
    """One post_analysis4periodicity pass per model, moved into per-model subdir.

    post_analysis4periodicity globs ``<save_dir>/dynamics/...`` so we must
    pass the top save_dir (where dynamics/ lives). It writes bar PDFs at
    the same top level, so after each per-model call we move the outputs
    into a per-model subdir to keep VHA-ReLU and VHA-ReLU-D distinct.
    """
    for m in MODELS:
        sub = dfs[dfs['model'] == m]
        if len(sub) == 0:
            continue
        out = os.path.join(save_dir, 'periodicity_bars', m)
        os.makedirs(out, exist_ok=True)
        try:
            post_analysis4periodicity(sub, save_dir, [_model_constraint(m)])
        except Exception as e:  # noqa: BLE001
            # post_analysis4periodicity has a stray breakpoint() at its tail;
            # the interesting figures are all written before that point.
            print(f'post_analysis4periodicity for {m} raised: {e}')
        # Keep only the box variant; drop bar + violin.
        for name in _BAR_KEEP_NAMES:
            src = os.path.join(save_dir, name)
            if os.path.exists(src):
                os.replace(src, os.path.join(out, name))
        for name in _BAR_DROP_NAMES:
            src = os.path.join(save_dir, name)
            if os.path.exists(src):
                os.remove(src)
        # The function also writes copies to cwd; drop everything it left.
        for name in ['non_zeros_violin.png', 'non_zeros_violin.pdf',
                     'non_zeros_box.png', 'non_zeros_box.pdf',
                     'periodicity_at_LM_violin.png', 'periodicity_at_LM_violin.pdf',
                     'periodicity_at_LM_box.png', 'periodicity_at_LM_box.pdf',
                     'ratio_violin.png', 'ratio_violin.pdf',
                     'ratio_box.png', 'ratio_box.pdf']:
            if os.path.exists(name):
                os.remove(name)


def aggregated_periodicity(dfs, save_dir, focus_seed):
    """draw_aggregated_periodicity across all seeds, once per model.

    Deliberately does NOT pass target_seeds=[focus_seed] -- the paper's
    population-level distinction (dist RNN more periodic-at-landmark than
    action RNN) is only clear in the aggregate; single-seed figures can
    look atypical (e.g. seed 43 does not show the expected direction).
    """
    for m in MODELS:
        sub = dfs[dfs['model'] == m]
        if len(sub) == 0:
            continue
        seeds = sorted(sub['seed'].unique().tolist())
        out = os.path.join(save_dir, 'agg_periodicity', m)
        os.makedirs(out, exist_ok=True)
        for target in ['visual__0_hidden', 'visual__0_base_hidden']:
            try:
                draw_aggregated_periodicity(
                    seeds, save_dir, constraint=[_model_constraint(m)],
                    target=target, save_dir=out, _format='pdf',
                    resolutions=[], mode='kde',
                    target_seeds=[],
                )
            except Exception as e:  # noqa: BLE001
                print(f'draw_aggregated_periodicity {m}/{target} raised: {e}')


# ---------------------------------------------------------------------------
# 3.5. .mat export -- rdm_analysis.py reads *.mat, not *.pth
# ---------------------------------------------------------------------------

def export_mats(dirname, mat_out_dir):
    """Convert run_info_*.pth -> .mat via export_run_info_to_mat.

    export_run_info_to_mat walks a dirname, matches on substrings, and writes one
    .mat per seed to `mat_out_dir`. We call it twice (once per model) so both
    VHA-ReLU-D (used by RDM) and VHA-ReLU (comparison) end up on disk.
    """
    os.makedirs(mat_out_dir, exist_ok=True)
    for m in MODELS:
        constraint = [_model_constraint(m)]
        print(f'\n--- export_run_info_to_mat {m} -> {mat_out_dir} ---')
        try:
            export_run_info_to_mat(dirname, constraint, mat_out_dir)
        except Exception as e:  # noqa: BLE001
            print(f'export_run_info_to_mat for {m} raised: {e}')


# ---------------------------------------------------------------------------
# 3.6. RDM post-aggregation perspective plots (EC and PPC separately)
# ---------------------------------------------------------------------------

def _rdm_perspective_plots(rdm_root):
    """For every aggregated_pair_scores_*.csv on disk, emit two boxplots:

    * EC perspective:  paired rho(EC, distanceRNN)  vs rho(EC, actionRNN)
                       -> is EC more similar to distanceRNN?  (positive Δ_EC)
    * PPC perspective: paired rho(7a, actionRNN)    vs rho(7a, distanceRNN)
                       -> is PPC more similar to actionRNN?   (positive Δ_PPC)

    One PDF each, sitting beside the aggregated CSV so you can compare
    variants (raw / pca3 / pcaEVR80 / cosine / pair) at a glance.
    """
    import scipy.stats as st  # noqa: WPS433  (kept local; used only here)
    csvs = sorted(glob.glob(os.path.join(rdm_root, '*', '*',
                                         'aggregated_pair_scores_*.csv')))
    for csv in csvs:
        df = pd.read_csv(csv)
        variant_dir = os.path.dirname(csv)
        tag = re.sub(r'^aggregated_pair_scores_', '',
                     os.path.basename(csv).replace('.csv', ''))
        for mode in sorted(df['mode'].unique()):
            sub = df[df['mode'] == mode].copy()

            # Build long-form frames per perspective for seaborn.
            long_ec = pd.concat([
                sub.assign(layer='distanceRNN', rho=sub['ec_ec']),
                sub.assign(layer='actionRNN',   rho=sub['ec_ppc_cross']),
            ], ignore_index=True)
            long_ppc = pd.concat([
                sub.assign(layer='actionRNN',   rho=sub['ppc_ppc']),
                sub.assign(layer='distanceRNN', rho=sub['ppc_ec_cross']),
            ], ignore_index=True)

            # Paired Wilcoxon (does the "expected" layer win?):
            #   EC:  ec_ec  > ec_ppc_cross  <=> distanceRNN preferred by EC
            #   PPC: ppc_ppc > ppc_ec_cross <=> actionRNN preferred by PPC
            def _wilcox(a, b):
                try:
                    stat, p = st.wilcoxon(a, b, alternative='greater')
                    return stat, p, len(a)
                except Exception:
                    return float('nan'), float('nan'), len(a)
            ec_stat, ec_p, n_ec = _wilcox(sub['ec_ec'], sub['ec_ppc_cross'])
            ppc_stat, ppc_p, n_ppc = _wilcox(sub['ppc_ppc'], sub['ppc_ec_cross'])

            for label, long_df, order, subtitle, stat, p, n in [
                ('EC',  long_ec,  ['distanceRNN', 'actionRNN'],
                 'monkey EC vs model layer  '
                 '(is EC more like distanceRNN?)',  ec_stat, ec_p, n_ec),
                ('PPC', long_ppc, ['distanceRNN', 'actionRNN'],
                 'monkey PPC (7a) vs model layer  '
                 '(is PPC more like actionRNN?)',   ppc_stat, ppc_p, n_ppc),
            ]:
                # Keep the per-layer color mapping stable across both panels
                # (distanceRNN blue, actionRNN orange) regardless of visual order.
                LAYER_COLORS = {'distanceRNN': '#33669a', 'actionRNN': '#d95527'}
                fig, ax = plt.subplots(figsize=(3.5, 3.2), dpi=200)
                sns.boxplot(data=long_df, x='layer', y='rho', order=order,
                            palette=[LAYER_COLORS[layer] for layer in order],
                            width=0.55, ax=ax)
                sns.stripplot(data=long_df, x='layer', y='rho', order=order,
                              color='k', size=2.2, alpha=0.55, jitter=0.15, ax=ax)
                ax.set_ylabel('Spearman ρ')
                ax.set_xlabel('')
                ax.set_title(f'{subtitle}\nWilcoxon (greater): '
                             f'stat={stat:.0f}, p={p:.2e}, N={n}',
                             fontsize=8)
                plt.tight_layout()
                out = os.path.join(variant_dir,
                                   f'{label}_perspective_{mode}_{tag}.pdf')
                fig.savefig(out)
                plt.close(fig)


MATLAB_BIN = '/orcd/software/core/001/pkg/matlab/R2025b/bin/matlab'
XCORR_MATLAB_SCRIPT = 'scripts/matlab_xcorr_all_seeds.m'


def _run_matlab_xcorr(mat_dir, out_root):
    """Invoke scripts/matlab_xcorr_all_seeds.m over ``mat_dir``.

    The MATLAB script reads paths from ``XCORR_MAT_DIR`` and
    ``XCORR_OUT_ROOT`` env vars (see the getenv block at the top of the .m
    file); we just forward the pipeline's canonical directories so the
    same code works for any sweep, not only the ReLU/ dir the script's
    hardcoded fallback assumes.
    """
    mat_dir_abs = os.path.abspath(mat_dir)
    out_root_abs = os.path.abspath(out_root)
    os.makedirs(out_root_abs, exist_ok=True)
    env = os.environ.copy()
    env['XCORR_MAT_DIR'] = mat_dir_abs
    env['XCORR_OUT_ROOT'] = out_root_abs
    repo_root = os.path.dirname(os.path.abspath(__file__))
    cmd = [MATLAB_BIN, '-batch',
           f"cd('{repo_root}'); run('{XCORR_MATLAB_SCRIPT}')"]
    # Stream MATLAB's output live so the pipeline log matches the .m fprintf.
    subprocess.run(cmd, env=env, check=True)


# ---------------------------------------------------------------------------
# 4. RDM: monkey EC / PPC vs model, seed 43 primary
# ---------------------------------------------------------------------------

def _seeds_with_full_conditions(mat_dir, seeds):
    """Return the subset of seeds whose VHA-ReLU-D .mat has all 10 signed
    distance conditions on successful trials -- required by
    rdm_analysis.condition_patterns.
    """
    import scipy.io  # local import to avoid cost when RDM step is skipped
    target = set([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
    ok = []
    for s in seeds:
        path = glob.glob(os.path.join(mat_dir, f'VHA-ReLU-D_*S{s}Ep*.mat'))
        if not path:
            continue
        m = scipy.io.loadmat(path[0])
        suc = m['suc'].squeeze().astype(bool)
        trajs = m['trajs']
        dist = ((trajs[:, 1] - trajs[:, 0]) / 12).astype(int)
        if target.issubset(set(dist[suc].tolist())):
            ok.append(s)
    return ok


def run_rdm(focus_seed, extra_seeds, save_dir, mat_dir):
    """Invoke rdm_analysis.py per seed, then aggregate across seeds.

    ``focus_seed`` is the primary seed; if its .mat is missing distance
    conditions we transparently drop it. All seeds passed in are also
    checked for coverage. After per-seed runs, invoke rdm_aggregate.py to
    produce the cross-seed distribution + specificity summary.
    """
    rdm_dir = os.path.join(save_dir, 'rdm')
    os.makedirs(rdm_dir, exist_ok=True)

    requested = list({focus_seed, *(extra_seeds or [])})
    if not requested:
        requested = [focus_seed]
    valid = _seeds_with_full_conditions(mat_dir, requested)
    dropped = [s for s in requested if s not in valid]
    if dropped:
        print(f'RDM: dropping seeds with incomplete distance coverage: {dropped}')
    if not valid:
        print(f'RDM: no seeds with full coverage among {requested}; skipping.')
        return

    def _run(seed, smoothing):
        cmd = [
            sys.executable, 'rdm_analysis.py',
            '--seed', str(seed),
            '--save_dir', rdm_dir,
            '--model_glob',
            os.path.join(mat_dir, f'VHA-ReLU-D_*S{seed}Ep*.mat'),
        ]
        if smoothing:
            cmd.append('--smoothing')
        print('$', ' '.join(cmd))
        proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) or '.')
        if proc.returncode != 0:
            print(f'rdm_analysis.py exited {proc.returncode} for seed={seed} '
                  f'smoothing={smoothing}')

    for s in valid:
        _run(s, smoothing=True)
    # Also unsmoothed for the primary seed as a comparison baseline.
    _run(valid[0], smoothing=False)

    # Aggregate across seeds if we have >1.
    if len(valid) >= 2:
        cmd = [sys.executable, 'rdm_aggregate.py',
               '--save_dir', rdm_dir, '--smoothing']
        print('$', ' '.join(cmd))
        subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) or '.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('-dirname', default=SWEEP_DIR,
                   help='Root sweep directory (default: %(default)s). RNN '
                        'and RNN-D baselines are read from the legacy dirs '
                        'in LEGACY_BASELINE_DIRS at the top of this module.')
    p.add_argument('-save_dir', default=SAVE_DIR,
                   help='Output root (default: %(default)s).')
    p.add_argument('-models', nargs='+', default=None, choices=KNOWN_MODELS,
                   help='Restrict analysis to a subset of models. Default '
                        'auto-detects every KNOWN_MODELS prefix present '
                        'under -dirname.')
    p.add_argument('-seeds', nargs='+', type=int, default=list(range(50)),
                   help='Seeds to include in the aggregate analyses.')
    p.add_argument('-focus_seed', type=int, default=43,
                   help='Seed to run the heavy single-neuron / RDM analyses on '
                        '(default 43).')
    p.add_argument('-extra_rdm_seeds', nargs='*', type=int, default=[],
                   help='Additional seeds to include in the RDM sweep.')
    p.add_argument('-skip', nargs='*', default=[],
                   choices=['acc', 'periodicity', 'bars', 'agg', 'mat', 'rdm', 'xcorr', 'pca'],
                   help='Sections to skip.')
    p.add_argument('-targets', nargs='+', default=list(DEFAULT_TARGETS),
                   choices=['hidden', 'base_hidden', 'gs', 'fc_hidden', 'imgs'],
                   help='run_info fields to analyze (per-target run_dir call). '
                        'Shard by target to parallelize.')
    p.add_argument('-force', action='store_true',
                   help='Force re-running per-seed analyses even if outputs exist.')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Rebind MODELS to what's actually present so downstream iterators
    # (per_seed_periodicity, pca sweep, etc.) don't run against absent models.
    global MODELS
    MODELS = _detect_models(args.dirname, allowed=args.models)
    if not MODELS:
        raise SystemExit(
            f'no known models found under {args.dirname}; expected one of '
            f'{KNOWN_MODELS}. Pass -models explicitly to override.')
    print(f'analyzing models: {MODELS}')

    dfs = load_all_logs(args.dirname, seeds=args.seeds)
    # `visual/all/0/accuracy` / `mental/all/0/accuracy` are computed by
    # calculate_average as 0.8 * seen + 0.2 * unseen per env.
    dfs = calculate_average(dfs)
    print(f'loaded {len(dfs)} log rows from {dfs["name"].nunique()} runs '
          f'across models {sorted(dfs["model"].unique())}')

    if 'acc' not in args.skip:
        aggregate_accuracy(dfs, args.save_dir)

    if 'periodicity' not in args.skip:
        per_seed_periodicity(args.dirname, args.save_dir,
                             seeds=args.seeds, focus_seed=args.focus_seed,
                             targets=tuple(args.targets),
                             force=args.force)

    if 'bars' not in args.skip:
        post_periodicity_bars(dfs, args.save_dir)

    if 'agg' not in args.skip:
        aggregated_periodicity(dfs, args.save_dir, focus_seed=args.focus_seed)

    mat_dir = os.path.join(args.save_dir, 'mat_exports')
    if 'mat' not in args.skip:
        export_mats(args.dirname, mat_dir)

    if 'rdm' not in args.skip:
        # Default: run RDM across every seed in the sweep so we can aggregate.
        # focus_seed is included; incomplete-distance-coverage seeds are dropped
        # inside run_rdm.
        extra = list(args.extra_rdm_seeds or args.seeds)
        run_rdm(args.focus_seed, extra, args.save_dir, mat_dir)
        # After per-seed npz + rdm_aggregate.py, emit two paper-style
        # perspective boxplots for each aggregated CSV variant on disk.
        _rdm_perspective_plots(os.path.join(args.save_dir, 'rdm'))

    if 'xcorr' not in args.skip:
        # Per-seed pairwise xcov heatmaps via the MATLAB batch
        # (scripts/matlab_xcorr_all_seeds.m) -- matches paper style
        # (parula, correct pbaspect per subplot) and disambiguates VHA-ReLU
        # from VHA-ReLU-D by exact filename (Python port shared a filename
        # prefix). One PDF per seed per model at
        # <save_dir>/xcorr/<model>/xcorr_S<seed>_cond<c>.pdf.
        xcorr_out = os.path.join(args.save_dir, 'xcorr')
        print(f'\n--- xcorr sweep (MATLAB) -> {xcorr_out} ---')
        _run_matlab_xcorr(mat_dir, xcorr_out)

    if 'pca' not in args.skip:
        # Per-seed trajectory PCA (adapted from pca_utils/new_pca.py). One PDF
        # per seed per layer per viewpoint at
        # <save_dir>/pca/<model>/{actionRNN,distanceRNN}_S<seed>_elevE_azimA.pdf.
        pca_out = os.path.join(args.save_dir, 'pca')
        for m in MODELS:
            print(f'\n--- pca sweep for {m} ---')
            plot_pca_for_sweep(mat_dir, pca_out, model=m)

    print(f'\ndone -> {args.save_dir}')


if __name__ == '__main__':
    main()
