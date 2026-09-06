"""
Helpers for aggregating per-seed RDM pairing scores across training seeds.

Used by the root-level ``rdm_aggregate.py`` driver. Pulled out here so it can
be reused (e.g. from notebooks) without re-importing the driver.
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns


# Per-seed cross-system Spearman pairings written by rdm_analysis.py.
PAIR_KEYS = ('ppc_ppc', 'ec_ec', 'ppc_ec_cross', 'ec_ppc_cross')
PAIR_LABELS = {
    'ppc_ppc':      '7a ~ hidden\n(PPC ~ PPC-like)',
    'ec_ec':        'EC ~ base_hidden\n(EC ~ EC-like)',
    'ppc_ec_cross': '7a ~ base_hidden\n(swapped)',
    'ec_ppc_cross': 'EC ~ hidden\n(swapped)',
}

# Specificity contrasts: each value is positive if the model layer prefers
# the *expected* monkey region over the swapped one.
SPECIFICITY = {
    # Δ_PPC = ρ(7a, hidden) − ρ(7a, base_hidden)
    'delta_ppc':  ('ppc_ppc',     'ppc_ec_cross'),
    # Δ_EC  = ρ(EC, base_hidden) − ρ(EC, hidden)
    'delta_ec':   ('ec_ec',       'ec_ppc_cross'),
}
SPECIFICITY_LABEL = {
    'delta_ppc': r'$\Delta_{\rm PPC}$' + '\nρ(7a,hidden) − ρ(7a,base)',
    'delta_ec':  r'$\Delta_{\rm EC}$'  + '\nρ(EC,base) − ρ(EC,hidden)',
    'diag_anti': '½ (diag − anti-diag)\n(combined specificity)',
}


def _scalar(x):
    """Unwrap 0-d numpy scalars saved by np.savez back to Python scalars."""
    a = np.asarray(x)
    if a.shape == ():
        return a.item()
    return a


def collect(pattern):
    """Walk every npz matching `pattern` and return a long-format DataFrame.

    One row per (file, mode) - i.e. each pair_scores npz holds *one* mode,
    so one row per file.
    """
    rows = []
    for path in sorted(glob.glob(pattern)):
        with np.load(path, allow_pickle=True) as d:
            row = {
                'file':      os.path.basename(path),
                'seed':      int(_scalar(d['seed'])),
                't_common':  int(_scalar(d['t_common'])),
                'mode':      str(_scalar(d['mode'])),
                'metric':    str(_scalar(d['metric'])),
                'smoothing': bool(_scalar(d['smoothing']))
                              if 'smoothing' in d.files else False,
            }
            # PCA bookkeeping (present only for runs after the PCA refactor).
            for opt_k in ('pca_use', 'pca_components',
                          'pca_k_7a', 'pca_k_ec',
                          'pca_k_hidden', 'pca_k_base_hidden'):
                if opt_k in d.files:
                    row[opt_k] = int(_scalar(d[opt_k]))
            if 'pca_evr' in d.files:
                row['pca_evr'] = float(_scalar(d['pca_evr']))
            for k in PAIR_KEYS:
                row[k] = float(_scalar(d[k]))
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no pair_scores files match {pattern!r}")
    return pd.DataFrame(rows)


def add_specificity_columns(df):
    """Augment the DataFrame with per-seed pairing-specificity contrasts.

    delta_ppc : Δ for the PPC <-> hidden mapping. Positive if 7a is more
                similar to model `hidden` than to `base_hidden`.
    delta_ec  : Δ for the EC <-> base_hidden mapping. Positive if EC is more
                similar to model `base_hidden` than to `hidden`.
    diag_anti : ½ ( (ppc_ppc + ec_ec) − (ppc_ec_cross + ec_ppc_cross) ),
                a single bundled specificity score (mean(diag) − mean(anti)).
    """
    for name, (pos, neg) in SPECIFICITY.items():
        df[name] = df[pos] - df[neg]
    diag = (df['ppc_ppc'] + df['ec_ec']) / 2.0
    anti = (df['ppc_ec_cross'] + df['ec_ppc_cross']) / 2.0
    df['diag_anti'] = diag - anti
    return df


def wilcoxon_table(df):
    """One-sided Wilcoxon signed-rank test, H0: median = 0, H1: median > 0.

    Returns a DataFrame with one row per (mode, specificity score).
    """
    rows = []
    for mode in sorted(df['mode'].unique()):
        sub = df[df['mode'] == mode]
        for name in ('delta_ppc', 'delta_ec', 'diag_anti'):
            v = sub[name].dropna().to_numpy()
            if v.size < 2 or np.allclose(v, 0):
                stat, p = (np.nan, np.nan)
            else:
                stat, p = wilcoxon(v, alternative='greater',
                                   zero_method='wilcox')
            rows.append(dict(mode=mode, score=name,
                             n=len(v),
                             mean=float(np.mean(v)),
                             median=float(np.median(v)),
                             sem=float(np.std(v, ddof=1) / np.sqrt(len(v))),
                             stat=float(stat) if np.isfinite(stat) else np.nan,
                             p_one_sided=float(p) if np.isfinite(p) else np.nan))
    return pd.DataFrame(rows)


def plot_specificity(df, save_path, title=''):
    """Box+swarm of the three specificity scores, split by mode.

    Each panel shows Δ_PPC, Δ_EC and the bundled diag−antidiag score with a
    red dashed reference line at 0. Asterisks above each box mark one-sided
    Wilcoxon significance (p < 0.05 / 0.01 / 0.001).
    """
    score_keys = ['delta_ppc', 'delta_ec', 'diag_anti']
    long = df.melt(id_vars=['seed', 'mode'], value_vars=score_keys,
                   var_name='score', value_name='delta')
    long = long.dropna(subset=['delta'])
    long['score_label'] = long['score'].map(SPECIFICITY_LABEL)
    order = [SPECIFICITY_LABEL[k] for k in score_keys]

    modes = sorted(df['mode'].unique())
    n = len(modes)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.6), squeeze=False)
    for j, m in enumerate(modes):
        ax  = axes[0, j]
        sub = long[long['mode'] == m]
        sns.boxplot(data=sub, x='score_label', y='delta', order=order,
                    ax=ax, showfliers=False, color='lightgrey')
        sns.swarmplot(data=sub, x='score_label', y='delta', order=order,
                      ax=ax, size=3, alpha=0.7, color='black')
        ax.axhline(0, color='red', lw=0.5, ls='--')
        ax.set_ylabel(r'$\Delta$ Spearman')
        ax.set_xlabel('')
        ax.set_title(f'mode = {m}', fontsize=10)
        ax.tick_params(axis='x', labelsize=8)

        for i, key in enumerate(score_keys):
            v = df[df['mode'] == m][key].dropna().to_numpy()
            if v.size < 2 or np.allclose(v, 0):
                continue
            _, p = wilcoxon(v, alternative='greater', zero_method='wilcox')
            star = ('***' if p < 1e-3 else
                    '**'  if p < 1e-2 else
                    '*'   if p < 5e-2 else 'ns')
            top = sub[sub['score'] == key]['delta'].max()
            ax.text(i, top + 0.02, f'{star}\np={p:.1e}',
                    ha='center', va='bottom', fontsize=7)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_distributions(df, save_path, title=''):
    """Box + swarm of the four pair scores, split by mode."""
    long = df.melt(id_vars=['seed', 'mode'], value_vars=list(PAIR_KEYS),
                   var_name='pair', value_name='spearman')
    long = long.dropna(subset=['spearman'])
    long['pair_label'] = long['pair'].map(PAIR_LABELS)

    modes = sorted(df['mode'].unique())
    n = len(modes)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), squeeze=False)
    for j, m in enumerate(modes):
        ax = axes[0, j]
        sub = long[long['mode'] == m]
        order = [PAIR_LABELS[k] for k in PAIR_KEYS]
        sns.boxplot(data=sub, x='pair_label', y='spearman', order=order,
                    ax=ax, showfliers=False, color='lightgrey')
        sns.swarmplot(data=sub, x='pair_label', y='spearman', order=order,
                      ax=ax, size=3, alpha=0.7, color='black')
        ax.axhline(0, color='red', lw=0.5, ls='--')
        # NB: hist of past values lived in roughly [-0.6, 1.05]; we crop tight
        # to highlight the upper region where pairings actually concentrate.
        ax.set_ylim(0.5, 1.05)
        ax.set_ylabel('RDM Spearman (vs monkey)')
        ax.set_xlabel('')
        ax.set_title(f'mode = {m}', fontsize=10)
        ax.tick_params(axis='x', labelsize=7)
        for tl in ax.get_xticklabels():
            tl.set_rotation(20); tl.set_horizontalalignment('right')
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
