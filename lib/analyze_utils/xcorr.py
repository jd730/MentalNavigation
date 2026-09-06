"""Pairwise cross-correlation / cross-covariance analysis per seed.

Python port of the ``draw_xcorr.m`` MATLAB runner. For each successful
trial (default ``cond=2`` -> 3rd trial, matching the MATLAB 1-indexed
default), computes pairwise xcov (or xcorr) with MATLAB's ``'coeff'``
normalization between every pair of units in each layer:

    * gs           -> monkey-EC-like grid-cell code
    * base_hidden  -> distance RNN                    (double-CTRNN only)
    * hidden       -> action RNN

Each row of the resulting (n_pairs, 2*maxlag+1) matrix is one cell-pair's
lag profile; rows are sorted by argmax lag and drawn as a heatmap.

For the mental-navigation task the paper convention marks landmark
boundaries as vertical grey lines at |lag| = 12 (one landmark step).
"""
import os
import re

import numpy as np
import scipy.io
import matplotlib.pyplot as plt


DEFAULT_MAXLAG = 20
DEFAULT_COND = 2
# MATLAB draw_xcorr.m uses gs_units_mnav(30:50, :) -- 1-indexed rows 30..50
# inclusive = 21 rows. Python 0-indexed equivalent: [29, 50) (also 21 rows).
DEFAULT_GS_WINDOW = (29, 50)


# ---------------------------------------------------------------------------
# Core: MATLAB-equivalent xcorr/xcov with 'coeff' normalization
# ---------------------------------------------------------------------------

def _pair_xcov(x, y, maxlag, use_xcov):
    """MATLAB-equivalent xcov / xcorr with 'coeff' normalization.

    Args:
        x, y: 1D arrays, same length.
        maxlag: keep lags in ``[-maxlag, maxlag]`` (output length 2*maxlag+1).
        use_xcov: if True, mean-center x and y first (xcov); else xcorr.

    Returns:
        (2*maxlag+1,) array of normalized cross-{covariance | correlation}.
        When ``maxlag > T-1``, entries at |lag| > T-1 are zero (matches
        MATLAB xcov/xcorr behavior).
    """
    if use_xcov:
        x = x - x.mean()
        y = y - y.mean()
    T = x.shape[0]
    full = np.correlate(x, y, mode='full')  # length 2T-1, lag=-(T-1)..(T-1)
    out = np.zeros(2 * maxlag + 1, dtype=full.dtype)
    # Overlap between the valid xcorr window [-(T-1), T-1] and the requested
    # window [-maxlag, maxlag].
    lag_lo = max(-(T - 1), -maxlag)
    lag_hi = min(T - 1, maxlag)
    if lag_lo > lag_hi:  # no overlap; return zeros
        return out
    src_lo = (T - 1) + lag_lo         # index into `full`
    src_hi = (T - 1) + lag_hi + 1
    dst_lo = maxlag + lag_lo          # index into `out`
    dst_hi = maxlag + lag_hi + 1
    out[dst_lo:dst_hi] = full[src_lo:src_hi]
    # 'coeff' normalization: divide by sqrt(dot(x,x) * dot(y,y)) on the same
    # (centered or raw) signals used to compute `full`.
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom > 0:
        out /= denom
    return out


def compute_pair_xcov(units, maxlag=DEFAULT_MAXLAG, use_xcov=True,
                       time_window=None):
    """Pairwise xcov/xcorr across all unordered pairs of columns of ``units``.

    Args:
        units: (T, n_units) trial activity.
        maxlag: keep lags in [-maxlag, maxlag].
        use_xcov: True -> xcov (mean-centered), False -> xcorr.
        time_window: optional (start, stop) time slice applied to both signals.

    Returns:
        (n_pairs, 2*maxlag+1) array. n_pairs = n_units*(n_units-1)/2, indexed
        in the same order as ``[(i, j) for i in range(n) for j in range(i+1, n)]``.
    """
    if time_window is not None:
        units = units[time_window[0]:time_window[1]]
    T, n = units.shape
    n_pairs = n * (n - 1) // 2
    out = np.empty((n_pairs, 2 * maxlag + 1), dtype=np.float32)
    k = 0
    for i in range(n):
        xi = units[:, i]
        for j in range(i + 1, n):
            out[k] = _pair_xcov(xi, units[:, j], maxlag, use_xcov)
            k += 1
    return out


def sort_by_peak_lag(xcorr_pairs):
    """Sort rows of an (n_pairs, L) xcorr matrix by argmax lag.

    Returns:
        (sorted_matrix, sort_order).
    """
    peak = xcorr_pairs.argmax(axis=1)
    order = np.argsort(peak)
    return xcorr_pairs[order], order


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_heatmap(ax, sorted_pairs, maxlag, title, landmark_step=12,
                   show_landmarks=True):
    """One panel: heatmap of sorted pairwise xcorr with landmark markers."""
    xxi = np.arange(-maxlag, maxlag + 1)
    # 'viridis' matches MATLAB's default parula colormap most closely so
    # side-by-side comparison with draw_xcorr.m outputs is direct.
    im = ax.imshow(sorted_pairs, aspect='auto',
                   extent=(xxi[0], xxi[-1], sorted_pairs.shape[0], 1),
                   cmap='viridis', vmin=-1, vmax=1, interpolation='nearest')
    ax.axvline(0, color='red', linewidth=1)
    if show_landmarks:
        ax.axvline(landmark_step, color='k', linewidth=0.6)
        ax.axvline(-landmark_step, color='k', linewidth=0.6)
    ax.set_xlim(-maxlag, maxlag)
    ax.set_xlabel('peak phase')
    ax.set_ylabel('# cell pairs')
    ax.set_title(title)
    return im


def plot_xcorr_for_mat(mat_path, save_dir, seed=None, model=None,
                       cond=DEFAULT_COND, maxlag=DEFAULT_MAXLAG,
                       use_xcov=True, gs_window=DEFAULT_GS_WINDOW,
                       use_topk_dist=False, top_k_dist=64):
    """End-to-end pairwise xcov/xcorr plot for one .mat file (one seed).

    Produces ``xcorr_S<seed>_cond<c>.pdf`` under ``save_dir``. Each figure
    has 2 or 3 rows depending on whether the model has a base_hidden layer:
      * gs (grid cells)
      * base_hidden (dist RNN)   [only for double-CTRNN models]
      * hidden (action RNN)
    """
    if seed is None:
        m = re.search(r'S(\d+)Ep', os.path.basename(mat_path))
        seed = int(m.group(1)) if m else 0
    if model is None:
        m = re.match(r'([A-Za-z\-]+(?:_[A-Z]+)*)_N\d+L',
                     os.path.basename(mat_path))
        model = m.group(1) if m else 'unknown'

    m = scipy.io.loadmat(mat_path)
    n_trials = m['hidden'].shape[0]
    if cond >= n_trials:
        print(f'  {mat_path}: cond={cond} out of range (n_trials={n_trials}); skipped')
        return

    def _grab(name):
        # MATLAB reference (draw_xcorr.m lines 203-205) sets any NaN in the
        # trial to 0 then uses the full 100-timestep vector -- no valid-window
        # slicing. Match that verbatim so paired monkey/model xcov comparisons
        # use the same effective signal length + padding contribution.
        arr = m[name]
        if arr.ndim == 3:
            return np.nan_to_num(arr[cond].astype(np.float64), nan=0.0)
        return None

    action_units = _grab('hidden')
    dist_units   = _grab('base_hidden') if 'base_hidden' in m else None
    gs_units     = _grab('gs')          if 'gs'         in m else None

    have_dist = dist_units is not None and dist_units.shape[-1] > 0 \
                and not np.allclose(dist_units.std(0), 0)

    if use_topk_dist and have_dist and 'dist_classifier.weight' in m:
        w = np.ravel(m['dist_classifier.weight'])
        k = min(top_k_dist, w.size)
        top = np.argsort(-np.abs(w))[:k]
        dist_units = dist_units[:, top]

    layers = []
    if gs_units is not None and gs_units.shape[-1] > 0:
        gs_pairs = compute_pair_xcov(gs_units, maxlag=maxlag, use_xcov=use_xcov,
                                     time_window=gs_window)
        gs_sorted, _ = sort_by_peak_lag(gs_pairs)
        layers.append((gs_sorted, f'grid cells (EC) — win {gs_window}'))
    if have_dist:
        dist_pairs = compute_pair_xcov(dist_units, maxlag=maxlag, use_xcov=use_xcov)
        dist_sorted, _ = sort_by_peak_lag(dist_pairs)
        layers.append((dist_sorted, 'distance RNN'))
    action_pairs = compute_pair_xcov(action_units, maxlag=maxlag, use_xcov=use_xcov)
    action_sorted, _ = sort_by_peak_lag(action_pairs)
    layers.append((action_sorted, 'action RNN'))

    n_rows = len(layers)
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(n_rows, 1, figsize=(4, 2.6 * n_rows), dpi=200)
    if n_rows == 1:
        axes = [axes]
    im = None
    for ax, (sorted_pairs, title) in zip(axes, layers):
        im = _draw_heatmap(ax, sorted_pairs, maxlag, title)
    corr_mode = 'xcov coeff' if use_xcov else 'xcorr coeff'
    dist_mode = (f'dist topK={top_k_dist}' if use_topk_dist
                 else 'dist all neurons')
    fig.suptitle(f'{model}  seed={seed}  cond={cond}\n{corr_mode} | {dist_mode}',
                 fontsize=9)
    if im is not None:
        fig.subplots_adjust(right=0.82, hspace=0.5, top=0.90)
        cax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cax)
    save_path = os.path.join(save_dir, f'xcorr_S{seed}_cond{cond}.pdf')
    fig.savefig(save_path)
    plt.close(fig)


def plot_xcorr_for_sweep(mat_dir, save_root, model=None, seeds=None,
                         cond=DEFAULT_COND, maxlag=DEFAULT_MAXLAG,
                         use_xcov=True, gs_window=DEFAULT_GS_WINDOW,
                         use_topk_dist=False, top_k_dist=64):
    """Iterate .mat exports and emit per-seed xcorr PDFs at
    ``<save_root>/<model>/xcorr_S<seed>_cond<c>.pdf``.

    Args:
        mat_dir: directory of .mat files.
        save_root: output root; per-model subdirs are created.
        model: exact-prefix filter (``'VHA-ReLU-D'`` / ``'VHA-ReLU'``); if
            None, processes every .mat file.
        seeds: optional int allow-list.
    """
    import glob
    pattern = os.path.join(mat_dir, f'{model}_*.mat' if model else '*.mat')
    paths = sorted(glob.glob(pattern))
    if model == 'VHA-ReLU':
        paths = [p for p in paths if 'VHA-ReLU-D_' not in os.path.basename(p)]
    seeds_set = set(seeds) if seeds is not None else None

    for p in paths:
        m_seed = re.search(r'S(\d+)Ep', os.path.basename(p))
        seed = int(m_seed.group(1)) if m_seed else 0
        if seeds_set is not None and seed not in seeds_set:
            continue
        m_model = re.match(r'([A-Za-z\-]+(?:_[A-Z]+)*)_N\d+L',
                           os.path.basename(p))
        _model = m_model.group(1) if m_model else 'unknown'
        save_dir = os.path.join(save_root, _model)
        plot_xcorr_for_mat(p, save_dir, seed=seed, model=_model,
                            cond=cond, maxlag=maxlag, use_xcov=use_xcov,
                            gs_window=gs_window,
                            use_topk_dist=use_topk_dist, top_k_dist=top_k_dist)
        print(f'  {_model} seed {seed:2d} -> {save_dir}')


__all__ = [
    'DEFAULT_MAXLAG',
    'DEFAULT_COND',
    'DEFAULT_GS_WINDOW',
    'compute_pair_xcov',
    'sort_by_peak_lag',
    'plot_xcorr_for_mat',
    'plot_xcorr_for_sweep',
]
