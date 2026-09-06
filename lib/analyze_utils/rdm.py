"""Representational Dissimilarity Matrix (RDM) library.

Functions for comparing monkey neural recordings to a trained RNN's hidden
activations via second-order RSA (Kriegeskorte 2008). Used as a CLI from
rdm_analysis.py and as an imported library by rdm_aggregate.py.
"""
import os

import numpy as np
import scipy.io
import mat73
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Conditions: signed distances, with 0 omitted (the monkey task never asks
# the animal to navigate distance 0). Kept under the original name for
# backward-compat callers (notebooks, rdm_noise_ceiling.py).
DISTANCE_CONDITIONS = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
# All 30 (curr, target) pairs in {1..6}^2 with curr != target. This is the
# "rich" 30-condition variant used by --conditions pair.
PAIR_CONDITIONS = [(c, t) for c in range(1, 7) for t in range(1, 7) if c != t]
# Default exposed name. Existing call sites that import CONDITIONS keep their
# current behaviour (signed distance).
CONDITIONS = DISTANCE_CONDITIONS


def compute_cond_idx(curr, target, conditions):
    """Per-trial condition index into `conditions` (-1 if no match).

    conditions : list of either ints (signed distance) or (int, int) tuples
                 (curr, target). Trial labels are computed accordingly.
    """
    curr = np.asarray(curr, dtype=int)
    target = np.asarray(target, dtype=int)
    if conditions and isinstance(conditions[0], tuple):
        lookup = {c: i for i, c in enumerate(conditions)}
        return np.array([lookup.get((int(c), int(t)), -1)
                         for c, t in zip(curr, target)], dtype=int)
    lookup = {c: i for i, c in enumerate(conditions)}
    dist = target - curr
    return np.array([lookup.get(int(d), -1) for d in dist], dtype=int)


def label_distance(label):
    """Absolute distance |target - curr| for either label kind."""
    if isinstance(label, tuple):
        return abs(int(label[1]) - int(label[0]))
    return abs(int(label))


def label_string(label):
    """Compact string form for plot tick labels."""
    if isinstance(label, tuple):
        return f'{label[0]}\u2192{label[1]}'
    return f'{label:+d}' if label != 0 else '0'


def variant_subdir(cond_mode: str, pca_use: int = 0, pca_evr=None) -> str:
    """One canonical subdirectory per analysis variant.

    Outputs live under ``<save_dir>/<cond_mode>/<pca>/`` so the same base dir
    can hold many variants without their files colliding. The shared
    ``_monkey_cache/`` is always kept at the base level (it doesn't change
    across PCA variants since the PCA basis is itself variant-keyed).

    Examples
    --------
    variant_subdir('distance')                 -> 'distance/raw'
    variant_subdir('distance', pca_use=3)      -> 'distance/pca3'
    variant_subdir('pair', pca_evr=0.8)        -> 'pair/pcaEVR80'
    """
    if pca_evr is not None:
        pca = f'pcaEVR{int(round(pca_evr * 100))}'
    elif pca_use > 0:
        pca = f'pca{pca_use}'
    else:
        pca = 'raw'
    return os.path.join(cond_mode, pca)

# Default length to which all per-condition trajectories are time-stretched
# in the 'flat' RDM mode. Trials of large |d| have many more original time
# bins than trials of small |d| - we resample to a common length so the
# "flat" pattern vectors all live in the same space. Override via --t_common.
T_COMMON = 12


# ===========================================================================
#  Loaders
# ===========================================================================

def load_monkey(path: str, smooth_sigma: int = 400, smooth_radius: int = 800):
    """Load one monkey's neural tensor and metadata for RDM analysis.

    Parameters
    ----------
    path : str
        Path to the *_a_neur_tensor_joyon.mat file.
    smooth_sigma : int
        Gaussian smoothing sigma along the time axis, in ms (binwidth=1ms).
        Matches the value used in pca.py.
    smooth_radius : int
        Truncation radius for the gaussian kernel, also in ms.

    Returns
    -------
    dict with
        activity      : (Trials, Time, Neurons) smoothed firing rate
        dist          : (Trials,) signed integer distance per trial
        keep          : (Trials,) bool, successful AND validtrials_mm
        edges         : (Time,) bin edges, in seconds (t=0 is cue onset)
        binwidth      : float, seconds per bin (1e-3)
        zero_bin      : int, bin index nearest t=0
        bins_per_unit : int, # bins corresponding to one signed-distance unit
                        (~0.65 s -> 650 bins). Used to build the navigation
                        window per condition.
    """
    d = mat73.loadmat(path)

    # Raw spike tensor: (Neuron, Time, Trial). 1 ms bins.
    neur = d['neur_tensor_joyon']

    # Trial metadata. cond_label is the column-name vector for cond_matrix.
    cond_label = list(d['cond_label'])
    cond_matrix = d['cond_matrix']
    succ  = cond_matrix[:, cond_label.index('succ')]           == 1
    valid = cond_matrix[:, cond_label.index('validtrials_mm')] == 1
    keep  = succ & valid                                # both must be true

    # Per-trial signed integer distance. `pm.dist_conditions` stores it
    # directly (already integer, in {-5,...,-1,1,...,5}).
    dist = d['pm']['dist_conditions'].astype(int)

    # Per-trial start and target landmark positions (integers in {1..6}).
    # Used by the 'pair' condition mode where each (curr, target) tuple is a
    # separate condition (30 total).
    curr   = cond_matrix[:, cond_label.index('curr')].astype(int)
    target = cond_matrix[:, cond_label.index('target')].astype(int)

    # 1) spike-count -> rate-ish (the /1000 is from pca.py; matches Hz when
    #    the bins are 1 ms).
    # 2) gaussian smooth along time with sigma=400 ms - heavy smoothing makes
    #    spikes into a continuous firing-rate trace.
    neur = neur / 1000.0
    neur = gaussian_filter1d(neur, sigma=smooth_sigma, axis=1,
                             radius=smooth_radius)

    # Reorder to (Trials, Time, Neurons) - the convention the rest of the
    # script uses.
    neur = neur.transpose(2, 1, 0)

    # Time edges (seconds). For these files: edges[0]=-1.999, edges[-1]=4.498.
    # t=0 marks the alignment event (cue onset for joyon-aligned tensors).
    edges = np.asarray(d['joyon']['edges']).squeeze()
    binwidth = float(d['joyon']['binwidth'])
    zero_bin = int(np.argmin(np.abs(edges)))

    # `ta` (column 0 of cond_matrix) is the *signed time* the animal is asked
    # to navigate, in seconds. Dividing by the integer distance gives the
    # per-unit time scale (~0.65 s/unit on these recordings). Converting to
    # bins via /binwidth gives the navigation-window length per |distance|.
    ta = cond_matrix[:, cond_label.index('ta')]
    nz = (dist != 0)
    sec_per_unit  = float(np.nanmean(ta[nz] / dist[nz]))
    bins_per_unit = int(round(sec_per_unit / binwidth))

    return {
        'activity': neur,
        'dist': dist,
        'curr':   curr,
        'target': target,
        'keep': keep,
        'edges': edges,
        'binwidth': binwidth,
        'zero_bin': zero_bin,
        'bins_per_unit': bins_per_unit,
    }


def load_model(path: str, smooth_sigma: int = 4, smooth_truncate: int = 2, dist_unit: int = 12):
    """Load one model run (one seed) of the vHMN2 mental-navigation RNN.

    Parameters
    ----------
    path : str
        Path to the seed's mat file (e.g. norm5_vHMN2_rev_abs/random_*S43*.mat).
    smooth_sigma, smooth_truncate
        Gaussian smoothing parameters along the time axis. Same as pca.py.
    dist_unit : int
        The model represents one monkey-distance-unit as `dist_unit` raw
        steps (12 by default). Dividing the raw distance by `dist_unit`
        gives the integer condition label that matches the monkey.

    Returns
    -------
    dict with
        hidden            : (Trials, T, N) "action RNN" activity (PPC-like)
        base_hidden       : (Trials, T, N) "distance RNN" activity (EC-like)
        dist              : (Trials,) signed integer condition label
        raw_dist          : (Trials,) raw model-units distance (multiple of 12)
        keep              : (Trials,) bool, model succeeded on this trial
        dist_unit         : int, raw_dist / dist_unit -> integer condition
        smooth_sigma      : passed through for downstream smoothing
        smooth_truncate   : passed through for downstream smoothing

    Why we do *not* smooth here
    ---------------------------
    Each trial's `hidden` is shape (T, N) where the *first* (T - (|raw_d|+1))
    timesteps are NaN-padding (the model only ran for |raw_d|+1 useful
    steps). If we gaussian-smooth here, NaN spreads across the kernel
    radius. Instead we slice each trial to its valid window in
    `condition_patterns` and smooth there.
    """
    m = scipy.io.loadmat(path)

    trajs    = m['trajs']                                          # (Trials, 2)
    raw_dist = (trajs[:, 1] - trajs[:, 0]).astype(int)             # in {-60..60}
    dist     = (raw_dist / dist_unit).astype(int)                  # in {-5..5}\{0}
    # Map raw positions {0,12,...,60} -> integer landmark positions {1..6}
    # so the model labels align with the monkey's `curr`, `target` columns.
    curr   = (trajs[:, 0] / dist_unit).astype(int) + 1
    target = (trajs[:, 1] / dist_unit).astype(int) + 1
    keep   = m['suc'].squeeze().astype(bool)

    return {
        'hidden':       m['hidden'],
        'base_hidden':  m['base_hidden'],
        'dist':         dist,
        'curr':         curr,
        'target':       target,
        'raw_dist':     raw_dist,
        'keep':         keep,
        'dist_unit':    dist_unit,
        'smooth_sigma': smooth_sigma,
        'smooth_truncate': smooth_truncate,
    }


# ===========================================================================
#  Pattern construction
# ===========================================================================

def _stretch_time(x, T_target):
    """Linearly resample axis=1 of x from T to T_target bins.

    x shape: (Trials, T, Neurons). Returns (Trials, T_target, Neurons).

    Used so that condition patterns from trials of different |distance|
    (and therefore different lengths) can be flattened into vectors of the
    same length for the 'flat' RDM mode.
    """
    n_trials, T, N = x.shape
    if T == T_target:
        return x

    src = np.linspace(0, 1, T)            # original normalized time
    dst = np.linspace(0, 1, T_target)     # new normalized time

    out = np.empty((n_trials, T_target, N), dtype=x.dtype)
    for t_idx, t_val in enumerate(dst):
        # find the bracketing pair (i-1, i) in src for t_val and lerp
        i = np.searchsorted(src, t_val).clip(1, T - 1)
        a = (t_val - src[i - 1]) / (src[i] - src[i - 1])
        out[:, t_idx, :] = (1 - a) * x[:, i - 1, :] + a * x[:, i, :]
    return out


def condition_patterns(activity, cond_idx, keep, conditions=CONDITIONS,
                       mode='time_mean', T_common=T_COMMON,
                       valid_window=None, smooth=None):
    """Build the per-condition response-pattern matrix for one system.

    Parameters
    ----------
    activity : (Trials, T, Neurons)
        Smoothed (or to-be-smoothed) activity.
    cond_idx : (Trials,) int
        Per-trial condition index into ``conditions`` (-1 = no match).
        Build via ``compute_cond_idx(curr, target, conditions)``.
    keep : (Trials,) bool
        Trials to include (e.g. successful, valid).
    conditions : sequence
        Condition labels in row order. Each label is either a signed
        int (distance) or a (curr, target) tuple.
    mode : 'time_mean' or 'flat'
        How to collapse the (n_c_trials, T_window, Neurons) tensor for one
        condition into a row vector. See module docstring.
    T_common : int
        Time axis is stretched to this length before flattening (mode='flat').
    valid_window : callable label -> (start_bin, end_bin)  or None
        Inclusive bin range that defines the navigation window for one
        condition label. If None, the entire time axis is used.
    smooth : dict or None
        kwargs passed to gaussian_filter1d (axis=1) after slicing. Use this
        for the model where we held off smoothing in load_model() to avoid
        NaN propagation.

    Returns
    -------
    (n_conditions, n_features) ndarray.
    """
    rows = []
    for ci, c in enumerate(conditions):
        # Trials belonging to this condition that we want to keep.
        idx = (cond_idx == ci) & keep

        # Restrict to the trial's "navigation" window first, *then* fancy-
        # index trials. Time-slice-first keeps the materialised copy small
        # when `activity` is a multi-GB mmap view.
        if valid_window is not None:
            s, e = valid_window(c)
            x = np.asarray(activity[:, s:e + 1, :][idx])
        else:
            x = np.asarray(activity[idx])
        if x.size == 0:
            raise ValueError(f"no trials for condition {c}")

        # Smooth in time (only if not already smoothed upstream).
        if smooth is not None:
            x = gaussian_filter1d(x, axis=1, **smooth)

        if mode == 'time_mean':
            # Single (Neurons,) vector: mean firing rate across trials and
            # within-trial timepoints. Most robust, ignores temporal shape.
            row = np.nanmean(x, axis=(0, 1))                    # (N,)
        elif mode == 'flat':
            # Trial-average, then resample time to T_common, then flatten.
            # Vector length = T_common * Neurons.
            xs  = _stretch_time(x, T_common)                    # (n_c, T_c, N)
            row = np.nanmean(xs, axis=0).reshape(-1)            # (T_c * N,)
        else:
            raise ValueError(mode)
        rows.append(row)

    return np.stack(rows, axis=0)


# ===========================================================================
#  PCA-space helpers
# ===========================================================================
#
# Following pca.py: for each system independently we (a) trial-average each
# condition over its valid window, (b) concatenate the per-condition
# time-resolved averages along time -> (sum_c T_c, D_neurons), (c) fit PCA on
# that matrix. The fitted basis is then used to project the *full* (Trials, T,
# D_neurons) activity tensor to a (Trials, T, K) tensor with K <<= D before
# building RDM patterns. The two knobs are:
#   pca_components : how many components to fit (e.g. 256 for a full-ish basis
#                    we can later inspect variance explained on).
#   pca_use        : how many of those components to actually keep when
#                    projecting (e.g. 3 -> RDM lives in a 3-D subspace).

def fit_pca_basis(activity, cond_idx, keep, conditions=CONDITIONS,
                  valid_window=None, smooth=None, n_components=256):
    """Fit one PCA basis on the trial-averaged, time-resolved activity.

    Mirrors what pca.py does: stack each condition's trial-mean trajectory
    (window-sliced and optionally smoothed) along time and fit PCA on the
    resulting (sum_c T_c, D) matrix.

    Returns
    -------
    sklearn.decomposition.PCA, fitted. Truncate to top-k after the fact via
    ``components_[:k]``.
    """
    chunks = []
    for ci, c in enumerate(conditions):
        idx = (cond_idx == ci) & keep
        # Slice the time window first (cheap view on the mmap), then fancy-
        # index trials. This materialises only (n_trials_c, T_c, D) bytes
        # instead of (n_trials_c, T_full, D), critical when `activity` is a
        # multi-GB on-disk array.
        if valid_window is not None:
            s, e = valid_window(c)
            x = np.asarray(activity[:, s:e + 1, :][idx])  # force into RAM
        else:
            x = np.asarray(activity[idx])
        if x.size == 0:
            raise ValueError(f"no trials for condition {c}")
        if smooth is not None:
            x = gaussian_filter1d(x, axis=1, **smooth)
        chunks.append(np.nanmean(x, axis=0))             # (T_c, D)
    X = np.concatenate(chunks, axis=0)                   # (sum_c T_c, D)
    X = X[~np.isnan(X).any(axis=1)]                      # drop any nan rows
    n = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n)
    pca.fit(X)
    return pca


def pick_k_evr(explained_variance_ratio, threshold: float, k_min: int = 2) -> int:
    """Smallest K such that cumulative explained variance >= threshold.

    Parameters
    ----------
    explained_variance_ratio : (n_components,) array from a fitted PCA.
    threshold : float in (0, 1]. e.g. 0.8 -> keep at least 80% of variance.
    k_min : int. K is clamped to be >= k_min. Default 2 because the
            'correlation' pdist metric is mathematically undefined on
            K=1 features (denominator of Pearson r is zero between
            scalars -> all-NaN RDM rows). With k_min=2 every RDM is well
            defined regardless of how concentrated the variance is.

    Returns
    -------
    int : k_min <= K <= n_components.
    """
    cum = np.cumsum(explained_variance_ratio)
    K = int(np.searchsorted(cum, threshold) + 1)
    K = max(K, k_min)
    return min(K, len(explained_variance_ratio))


def project_activity(activity, pca, k: int, chunk_rows: int = 8192):
    """Project (..., D) along the last axis to (..., k) using the top-k PCs.

    Done in row chunks so a multi-GB mmap'd `activity` never has to be fully
    materialised in RAM. We also rewrite (x - mean) @ comp as
    x @ comp - (mean @ comp) to skip the giant centred-copy intermediate.

    NaN-padded entries in `activity` propagate as NaN, which is fine because
    they get sliced out (and nan-meaned over) by condition_patterns.
    """
    D = activity.shape[-1]
    leading = activity.shape[:-1]
    flat = activity.reshape(-1, D)                       # view if possible
    comp = pca.components_[:k].T                         # (D, k)
    mean_proj = pca.mean_ @ comp                         # (k,)
    N = flat.shape[0]
    out = np.empty((N, k), dtype=np.float64)
    for s in range(0, N, chunk_rows):
        # np.asarray forces just this slice's pages from mmap into RAM.
        out[s:s + chunk_rows] = (
            np.asarray(flat[s:s + chunk_rows]) @ comp - mean_proj)
    return out.reshape(*leading, k)


# ===========================================================================
#  RDM computation and comparison
# ===========================================================================

# Pattern dissimilarity metrics that pdist understands. The driver picks
# exactly one of these per run via --metric (default 'correlation', which is
# 1 - Pearson r and is the standard Yamins/Kriegeskorte choice).
METRIC_CHOICES = ('correlation', 'euclidean', 'cosine')


def compute_rdm(patterns, metric: str = 'correlation'):
    """Pairwise dissimilarity between rows of `patterns`.

    Parameters
    ----------
    patterns : (n_conditions, n_features)
    metric   : str, any metric understood by scipy.spatial.distance.pdist

    Returns
    -------
    (n_conditions, n_conditions) symmetric RDM with zero diagonal.
    """
    return squareform(pdist(patterns, metric=metric))


def upper_tri(rdm):
    """Off-diagonal entries of an RDM, flattened (k=1 excludes the diagonal)."""
    iu = np.triu_indices_from(rdm, k=1)
    return rdm[iu]


def rdm_spearman(rdm_a, rdm_b):
    """Standard 2nd-order RSA: Spearman rho between the two RDMs' upper
    triangles. This is the rank-correlation of the dissimilarities, so it is
    insensitive to the absolute dissimilarity scale (which differs across
    metrics and across systems with different numbers of neurons).
    """
    return spearmanr(upper_tri(rdm_a), upper_tri(rdm_b)).correlation


# ===========================================================================
#  Plotting helpers
# ===========================================================================

def plot_rdm_grid(rdms, conditions, metric: str, save_path: str, title: str = '') -> None:
    """Plot one RDM heatmap per system (single column, single metric).

    rdms : dict[system_name] = (n, n) RDM
    metric : str, used only in the per-panel title.
    """
    sys_names = list(rdms.keys())
    n_rows    = len(sys_names)
    n_cond    = len(conditions)
    tick_lbl  = [label_string(c) for c in conditions]
    # 30 (curr,target) labels is too dense for default font; shrink.
    tick_fs   = 7 if n_cond <= 12 else 5

    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(4.0 + 0.05 * max(0, n_cond - 10),
                                      3.4 * n_rows),
                             squeeze=False)
    for i, sname in enumerate(sys_names):
        ax = axes[i, 0]
        R  = rdms[sname]
        im = ax.imshow(R, cmap='viridis', aspect='equal')
        ax.set_title(f"{sname}\n{metric}", fontsize=9)
        ax.set_xticks(range(n_cond)); ax.set_yticks(range(n_cond))
        ax.set_xticklabels(tick_lbl, fontsize=tick_fs, rotation=90)
        ax.set_yticklabels(tick_lbl, fontsize=tick_fs)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_cross_similarity(rdms, metric: str, save_path: str, title: str = '') -> None:
    """Plot the system x system Spearman-correlation matrix for one metric.

    rdms : dict[system_name] = (n, n) RDM
    """
    sys_names = list(rdms.keys())
    n         = len(sys_names)

    cross = np.zeros((n, n))
    for i_a, a in enumerate(sys_names):
        for i_b, b in enumerate(sys_names):
            cross[i_a, i_b] = rdm_spearman(rdms[a], rdms[b])

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.5))
    im = ax.imshow(cross, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(f"{metric}\n(Spearman of upper tri)", fontsize=10)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(sys_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(sys_names, fontsize=8)
    for i_a in range(n):
        for i_b in range(n):
            ax.text(i_b, i_a, f"{cross[i_a,i_b]:.2f}",
                    ha='center', va='center', fontsize=7,
                    color='black' if abs(cross[i_a, i_b]) < 0.6 else 'white')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
#  Monkey RDM cache
# ===========================================================================
#
# The monkey RDMs do not depend on the model seed, so over a batch run across
# many model seeds we only need to compute them once per (region, mode,
# T_common). MonkeyRDMCache lazily loads the monkey .mat file only on a cache
# miss, so seeds 1..49 of a 50-seed sweep can reuse the seed-0 cache without
# re-running the 6498-bin gaussian smoothing.

MONKEY_FILES = {
    '7a': '7a_amadeus06242019_a_neur_tensor_joyon.mat',
    'EC': 'ec_amadeus08292019_a_neur_tensor_joyon.mat',
}


class MonkeyRDMCache:
    """Cached per-(region, mode, T_common, pca_k, cond_mode) monkey RDMs.

    Files written under `cache_dir`:
        monkey_<region>_smoothed.npy / _meta.npz        -- smoothed activity
        monkey_<region>_pca<n_components>[_<cond>].npz  -- fitted PCA basis
        monkey_<region>_<mode>_T<t_common>[_pca<k>][_<cond>].npz -- patterns

    `<cond>` is empty for the default signed-distance conditions
    (back-compat) and `pair` for the 30 (curr, target) variant; the PCA
    basis is fit on the same conditions used for patterns, so it depends on
    cond_mode too.

    The monkey .mat file is only loaded on a cache miss; smoothed activity
    + PCA fit are both cached so the 50-seed batch shares them.
    """

    def __init__(self, region, mat_path, cache_dir, pca_components=256,
                 cond_mode='distance'):
        self.region   = region
        self.mat_path = mat_path
        self.cache_dir = cache_dir
        self.pca_components = pca_components
        self.cond_mode = cond_mode
        self.conditions = (PAIR_CONDITIONS if cond_mode == 'pair'
                           else DISTANCE_CONDITIONS)
        os.makedirs(cache_dir, exist_ok=True)
        # Loaded monkey dict; filled lazily on first cache miss.
        self._monk = None
        # Per-trial condition index (matches self.conditions).
        self._cond_idx = None
        # Fitted PCA, filled lazily on first PCA need.
        self._pca = None

    def _smoothed_paths(self):
        # Heavy activity tensor goes to a .npy so we can mmap it on subsequent
        # runs (50-seed SLURM jobs all share one file). The small metadata
        # lives in a sibling .npz for human-readable inspection.
        base = os.path.join(self.cache_dir, f'monkey_{self.region}_smoothed')
        return base + '.npy', base + '_meta.npz'

    def _load(self):
        """Return the smoothed monkey dict. The expensive sigma=400 gaussian
        smoothing inside load_monkey is the dominant cost; we run it once,
        save the (Trials, T, Neurons) smoothed activity to disk, and mmap it
        on subsequent calls so concurrent SLURM tasks never re-smooth.
        """
        if self._monk is not None:
            return self._monk
        act_path, meta_path = self._smoothed_paths()
        if os.path.exists(act_path) and os.path.exists(meta_path):
            print(f'Loading cached smoothed activity for monkey '
                  f'{self.region} (mmap) ...')
            activity = np.load(act_path, mmap_mode='r')
            with np.load(meta_path) as d:
                files = set(d.files)
                self._monk = {
                    'activity':      activity,
                    'dist':          d['dist'].copy(),
                    'keep':          d['keep'].copy(),
                    'edges':         d['edges'].copy(),
                    'binwidth':      float(d['binwidth']),
                    'zero_bin':      int(d['zero_bin']),
                    'bins_per_unit': int(d['bins_per_unit']),
                    'curr':   d['curr'].copy()   if 'curr'   in files else None,
                    'target': d['target'].copy() if 'target' in files else None,
                }
            # Backfill curr/target if this meta was written by an older
            # version that didn't include them. We only need cond_matrix
            # from the .mat (small), not the full neural tensor; but mat73
            # loads the whole file, so this is a one-off ~30s hit per region.
            if self._monk['curr'] is None or self._monk['target'] is None:
                print(f'  meta lacks curr/target; reloading from .mat to '
                      f'backfill (one-off)')
                d = mat73.loadmat(self.mat_path)
                cl = list(d['cond_label'])
                cm = d['cond_matrix']
                self._monk['curr']   = cm[:, cl.index('curr')].astype(int)
                self._monk['target'] = cm[:, cl.index('target')].astype(int)
                np.savez(meta_path,
                         dist=self._monk['dist'], keep=self._monk['keep'],
                         curr=self._monk['curr'], target=self._monk['target'],
                         edges=self._monk['edges'],
                         binwidth=self._monk['binwidth'],
                         zero_bin=self._monk['zero_bin'],
                         bins_per_unit=self._monk['bins_per_unit'])
                print(f'  updated meta {meta_path}')
            print(f'  shape: {activity.shape}  '
                  f'valid trials: {int(self._monk["keep"].sum())}')
            return self._monk
        # Cache miss: run the expensive load+smooth and persist results.
        print(f'Loading monkey {self.region} (cache miss, smoothing then '
              f'caching) ...')
        self._monk = load_monkey(self.mat_path)
        print(f'  shape: {self._monk["activity"].shape}  '
              f'valid trials: {int(self._monk["keep"].sum())}')
        # Save activity as a plain .npy (mmap-compatible). Metadata side-car.
        np.save(act_path, self._monk['activity'])
        np.savez(meta_path,
                 dist=self._monk['dist'], keep=self._monk['keep'],
                 curr=self._monk['curr'], target=self._monk['target'],
                 edges=self._monk['edges'],
                 binwidth=self._monk['binwidth'],
                 zero_bin=self._monk['zero_bin'],
                 bins_per_unit=self._monk['bins_per_unit'])
        print(f'  cached smoothed activity -> {act_path}')
        return self._monk

    def _window(self):
        monk = self._monk if self._monk is not None else self._load()
        # Window length depends on |target - curr| for both label kinds.
        return (lambda c: (monk['zero_bin'],
                           monk['zero_bin']
                           + label_distance(c) * monk['bins_per_unit'] - 1))

    def _get_cond_idx(self):
        if self._cond_idx is None:
            monk = self._load()
            self._cond_idx = compute_cond_idx(monk['curr'], monk['target'],
                                              self.conditions)
        return self._cond_idx

    def _cond_tag(self):
        return '' if self.cond_mode == 'distance' else f'_{self.cond_mode}'

    def _pca_cache_path(self):
        return os.path.join(
            self.cache_dir,
            f'monkey_{self.region}_pca{self.pca_components}'
            f'{self._cond_tag()}.npz')

    def get_pca(self):
        """Return a PCA-like object exposing components_, mean_,
        explained_variance_ratio_. Cached on disk for cross-seed reuse."""
        if self._pca is not None:
            return self._pca
        p = self._pca_cache_path()
        if os.path.exists(p):
            with np.load(p) as d:
                pca = PCA(n_components=int(d['components_'].shape[0]))
                pca.components_ = d['components_'].copy()
                pca.mean_       = d['mean_'].copy()
                pca.explained_variance_ratio_ = d['evr'].copy()
                pca.n_components_ = pca.components_.shape[0]
            self._pca = pca
            print(f'  loaded cached PCA basis for monkey {self.region} '
                  f'({pca.components_.shape[0]} comps)')
            return pca
        monk = self._load()
        pca = fit_pca_basis(
            monk['activity'], self._get_cond_idx(), monk['keep'],
            conditions=self.conditions,
            valid_window=self._window(), smooth=None,
            n_components=self.pca_components)
        np.savez(p, components_=pca.components_, mean_=pca.mean_,
                 evr=pca.explained_variance_ratio_)
        print(f'  cached monkey {self.region} PCA basis '
              f'({pca.components_.shape[0]} comps) -> {p}')
        self._pca = pca
        return pca

    def _cache_path(self, mode, t_common, pca_k=0):
        # We cache the *pattern matrix* (not the RDM): patterns are the
        # expensive thing (loading + smoothing 6498-bin monkey data); RDMs
        # are cheap and metric-dependent, so we recompute the RDM from the
        # cached pattern on each call. time_mean ignores t_common but we
        # still encode it for filename uniformity. pca_k=0 means raw neurons.
        suffix = f'_pca{pca_k}' if pca_k > 0 else ''
        return os.path.join(
            self.cache_dir,
            f'monkey_{self.region}_{mode}_T{t_common}{suffix}'
            f'{self._cond_tag()}.npz')

    def _get_pattern(self, mode, t_common, pca_k=0):
        path = self._cache_path(mode, t_common, pca_k)
        if os.path.exists(path):
            with np.load(path) as d:
                return d['pattern'].copy()
        # Cache miss: load monkey lazily, optionally project to PCA space,
        # then build the per-condition pattern.
        monk   = self._load()
        window = self._window()
        if pca_k > 0:
            pca = self.get_pca()
            act = project_activity(monk['activity'], pca, pca_k)
        else:
            act = monk['activity']
        P = condition_patterns(
            act, self._get_cond_idx(), monk['keep'],
            conditions=self.conditions,
            mode=mode, T_common=t_common,
            valid_window=window, smooth=None)
        np.savez(path, pattern=P,
                 conditions=np.array(self.conditions, dtype=object),
                 cond_mode=np.array(self.cond_mode),
                 t_common=np.array(t_common),
                 pca_k=np.array(pca_k))
        print(f'  cached monkey {self.region} mode={mode} T={t_common}'
              f'{" pca_k="+str(pca_k) if pca_k>0 else ""} '
              f'cond={self.cond_mode} -> {path}')
        return P

    def get_rdm(self, mode, t_common, metric, pca_k=0):
        """Return the (n_cond, n_cond) monkey RDM for one metric.
        pca_k > 0 returns the RDM in K-dim PCA-projected space.
        """
        P = self._get_pattern(mode, t_common, pca_k=pca_k)
        return compute_rdm(P, metric=metric)


# ===========================================================================
#  Driver
# ===========================================================================

