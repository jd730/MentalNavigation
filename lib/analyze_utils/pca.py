"""PCA trajectory visualization for VHA / VHA-D-family model activity.

Adapted from the prototypes at ``pca_utils/`` (``new_pca.py``,
``get_model_pca.py``, ``visualize_specific_viewpoint.py``). Convention:

    * Signed raw distance (target - start, in RAW model timesteps -- multiples
      of 12 for the standard mental-navigation task, i.e. one landmark = 12
      steps).
    * Per-trial: keep only the last ``|d|+1`` time steps (drops the pre-onset
      padding that varies per trial length).
    * Per (signed) distance: gaussian-smooth along time (sigma=4, truncate=2)
      then average across trials to get one trajectory per distance.
    * PCA(n_components=3) fit on the concatenation of all distance curves.
    * 3D visualization mirrors ``visualize_specific_view``: transparent panes,
      grid off, per-distance ``Reds`` color, start marker = circle, end
      marker = 'x'.

Public entry points:
    * :func:`plot_pca_for_mat`    -- one .mat file, one seed.
    * :func:`plot_pca_for_sweep`  -- iterate every .mat export of a sweep.

Predefined viewpoints (paper-canonical, picked from seed 19):
    * distanceRNN: (elev=330, azim=150)
    * actionRNN:   (elev=270, azim=120)
"""
import glob
import os
import re

import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


DEFAULT_SIGMA = 4
DEFAULT_TRUNCATE = 2
DEFAULT_N_COMPONENTS = 3
LANDMARK_STEP = 12  # raw model timesteps per landmark-unit distance

# Paper-canonical viewpoints picked from seed 19 rotation grids (elev, azim).
# distanceRNN (330, 150) shows the rotating structure cleanly; actionRNN
# (270, 120) shows the divergent ramps by distance sign.
DEFAULT_VIEWPOINTS = {
    'distanceRNN': [(330, 150)],
    'actionRNN':   [(270, 120)],
    'RNN':         [(270, 120)],  # single-CTRNN model -> action-like
}


# ---------------------------------------------------------------------------
# Data preparation (mirrors pca_utils/new_pca.py:split + smooth + aggregate)
# ---------------------------------------------------------------------------

def split_by_signed_distance(hidden, distance, trim_tail=0):
    """Group hidden states by signed distance, keep only last |d|+1 steps.

    Args:
        hidden: (n_trials, T, n_units).
        distance: signed integer per trial (target - start, RAW timesteps).
        trim_tail: drop this many final steps from every trajectory before
            returning, to strip target-reached / reset transients. Distances
            whose kept window would be empty are omitted.

    Returns:
        dict[int, np.ndarray]: signed distance -> (n_trials_d, |d|+1-trim_tail, n_units).
    """
    out = {}
    for d in np.unique(distance):
        d_int = int(d)
        h = hidden[distance == d]
        # Trials are right-padded with NaN; trim to the last |d|+1 valid steps.
        h = h[:, -abs(d_int) - 1:]
        if trim_tail > 0:
            if h.shape[1] <= trim_tail:
                continue
            h = h[:, :-trim_tail]
        out[d_int] = h
    return out


def smooth_and_aggregate(data_dict, sigma=DEFAULT_SIGMA, truncate=DEFAULT_TRUNCATE):
    """Smooth each distance group along time, average across trials.

    Args:
        data_dict: signed distance -> (n_trials_d, T_d, n_units).
        sigma, truncate: gaussian_filter1d params along the time axis.

    Returns:
        dict[int, np.ndarray]: signed distance -> (T_d, n_units) trial-averaged
        smoothed trajectory.
    """
    out = {}
    for d, v in data_dict.items():
        v_smooth = gaussian_filter1d(v, sigma=sigma, axis=1, truncate=truncate)
        out[d] = v_smooth.mean(axis=0)
    return out


def fit_pca(data_dict, n_components=DEFAULT_N_COMPONENTS):
    """Fit PCA on the concatenation of all per-distance trajectories.

    Args:
        data_dict: signed distance -> (T_d, n_units) trajectory.
        n_components: number of PCA components (default 3).

    Returns:
        tuple[dict[int, np.ndarray], PCA]:
            * signed distance -> (T_d, n_components) projection.
            * fitted sklearn PCA.
    """
    distances = sorted(data_dict.keys())
    stacked = np.concatenate([data_dict[d] for d in distances], axis=0)
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(stacked)
    out = {}
    idx = 0
    for d in distances:
        T = data_dict[d].shape[0]
        out[d] = projected[idx:idx + T]
        idx += T
    return out, pca


# ---------------------------------------------------------------------------
# Visualization (matches pca_utils/visualize_specific_viewpoint.py)
# ---------------------------------------------------------------------------

def visualize_specific_view(embedded_by_distance, save_path, elev=30, azim=45,
                             cmap_name='Reds'):
    """Single-panel 3D trajectory plot at a fixed viewpoint.

    Mirrors ``pca_utils/visualize_specific_viewpoint.py:visualize_specific_view``:
    transparent panes, grid off, tick labels off, one color per (sorted)
    distance from the ``Reds`` palette; start = circle, end = 'x'.

    Args:
        embedded_by_distance: dict[int, (T, 3)] as produced by :func:`fit_pca`.
        save_path: output file path (``.pdf`` recommended).
        elev, azim: camera angle in degrees.
        cmap_name: matplotlib colormap name (default ``'Reds'``).
    """
    distances = sorted(embedded_by_distance.keys())
    fig = plt.figure(figsize=(6, 4), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)

    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.4, 1, len(distances)))

    for c, d in zip(colors, distances):
        traj = embedded_by_distance[d]
        if traj.ndim < 2:
            continue
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c=c)
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c=[c])                # start = circle
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c=[c], marker='x')  # end = x

    plt.tight_layout()
    fig.savefig(save_path, dpi=1000)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------

def _load_and_prepare(mat_path):
    """Load one .mat, restrict to successful non-zero-distance trials."""
    m = scipy.io.loadmat(mat_path)
    suc = m['suc'].squeeze().astype(bool)
    if not suc.any():
        return None
    trajs = m['trajs']
    signed = (trajs[:, 1] - trajs[:, 0]).astype(int)  # RAW timesteps
    keep = suc & (signed != 0)
    if not keep.any():
        return None
    return {
        'hidden':      m['hidden'][keep],
        'base_hidden': m['base_hidden'][keep] if 'base_hidden' in m else None,
        'signed':      signed[keep],
    }


def _extract_model_seed(mat_path):
    """Parse ``<model>_...S<seed>Ep...`` out of the .mat filename."""
    base = os.path.basename(mat_path)
    m_model = re.match(r'([A-Za-z\-]+(?:_[A-Z]+)*)_N\d+L', base)
    model = m_model.group(1) if m_model else 'unknown'
    m_seed = re.search(r'S(\d+)Ep', base)
    seed = int(m_seed.group(1)) if m_seed else 0
    return model, seed


def plot_pca_for_mat(mat_path, save_dir, seed=None, model=None,
                     sigma=DEFAULT_SIGMA, truncate=DEFAULT_TRUNCATE,
                     n_components=DEFAULT_N_COMPONENTS,
                     viewpoints=None):
    """End-to-end PCA-trajectory plot for one .mat file (one seed).

    Produces up to two PDFs per predefined viewpoint:
      * ``actionRNN_S<seed>_elev{e}_azim{a}.pdf``
      * ``distanceRNN_S<seed>_elev{e}_azim{a}.pdf``  (double-CTRNN only).

    Args:
        mat_path: path to the .mat file.
        save_dir: output directory.
        seed, model: optional overrides; parsed from filename if omitted.
        sigma, truncate: gaussian_filter1d params.
        n_components: PCA dimensions (default 3).
        viewpoints: optional dict {label -> [(elev, azim), ...]}. Defaults to
            :data:`DEFAULT_VIEWPOINTS`.
    """
    if seed is None or model is None:
        m_model, m_seed = _extract_model_seed(mat_path)
        model = model or m_model
        seed = seed if seed is not None else m_seed
    if viewpoints is None:
        viewpoints = DEFAULT_VIEWPOINTS

    prepared = _load_and_prepare(mat_path)
    if prepared is None:
        print(f'  {mat_path}: no successful trials, skipped')
        return

    os.makedirs(save_dir, exist_ok=True)

    layer_labels = {'hidden': 'actionRNN', 'base_hidden': 'distanceRNN'}
    for layer, label in layer_labels.items():
        arr = prepared[layer]
        if arr is None:
            continue
        # Single-CTRNN models still write base_hidden but it's constant/zero.
        if arr.ndim != 3 or arr.shape[-1] == 0 or np.allclose(arr.std(0), 0):
            continue
        by_dist = split_by_signed_distance(arr, prepared['signed'])
        smoothed = smooth_and_aggregate(by_dist, sigma=sigma, truncate=truncate)
        embedded, _ = fit_pca(smoothed, n_components=n_components)

        # Emit one file per predefined viewpoint (paper-style: pick the one
        # that best shows the divergence).
        for (elev, azim) in viewpoints.get(label, [(30, 45)]):
            save_path = os.path.join(
                save_dir, f'{label}_S{seed}_elev{elev}_azim{azim}.pdf')
            visualize_specific_view(embedded, save_path, elev=elev, azim=azim)


def plot_pca_for_sweep(mat_dir, save_root, model=None,
                       sigma=DEFAULT_SIGMA, truncate=DEFAULT_TRUNCATE,
                       n_components=DEFAULT_N_COMPONENTS,
                       viewpoints=None):
    """Iterate every ``.mat`` file matching ``model`` and emit per-seed PCA plots.

    Args:
        mat_dir: dir with the ``.mat`` exports (e.g.
            ``results/ReLU/mat_exports``).
        save_root: root under which per-model PDFs land as
            ``<save_root>/<model>/{actionRNN,distanceRNN}_S<seed>_elevE_azimA.pdf``.
        model: optional exact-prefix filter (``'VHA-ReLU-D'`` /
            ``'VHA-ReLU'``); if None, processes all ``.mat`` files.
        sigma, truncate, n_components, viewpoints: forwarded.
    """
    pattern = os.path.join(mat_dir, f'{model}_*.mat' if model else '*.mat')
    paths = sorted(glob.glob(pattern))
    if model == 'VHA-ReLU':
        # VHA-ReLU-D also matches "VHA-ReLU_" prefix -- exclude explicitly.
        paths = [p for p in paths if 'VHA-ReLU-D_' not in os.path.basename(p)]

    for p in paths:
        _model, seed = _extract_model_seed(p)
        save_dir = os.path.join(save_root, _model)
        plot_pca_for_mat(p, save_dir, seed=seed, model=_model,
                         sigma=sigma, truncate=truncate,
                         n_components=n_components, viewpoints=viewpoints)
        print(f'  {_model} seed {seed:2d} -> {save_dir}')


def visualize_rotation_grid(embedded_by_distance, save_path,
                             angles=None, title='',
                             cmap_name='Reds', figsize_scale=1.5):
    """Multi-viewpoint grid on one page, mirroring
    ``pca_utils/visualize_various_viewpoint.py:visualize_rotation_grid``.

    Rows = elev, cols = azim over ``angles`` (default 0..360 step 30 -> 12x12).
    Same visual style as :func:`visualize_specific_view` (transparent panes,
    grid off, tick labels off, Reds cmap, start=circle, end='x').
    """
    if angles is None:
        angles = np.arange(0, 360, 30)
    n = len(angles)
    distances = sorted(embedded_by_distance.keys())
    fig, axes = plt.subplots(
        n, n, subplot_kw={'projection': '3d'},
        figsize=(n * figsize_scale, n * figsize_scale),
        facecolor='white',
    )
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.4, 1, len(distances)))
    for r, elev in enumerate(angles):
        for c, azim in enumerate(angles):
            ax = axes[r, c]
            ax.view_init(elev=elev, azim=azim)
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.grid(False)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            for color, d in zip(colors, distances):
                traj = embedded_by_distance[d]
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c=color, alpha=0.7, lw=0.8)
                ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c=[color], s=8, marker='o')
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c=[color], s=12, marker='x')
            if r == 0:
                ax.set_title(f'azim={azim}', fontsize=8)
            if c == 0:
                ax.set_ylabel(f'elev={elev}', fontsize=8, labelpad=10)
    if title:
        fig.suptitle(title, fontsize=14, y=1.005)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_pca_rotation_grid_for_mat(mat_path, save_dir, seed=None, model=None,
                                    sigma=DEFAULT_SIGMA, truncate=DEFAULT_TRUNCATE,
                                    n_components=DEFAULT_N_COMPONENTS,
                                    angles=None, trim_tail=0,
                                    filename_suffix=''):
    """End-to-end rotation-grid PCA for one .mat file (one seed).

    Emits one PDF per non-trivial layer at
    ``<save_dir>/{actionRNN, distanceRNN}_S<seed>{filename_suffix}_rotation.pdf``.
    ``trim_tail`` is forwarded to :func:`split_by_signed_distance` and drops the
    final N steps of every trajectory before smoothing.
    """
    if seed is None or model is None:
        _model, _seed = _extract_model_seed(mat_path)
        model = model or _model
        seed = seed if seed is not None else _seed
    prepared = _load_and_prepare(mat_path)
    if prepared is None:
        print(f'  {mat_path}: no successful trials, skipped')
        return
    os.makedirs(save_dir, exist_ok=True)
    layer_labels = {'hidden': 'actionRNN', 'base_hidden': 'distanceRNN'}
    for layer, label in layer_labels.items():
        arr = prepared[layer]
        if arr is None:
            continue
        if arr.ndim != 3 or arr.shape[-1] == 0 or np.allclose(arr.std(0), 0):
            continue
        by_dist = split_by_signed_distance(arr, prepared['signed'], trim_tail=trim_tail)
        if not by_dist:
            continue
        smoothed = smooth_and_aggregate(by_dist, sigma=sigma, truncate=truncate)
        embedded, pca = fit_pca(smoothed, n_components=n_components)
        evr = pca.explained_variance_ratio_
        tag = f' trim_tail={trim_tail}' if trim_tail else ''
        title = (f'{model}  seed {seed}  {label}{tag}\n'
                 f'PC1..3 EVR = {evr[0]:.2f}, {evr[1]:.2f}, {evr[2]:.2f}')
        out = os.path.join(save_dir, f'{label}_S{seed}{filename_suffix}_rotation.pdf')
        visualize_rotation_grid(embedded, out, angles=angles, title=title)


__all__ = [
    'DEFAULT_VIEWPOINTS',
    'LANDMARK_STEP',
    'split_by_signed_distance',
    'smooth_and_aggregate',
    'fit_pca',
    'visualize_specific_view',
    'visualize_rotation_grid',
    'plot_pca_for_mat',
    'plot_pca_for_sweep',
    'plot_pca_rotation_grid_for_mat',
]
