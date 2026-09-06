"""Temporal-distance vs state-space speed.

For each (seed, model, layer) we walk every successful trial, compute
per-timestep speed s(t) = ||h[t+1] - h[t]||, and pair each s(t) with the
remaining distance to target at t (in landmark units: raw_time / 12).
Then bin by remaining distance across all trials of a seed and take the
mean speed per bin. Plot speed vs distance-remaining.

The paper's related analysis (analyze_speed in lib/analyze_utils/neurons.py)
already bins by whole-trial distance; this script bins WITHIN a trial
by the currently-remaining distance so the reader can see if the RNN
modulates its state-space speed as it closes in on the target.

Outputs (at results/ReLU/speed_vs_distance/):
    <model>_<layer>.csv         long-form (seed, dist, speed_mean, n)
    <model>_<layer>.pdf         mean +- SEM across seeds
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns


MAT_DIR = 'results/ReLU/mat_exports'
SAVE_DIR = 'results/ReLU/speed_vs_distance'
LANDMARK_STEP = 12  # raw timesteps per landmark unit


def _iter_valid_slots(mat_path):
    """Yield (seed, layer_name, hidden) for each usable (mat, layer) pair.

    ``hidden`` is (n_trials, T_max, n_units) with NaN padding preserved.
    """
    m = scipy.io.loadmat(mat_path)
    suc = m['suc'].squeeze().astype(bool)
    if not suc.any():
        return
    trajs = m['trajs'][suc]
    seed_match = re.search(r'S(\d+)Ep', os.path.basename(mat_path))
    if not seed_match:
        return
    seed = int(seed_match.group(1))
    layers = {'actionRNN': m['hidden']}
    if 'base_hidden' in m:
        layers['distanceRNN'] = m['base_hidden']
    for name, h in layers.items():
        yield seed, name, h[suc], trajs


def _per_seed_bins(hidden, trajs):
    """Bin per-timestep speed by remaining distance (landmark units)."""
    pairs = []
    for i in range(hidden.shape[0]):
        signed = int(trajs[i, 1] - trajs[i, 0])
        if signed == 0:
            continue
        # Right-padded: valid rows are the LAST |d|+1 timesteps.
        window = abs(signed) + 1
        h = hidden[i, -window:]     # (window, n_units)
        # remaining raw-distance at time t: window - 1 - t (t=0 is landmark
        # far from target; t=window-1 is target).
        for t in range(window - 1):
            speed = np.linalg.norm(h[t + 1] - h[t])
            remaining_raw = (window - 1 - t)  # in raw timesteps
            # bin into landmark units (12 raw steps each).
            bin_idx = remaining_raw // LANDMARK_STEP
            pairs.append((bin_idx, float(speed)))
    if not pairs:
        return pd.DataFrame(columns=['dist_bin', 'speed'])
    df = pd.DataFrame(pairs, columns=['dist_bin', 'speed'])
    return df.groupby('dist_bin')['speed'].agg(['mean', 'std', 'count']).reset_index()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    per_layer_rows = {}
    for mp in sorted(glob.glob(os.path.join(MAT_DIR, 'VHA-ReLU*.mat'))):
        model = 'VHA-ReLU-D' if 'VHA-ReLU-D_' in os.path.basename(mp) else 'VHA-ReLU'
        for seed, layer, hidden, trajs in _iter_valid_slots(mp):
            binned = _per_seed_bins(hidden, trajs)
            if binned.empty:
                continue
            binned.insert(0, 'seed', seed)
            binned.insert(0, 'model', model)
            binned.insert(0, 'layer', layer)
            per_layer_rows.setdefault((model, layer), []).append(binned)

    for (model, layer), chunks in per_layer_rows.items():
        df = pd.concat(chunks, ignore_index=True)
        tag = f'{model}_{layer}'
        df.to_csv(os.path.join(SAVE_DIR, f'{tag}.csv'), index=False)

        fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=200)
        # Mean +- SEM across seeds per bin.
        sns.lineplot(data=df, x='dist_bin', y='mean', errorbar='se', ax=ax,
                     marker='o', color='#33669a')
        ax.set_xlabel('Remaining distance (landmark units)')
        ax.set_ylabel(f'State-space speed  ({layer})')
        ax.set_title(f'{model}  {layer}  (mean±SEM across seeds)')
        plt.tight_layout()
        fig.savefig(os.path.join(SAVE_DIR, f'{tag}.pdf'))
        plt.close(fig)
        print(f'wrote {SAVE_DIR}/{tag}.pdf ({df["seed"].nunique()} seeds, '
              f'bins={sorted(df["dist_bin"].unique())})')


if __name__ == '__main__':
    main()
