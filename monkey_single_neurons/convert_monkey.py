"""
Convert monkey single-neuron CSV files into the hiddens list format used by
run_firing_rate_shifted / run_firing_rate_stretched.

Format: list of numpy arrays, one element per distance condition.
Each array has shape [T, 1, D] where:
  T = number of time steps for that distance (min across neurons, 0.04s bins)
  1 = batch dimension (always 1, matching RNN convention)
  D = number of neurons

Usage:
    import pickle
    with open('hiddens_onset.pkl', 'rb') as f:
        hiddens = pickle.load(f)
    # hiddens[i].shape == [T_i, 1, 4]
"""

import os
import pickle
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

FILES = [
    'monkey_EC_neur13_sess1.csv',
    'monkey_EC_neur2_sess5.csv',
    'monkey_PPC_neur04_sess1.csv',
    'monkey_PPC_neur19_sess1.csv',
]

NEURON_NAMES = [
    'EC_neur13_sess1',
    'EC_neur2_sess5',
    'PPC_neur04_sess1',
    'PPC_neur19_sess1',
]

def load_neuron(fpath, time_col, fr_col):
    """Return dict: dist -> 1-D numpy array of firing rates sorted by time."""
    df = pd.read_csv(fpath)
    result = {}
    for dist, grp in df.groupby('dist'):
        grp = grp.sort_values(time_col)
        result[int(dist)] = grp[fr_col].to_numpy(dtype=np.float32)
    return result


def build_hiddens(time_col, fr_col, out_name):
    # Load all neurons
    all_data = []   # list of {dist: array}
    for fname in FILES:
        fpath = os.path.join(BASE, fname)
        all_data.append(load_neuron(fpath, time_col, fr_col))

    D = len(all_data)
    dists = sorted(all_data[0].keys())

    # Find min time points per distance (across neurons) to get a common grid
    min_pts = {}
    for dist in dists:
        min_pts[dist] = min(len(nd[dist]) for nd in all_data)
        n_pts = [len(nd[dist]) for nd in all_data]
        print(f'  dist={dist}: time pts per neuron = {n_pts}  -> using {min_pts[dist]}')

    # Build one [T, 1, D] array per distance
    hiddens = []
    for dist in dists:
        T = min_pts[dist]
        arr = np.zeros((T, 1, D), dtype=np.float32)
        for ni, nd in enumerate(all_data):
            arr[:, 0, ni] = nd[dist][:T]
        hiddens.append(arr)
        print(f'  dist={dist}: array shape {arr.shape}')

    out_path = os.path.join(BASE, out_name)
    with open(out_path, 'wb') as f:
        pickle.dump({'hiddens': hiddens, 'neuron_names': NEURON_NAMES, 'dists': dists}, f)
    print(f'\nSaved -> {out_path}')
    return hiddens


if __name__ == '__main__':
    print('=== onset-aligned (time_JS_onset / firing_rate) ===')
    hiddens_onset = build_hiddens('time_JS_onset', 'firing_rate', 'hiddens_onset.pkl')

    print('\n=== offset-aligned (time_JS_offset / firing_rate_offset) ===')
    hiddens_offset = build_hiddens('time_JS_offset', 'firing_rate_offset', 'hiddens_offset.pkl')
