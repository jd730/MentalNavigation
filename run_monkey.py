"""
Run firing rate shift and stretch analysis on monkey single-neuron data.
Loads hiddens_onset.pkl from monkey_single_neurons/ and runs all three
alignment options, saving results to results/monkey/.
"""
import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.analyze_utils import run_firing_rate_shifted, run_firing_rate_stretched

BASE    = os.path.dirname(os.path.abspath(__file__))
PKL_DIR = os.path.join(BASE, 'monkey_single_neurons')
SAVE_DIR = os.path.join(BASE, 'results', 'monkey')
os.makedirs(SAVE_DIR, exist_ok=True)

# Load onset-aligned hiddens
with open(os.path.join(PKL_DIR, 'hiddens_onset.pkl'), 'rb') as f:
    data = pickle.load(f)

hiddens      = data['hiddens']
neuron_names = data['neuron_names']
dists        = data['dists']

print(f'Loaded {len(hiddens)} distance conditions, {hiddens[0].shape[2]} neurons')
print(f'Neurons: {neuron_names}')
print(f'Distances: {dists}')
for i, h in enumerate(hiddens):
    print(f'  dist={dists[i]}: shape={h.shape}')

base_shift_opts = {
    'ignore_last':       False,   # monkey data is already trimmed
    'smoothing_sigma':   2,
    'smoothing_truncate': 2,
}

base_stretch_opts = {
    'ignore_last':       False,
    'smoothing_sigma':   2,
    'smoothing_truncate': 2,
    'sx_range':          np.linspace(0.5, 4.0, 71),
}

for alignment in ('option1', 'option2', 'option3'):
    print(f'\n===== alignment={alignment} =====')

    shift_opts = {**base_shift_opts, 'alignment': alignment}
    run_firing_rate_shifted(
        hiddens, SAVE_DIR,
        save_indiv=True, _format='pdf',
        options=shift_opts,
    )

    stretch_opts = {**base_stretch_opts, 'alignment': alignment}
    run_firing_rate_stretched(
        hiddens, SAVE_DIR,
        save_indiv=True, _format='pdf',
        options=stretch_opts,
    )

print('\nDone.')
