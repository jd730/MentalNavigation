import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from grid_cells import sensory_remap as remap
from .surprisal import SurprisalLinear, SurprisalExponential, SurprisalExponentialBVC, SurprisalRegion

def plot_grid_results_2d(encoded_pos, figsize=(10, 5)):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    for grid_pos_new in encoded_pos[0]:
        ax[0].plot(grid_pos_new)
    for grid_pos_new in encoded_pos[1]:
        ax[1].plot(grid_pos_new)
    ax[0].set(xlabel='Real position along path',
             ylabel='Position coded by grid pattern, x')
    ax[1].set(xlabel='Real position along path',
             ylabel='Position coded by grid pattern, y')
    fig.tight_layout()
    return fig, ax

def plot_grid_space(encoded_pos, c=None, fixed_frame=False, Npos=210, **kwargs):
    nruns = encoded_pos.shape[1]
    nrows = int(np.ceil(nruns / 3))
    fig, ax = plt.subplots(nrows, 3, figsize=(12, nrows*4))

    for run in range(nrows*3):
        if run < nruns:
            encoded_pos_run = encoded_pos[:, run, :]    
            if c is None:
                c = np.arange(encoded_pos_run.shape[1])

            ax[run//3][run%3].scatter(x=encoded_pos_run[0], y=encoded_pos_run[1], 
                                    c=c, **kwargs)
            ax[run//3][run%3].plot(encoded_pos_run[0], encoded_pos_run[1], linewidth=0.5)
            ax[run//3][run%3].set(title=f'Run {run+1}')

            if fixed_frame:
                ax[run//3][run%3].set(xlim=(0, Npos), ylim=(0, Npos))
        else:
            ax[run//3][run%3].axis('off')

    return fig, ax

def plot_grid_space_per_axis(encoded_pos, red_lines=[], green_lines=[]):
    nruns = encoded_pos.shape[1]
    fig, ax = plt.subplots(nruns, 2, figsize=(12, nruns*5))
    for run in range(nruns):
        x, y = encoded_pos[:, run, :]
        ax[run][0].plot(x)
        ax[run][1].plot(y)
        for i in range(2):
            for j in red_lines:
                ax[run][i].axvline(x=j, c='red', linestyle=':')
            for j in green_lines:
                ax[run][i].axvline(x=j, c='green', linestyle=':')
        ax[run][0].set(title=f'Run {run+1}, x')
        ax[run][1].set(title=f'Run {run+1}, y')
        ax[run][0].set(xlabel='Real position along path',
             ylabel='Position coded by grid pattern, x')
        ax[run][1].set(xlabel='Real position along path',
             ylabel='Position coded by grid pattern, y')
    return fig, ax

# only for surprisal region
def find_remap_segments(surprisal_history, threshold, return_threshold=None, run=0):
    if return_threshold is None:
        return_threshold = threshold
    condition_enter = lambda i: surprisal_history[run, i] >= threshold
    condition_exit = lambda i: surprisal_history[run, i] < return_threshold
    n = len(surprisal_history[run])
    return find_remap_segments_master(n, condition_enter, condition_exit)

def find_remap_segments_master(n, condition_enter, condition_exit):
    segment_ids = []
    id_ = 0
    on = False
    surprisal_exit_indices = []
    surprisal_enter_indices = []
    for i in range(n):
        if i == 0: # first point is fake
            segment_ids.append(0)
            continue
        if (not on) and condition_enter(i):
            on = True
            surprisal_enter_indices.append(i)
        elif on and condition_exit(i):
            on = False
            surprisal_exit_indices.append(i)
            id_ += 1
        if on:
            segment_ids.append(-1)
        else:
            segment_ids.append(id_)
    return surprisal_exit_indices, surprisal_enter_indices, segment_ids

def get_segment_ids(surprisal_enter_indices, surprisal_exit_indices, n):
    condition_enter = lambda i: i in surprisal_enter_indices
    condition_exit = lambda i: i in surprisal_exit_indices
    return find_remap_segments_master(n, condition_enter, condition_exit)[2]
    
def plot_env(env, ax=None, figsize=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    lines = env['segs'].reshape(-1, 2, 2)
    lc = mc.LineCollection(lines, linewidths=2)
    ax.add_collection(lc)
    ax.autoscale()
    return ax

def plot_path(env, path, figsize=(8, 8), s=None, c='default', cmap=None, ax=None,
             **kwargs):
    ax = plot_env(env, ax, figsize)
    if c == 'default':
        c = np.arange(path.shape[1])
    ax.scatter(*path, s=s, marker='o', c=c, cmap=cmap, **kwargs);
    return ax

def plot_path_surprisal(env, path_locations, surprisal_history, threshold=None, 
                       run=0, s=None):
    c = surprisal_history[1:, run]
    if threshold is not None:
        c = c >= threshold
    plot_path(env, path_locations[2:, ].T, c=c, cmap='Reds', s=s)

def plot_remapping_points(env, path_locations, surprisal_exit_indices, s=None, ax=None, c='red', **kwargs):
    return plot_path(env, path_locations[:, surprisal_exit_indices], c=c, s=s, ax=ax, **kwargs)

# this works for return_threshold != threshold too
# this does not include remapping points
def plot_high_surprisal_points(env, path_locations, segment_ids, s=None, ax=None, c='red', **kwargs):
    return plot_path(env, path_locations[:, np.array(segment_ids) == -1], c=c, s=s, ax=ax, **kwargs)

def compute_surprisal(sbook, surprisal_method, tau,
                     Npatts_test=None, normalized=False, flipped=False,
                     b=0, predict=False, vs=None):
    # this function only returns the raw surprisal 
    threshold = 0
    return_threshold = 0
    nruns = sbook.shape[0]
    if Npatts_test is None:
        Npatts_test = sbook.shape[2]
    if surprisal_method == 'linear':
        surprisal_engine = SurprisalLinear(tau, threshold, nruns, normalized, flipped)
    elif surprisal_method == 'exponential':
        surprisal_engine = SurprisalExponential(tau, threshold, nruns, normalized, flipped)
    elif surprisal_method == 'exponential_bvc':
        surprisal_engine = SurprisalExponentialBVC(b, tau, threshold, nruns, normalized, flipped)
    surprisal_engine = SurprisalRegion(surprisal_engine, threshold, return_threshold)
    surprisal_history = np.zeros((nruns, Npatts_test)) # store raw surprisal
    for x in range(1, Npatts_test): 
        strue = sbook[:,:,x,None]
        surprisal_input = strue.squeeze(axis=-1)
        if x == 1:
            strue_start = sbook[(slice(None), slice(None), 0, None)] 
            _ = surprisal_engine.visit(strue_start.squeeze(axis=-1), update_state=False)
        if predict:
            v = (0,) + tuple(vs[x])
            end_current_map, start_new_map, raw_surprisal = surprisal_engine.visit(surprisal_input, v=v)
        else:
            end_current_map, start_new_map, raw_surprisal = surprisal_engine.visit(surprisal_input)
        surprisal_history[:, x] = raw_surprisal.squeeze()
    return surprisal_history

def plot_segments_from_surprisal(env, path_locations, segment_ids):
    n = max(segment_ids) + 1
    nrows = int(np.ceil(n / 5))
    fig, ax = plt.subplots(nrows, 5, figsize=(20, nrows*4))
    for i in range(n):
        plot_path(env, path_locations[:, np.array(segment_ids) == i], ax=ax[i//5][i%5])
    for i in range(n, nrows*5):
        ax[i//5][i%5].axis('off')
    return fig, ax

def compute_segment_centroids(encoded_pos, segment_ids, segment_to_compartment, compartments=(0, 1, -1)):
    # compartments = sorted(set(segment_to_compartment.values()) - {None})
    compartment_to_segments = defaultdict(list)
    for seg, comp in segment_to_compartment.items():
        compartment_to_segments[comp].append(seg)
    results = []
    for compartment in compartments:
        pos_segments = []
        for segment in compartment_to_segments[compartment]:
            selection = (np.array(segment_ids) == segment)
            selection[0] = False # 1st location is fake :)
            if np.all(~selection):
                continue
            pos = encoded_pos[:, :, selection]
            pos_segments.append(np.mean(pos, axis=2))
        pos_segments = np.array(pos_segments)
        results.append(pos_segments)
    return results

def compute_segment_shifts(encoded_pos, surprisal_exit_indices, segment_to_compartment,
                   path_locations_discretized, compartments):
    compartment_to_segments = defaultdict(list)
    for seg, comp in segment_to_compartment.items():
        compartment_to_segments[comp].append(seg)
    results = []
    for compartment in compartments:
        shifts = []
        for segment in compartment_to_segments[compartment]:
            if segment == 0:
                if surprisal_exit_indices[0] <= 5:
                    print("Skipping first segment because it's too short!")
                    continue
                start = 1
            else:
                start = surprisal_exit_indices[segment-1]
            anchor_real = path_locations_discretized[start]
            anchor_grid = encoded_pos[:, :, start]
            shift = anchor_grid - anchor_real[:, None]
            shifts.append(shift)
        shifts = np.array(shifts)
        results.append(shifts)
    return results

def plot_segment_info(segment_info, alpha=0.2, s=150, colors=('red', 'green', 'blue')):
    # segment_info: list (length=#compartments), 
    # each elt has shape (#segments, 2, nruns)
    n_compartments = len(segment_info)
    if n_compartments > len(colors):
        print('Update colors please!')
        return
    
    nruns = segment_info[0].shape[2]
    nrows = int(np.ceil(nruns / 3))
    fig, ax = plt.subplots(nrows, 3, figsize=(12, nrows*4))
    for run in range(nruns):
        for i in range(n_compartments):
            color = colors[i]
            if color is None:
                continue
            ax[run//3][run%3].scatter(*np.array(segment_info[i])[:, :, run].T, 
                                      color=color, alpha=alpha, s=s)
        ax[run//3][run%3].set(title=f'Run {run+1}')
    for i in range(nruns, nrows*3):
        ax[i//3][i%3].axis('off')
    return fig, ax

def irregular_jumps(encoded_pos, high_surprisal_indices):
    nruns = encoded_pos.shape[1]
    indices = [[] for i in range(nruns)]
    for i in range(2, encoded_pos.shape[2]):
        if i not in high_surprisal_indices:
            for run in range(nruns):
                vec = np.abs(encoded_pos[:, run, i] - encoded_pos[:, run, i-1])
                if tuple(vec) not in [(0, 1), (1, 0), (0, 209), (209, 0)]:
                    indices[run].append(i)
    return indices