"""Autocorrelation analysis and periodicity peak detection."""
import os
import pickle

import numpy as np
import torch
import scipy.signal
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def run_autocorr(hiddens: list, save_dir: str, vis: bool = True, image_interval: int = 12, filename: str = 'temp.npy', lambdas: list = [11, 12, 13], _format: str = 'pdf', first_half: bool = False, options: dict = {}) -> None:
    """Compute per-unit hidden-state autocorrelation and detect periodicity peaks.

    Args:
        hiddens: List of trial hidden-state arrays. ``hiddens[i]`` has
            shape ``(T_i, 1, n_units)``.
        save_dir: Directory to write figures and the npy cache into.
        vis: Whether to render the autocorrelation-trace and peak-interval
            histogram figures. Set ``False`` to only update the cache.
        image_interval: Per-image stimulus duration in timesteps, used to
            convert peak indices into image-period units on the x axis.
        filename: Cache filename for the (xs, corrs) tuple, written under
            ``save_dir``.
        lambdas: Target grid-cell periods (in image units). Vertical
            reference lines are drawn at each lambda on the histogram.
        _format: Figure extension. Default ``'pdf'``.
        first_half: If True, restrict the autocorr window to the first
            half of each trial.
        options: Extra knobs. Supported keys: ``'use'`` (skip when False),
            ``'direction'`` (``'left'``/``'right'``/``''`` to filter
            unidirectional trials), ``'ylim'`` (y-axis range).

    Returns:
        None. Writes the npy cache and (when ``vis``) the autocorr-trace
        and peak-interval histogram figures into ``save_dir``.
    """
    if not options.get('use', True):
        return

    direction = options.get('direction', '')
    if len(hiddens) == 0:
        print("No Hidden States")
        return None
    hid = hiddens[0][:,0]
    coors = []
    if vis:
        XX = int(hid.shape[-1] ** 0.5)
        YY = int(np.ceil(hid.shape[-1]/ XX))
        fig = plt.figure(figsize=(YY*3, XX*3))
        gs = gridspec.GridSpec(XX, YY, figure=fig)
    
    corrs = []
    data = []
    peak_indices  = []

    for hid_id in tqdm(range(hiddens[0].shape[-1])):
        corr = []
        xs = []
        if vis:
            i = hid_id // YY
            j = hid_id % YY
            ax = fig.add_subplot(gs[i, j])

        for traj_id in range(len(hiddens)):
            hid = hiddens[traj_id][:, 0, hid_id]
            
            if first_half:
                hid = hid[:(len(hid)+1)//2]

            if type(hid) is torch.Tensor:
                hid = hid.cpu().numpy()
            _corr = np.correlate(hid, hid, mode='full')
            corr.append(_corr)
            xs.append(np.arange(len(_corr)) - len(_corr) // 2)
            if vis:
                ax.plot(xs[-1], corr[-1], alpha=0.1, color='grey')
        corrs.append(corr)
        X = np.concatenate(xs)
        Y = np.concatenate(corr)
        X_unique = np.unique(X)
        Y_avg = [np.mean(np.array(Y)[X == x]) for x in X_unique]
        peak_idx = scipy.signal.find_peaks(Y_avg)[0]
        peak_indices.append(peak_idx)

        if vis and False:
            ax.plot(X_unique, Y_avg, linewidth=2.5)
            x_peaks = X_unique[peak_idx]
            Y_avg = np.asarray(Y_avg)
            y_peaks = Y_avg[peak_idx]
            for pi in peak_idx:
                ax.text(X_unique[pi], Y_avg[pi], pi - len(_corr) // 2)
            ax.plot(x_peaks, y_peaks, color='red', linestyle='None')

            ax.set_yticklabels([])
#                        fig.savefig('auto_corr_all.pdf')
    
#    analyze_periodicity(peak_indices) # there is an artifact if we aggregate all.
    if first_half:
        auto_corr_dir = os.path.join(save_dir, 'half_auto_corr')
        with open(save_dir+'_half' + '.pkl', 'wb') as f:
            pickle.dump((xs, corrs), f)
        print(os.path.join(save_dir + '_half' + '.pkl'))
    else:
        auto_corr_dir = os.path.join(save_dir, 'auto_corr')
        with open(save_dir + '.pkl', 'wb') as f:
            pickle.dump((xs, corrs), f)
        print(os.path.join(save_dir + '.pkl'))
    if not os.path.exists(auto_corr_dir):
        os.mkdir(auto_corr_dir)
    # auto_corr_all.pdf skipped -- not used in the paper; avg_of_avg.pdf below is
    # the load-bearing periodicity histogram.
    plt.close(fig)

    lengths = np.asarray([len(hidden) for hidden in hiddens])
    total_intervals = []
    vis = False
    for l in np.unique(lengths):
        peak_indices = []
        if vis:
            fig = plt.figure(figsize=(YY*3, XX*3))
            gs = gridspec.GridSpec(XX, YY, figure=fig)
            fig2 = plt.figure(figsize=(YY*3, XX*3))
            gs2 = gridspec.GridSpec(XX, YY, figure=fig2)
        mask = lengths == l
        for hid_id, corr in enumerate(corrs):
            i = hid_id // YY
            j = hid_id % YY
            corr = [corr[i] for i in range(len(corr)) if mask[i]] 

            X = np.arange(2*l - 1) - l
            Y = np.asarray(corr).mean(0)

            fft = np.fft.fft(Y)

            peak_idx = scipy.signal.find_peaks(Y)[0]
            peak_indices.append(peak_idx)
            x_peaks = X[peak_idx]
            y_peaks = Y[peak_idx]
            if vis:
                ax = fig.add_subplot(gs[i, j])
                ax2 = fig2.add_subplot(gs2[i, j])
                for _corr in corr:
                    ax.plot(X, _corr, alpha=0.1, color='grey')
                ax.plot(X, Y, linewidth=2.5)
                ax2.plot(X, fft, linewidth=2.5)
#                for pi in peak_idx:
#                    ax.text(X[pi], Y[pi], pi)
                ax.plot(x_peaks, y_peaks, color='red', linestyle='None')

                ax.set_yticklabels([])

        # period_{l}.pdf skipped -- not used in the paper. Passing name=None
        # keeps the intervals computation (needed for the npy dump below).
        intervals = analyze_periodicity(peak_indices, name=None)
        total_intervals.append(intervals)
        if vis:
            fig.savefig(os.path.join(auto_corr_dir,f'auto_corr_L={l}.{_format}'))
            fig2.savefig(os.path.join(auto_corr_dir,f'FFT_auto_corr_L={l}.{_format}'))
            print(os.path.join(auto_corr_dir,f'auto_corr_L={l}.{_format}'))
    
    total_intervals = np.asarray(total_intervals)

    if first_half:
        interval_dir = '{}/{}half_periodicity/{}'.format(save_dir, direction, '_'.join([str(e) for e in lambdas]))
    else:
        interval_dir = '{}/{}periodicity/{}'.format(save_dir, direction, '_'.join([str(e) for e in lambdas]))
    if not os.path.exists(interval_dir):
        os.makedirs(interval_dir)
    np.save(os.path.join(interval_dir, filename), total_intervals)
    print(os.path.join(interval_dir, filename))

    avg_intervals = total_intervals.sum(0) / ((total_intervals >0).sum(0) + 1e-16)
    # Exclude avg==0 units from the histogram body (they dominate visually
    # and swamp the real 11..13 peak). Report their count in the title so
    # the reader still sees how much of the population was excluded.
    n_zero_units = int((avg_intervals == 0).sum())
    n_units_total = int(avg_intervals.size)
    plot_intervals = avg_intervals[avg_intervals > 0]
    fig = plt.figure(figsize=(6,4),dpi=200)
    if plot_intervals.size:
        n, bins, patches = plt.hist(plot_intervals, bins=np.arange(plot_intervals.max()+1), color='black', alpha=0.2)
    else:
        n, bins, patches = [], [], []
    plt.title(f'# zero units = {n_zero_units} / {n_units_total}')
    if 'ylim' in options:
        plt.ylim(options['ylim'])
    if 'xlim' in options:
        plt.xlim(options['xlim'])

    print(n, bins, patches)
#    patches[11].set_fc('b')
#    patches[12].set_fc('b')
#    patches[13].set_fc('b')
    locs, labels = plt.xticks()
    plt.xticks([e+0.5 for e in locs[1:]], [int(e) for e in locs[1:]])
#    ticks = [(patch._x0 + patch._x1)/2 for patch in patches]
#    ticklabels = [i for i in range(len(bins))]
#    plt.xticks(ticks, ticklabels)
    plt.xlabel('Perioidicity (a.u.)')
#    plt.ylabel('Probability')
    plt.ylabel('# of RNN units')
    for g in lambdas:
        plt.axvline(x=g+0.5, color='blue')
#    else:
#        plt.axvline(x=11, color='blue')
#        plt.axvline(x=14, color='blue')
#    plt.axvline(x=12, color='gray')
    plt.axvline(x=image_interval + 0.5, color='black')
#    plt.axvline(x=13, color='gray')
    mean = avg_intervals[avg_intervals !=0].mean()
    plt.axvline(x=mean+0.5, color='red', linewidth=1)
    plt.xlim(-0.5, 24)
    print("Intervals")
    print(np.unique(avg_intervals, return_counts=True))
    plt.savefig(os.path.join(auto_corr_dir,f'avg_of_avg.{_format}'))
    plt.close()


def analyze_periodicity(peak_indices, name=None):
    """Compute the across-trial first inter-peak interval per unit.

    Args:
        peak_indices: Sequence (one entry per trial) of arrays of peak
            timestep indices, as produced by ``scipy.signal.find_peaks``.
        name: Optional filename for a diagnostic ``period_l.pdf`` histogram.
            Pass ``None`` to skip the figure (paper convention -- the
            per-unit intervals are consumed downstream via the npy dump).

    Returns:
        np.ndarray: Per-trial mean first inter-peak interval.
    """
    intervals = []
    for peak_idx in peak_indices:
        interval = [peak_idx[i+1] - peak_idx[i] for i in range(len(peak_idx)-1)]
        interval = interval[:1]
        if len(interval) == 0:
            interval = [0]
        intervals.append(interval)
    avg_intervals = np.asarray([np.mean(interval) for interval in intervals])
    if name is not None:
        fig = plt.figure(figsize=(6, 4), dpi=200)
        plt.hist(avg_intervals, bins=np.arange(avg_intervals.max() + 1), density=True)
        plt.xlabel('Perioid')
        plt.ylabel('Frequency')
        fig.savefig(name)
        plt.close(fig)
    return avg_intervals



