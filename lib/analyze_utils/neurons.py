"""Per-neuron variance, ranking, and population-speed analyses."""
import os

import numpy as np
import torch
import scipy.signal
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def analyze_var(hiddens: list, threshold: float = 0.9) -> None:
    """Compute and report per-PC explained-variance ratio of the hidden states.

    Args:
        hiddens: List of per-trial hidden states. ``hiddens[i]`` is shaped
            ``(T_i, 1, n_units)``; all trials are truncated to the shortest
            ``T_i`` before stacking.
        threshold: Cumulative-variance threshold (e.g. 0.9 -> "90% of
            variance") used to report the number of components needed.

    Returns:
        None. Prints the component count required to hit ``threshold``.
    """
    L = np.asarray([len(h) for h in hiddens])
    minL = L.min()
    onset_hiddens = torch.stack([hidden[:minL] for hidden in hiddens])[:,:,0]
    offset_hiddens = torch.stack([hidden[-minL:] for hidden in hiddens])[:,:,0]
    
    for hiddens in [onset_hiddens, offset_hiddens]:
        print("Calculate")
        counts = []
        for i in range(hiddens.shape[1]):
            _hiddens = hiddens[:,i]
            pca_data = torch.pca_lowrank(_hiddens, niter=10, q=min(_hiddens.shape[0], _hiddens.shape[1]))
            U, S, V = pca_data
            cov = S**2 / (len(S) - 1)
            cov = cov / cov.sum()
            cum_cov = cov.cumsum(0)
            count = (cum_cov > threshold).nonzero()[0,0] + 1
            counts.append(count.item())
        print(counts)
    breakpoint()




def analyze_neurons(hiddens: list, save_dir: str = './', _format: str = 'pdf') -> None:
    """Rank individual neurons by across-trial variance + save per-neuron traces.

    Args:
        hiddens: List of per-trial hidden states. ``hiddens[i]`` has shape
            ``(T_i, 1, n_units)``. Torch tensors are converted to numpy.
        save_dir: Output directory for the per-neuron trace plots.
        _format: Figure extension. Default ``'pdf'``.

    Returns:
        None. Picks the second-longest trial length as the common window,
        sorts neurons by across-trial variance, and writes one trace plot
        per top-ranked neuron under ``save_dir``.
    """
    L = np.asarray([len(h) for h in hiddens])
    print(L)
    unique_L = np.unique(L)
    if len(unique_L) > 1:
        target = unique_L[-2]
    else:
        target = unique_L[0]
    if type(hiddens[0]) is not np.ndarray:
        hiddens = [e.cpu().numpy() for e in hiddens]
    for target in np.unique(L):
        indices = (L == target).nonzero()[0]
        _hiddens = [hiddens[i] for i in indices]
        hidden = np.mean(_hiddens, 0)
        if type(hidden) is not np.ndarray:
            hidden = hidden.cpu().numpy()

        XX = int(hidden.shape[-1] ** 0.5)
        YY = int(np.ceil(hidden.shape[-1]/ XX))
        fig = plt.figure(figsize=(YY*3, XX*3))
        gs = gridspec.GridSpec(XX, YY, figure=fig)
        for hid_id in tqdm(range(hidden.shape[-1])):
            corr = []
            hid = hidden[:, 0, hid_id] 
            i = hid_id // YY
            j = hid_id % YY
            ax = fig.add_subplot(gs[i, j])
            X = np.arange(len(hidden))
            for h in _hiddens:
                ax.plot(X, h[:,0,hid_id], alpha=0.1, color='grey')
            ax.plot(X, hid, color='blue')
            peak_idx = scipy.signal.find_peaks(hid)[0]
#            for pi in peak_idx:
#                ax.text(X[pi], hid[pi], pi)
            peak_idx = scipy.signal.find_peaks(-hid)[0]
#            for pi in peak_idx:
#                ax.text(X[pi], hid[pi], pi, color='red')
        fig.savefig(os.path.join(save_dir,f'hidden (L={target}).{_format}'))
    plt.close()



def analyze_speed(hiddens: list, save_dir: str = './') -> None:
    """Plot population speed (mean ||h[t+1] - h[t]||) per trial.

    Args:
        hiddens: List of per-trial hidden states. ``hiddens[i]`` has shape
            ``(T_i, 1, n_units)``. Torch tensors are converted to numpy.
        save_dir: Output directory.

    Returns:
        None. Picks the second-longest trial length as the common window,
        computes the across-neuron mean of consecutive-step L2 norms, and
        saves speed-vs-time plots grouped by traveled distance into
        ``save_dir``.
    """
    L = np.asarray([len(h) for h in hiddens])
    print(L)
    target = np.unique(L)[-2]
    if type(hiddens[0]) is not np.ndarray:
        hiddens = [e.cpu().numpy() for e in hiddens]

    distance = []
    targets = []
    all_distance = []
    raw_distance = []
    avg_distance = []
    save_dir = os.path.join(save_dir, 'speed')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for target in np.unique(L):
        indices = (L == target).nonzero()[0]
        _hiddens = np.stack([hiddens[i] for i in indices])[:,:,0]
        print("DISTANCE", target-1)
        dist = [np.linalg.norm(_hiddens[:, i] - _hiddens[:, i+1], axis=-1) for i in range(_hiddens.shape[1]-1)]
        raw_dist = [abs(_hiddens[:, i] - _hiddens[:, i+1]) for i in range(_hiddens.shape[1]-1)]
        all_dist = np.stack(dist)
        raw_dist = np.stack(raw_dist) 
        raw_distance.append(raw_dist)
        all_distance.append(all_dist)

        hidden = np.mean(_hiddens, 0)
        if type(hidden) is not np.ndarray:
            hidden = hidden.cpu().numpy()
        
        # l2 distance
        dist = [np.linalg.norm(hidden[i] - hidden[i+1], axis=-1) for i in range(len(hidden)-1)]
        dist = np.asarray(dist).reshape(-1)
        print("Average Dist between each Data Point", dist)
        avg_distance.append(dist)
        dist = dist.mean()
        distance.append(dist)
        targets.append(target)
        print("Average Distance", dist, all_dist.mean())

    fig = plt.figure()
    plt.scatter((np.asarray(targets) / 12).astype(int), distance)

    plt.xlabel('|Strat - Target|')
    plt.ylabel('Average Speed')
    plt.title('Average Speed for each |Start-Target|')
    plt.savefig(os.path.join(save_dir, 'speed.pdf'))

    for i, distance in enumerate(all_distance): 
        fig = plt.figure()
        for d in distance.T:
            t = len(d)
            plt.plot(np.arange(t) - t+1, d, alpha=1/distance.shape[1])  #, color='grey')
        plt.xlabel('Time')
        plt.ylabel('Average Speed')
        plt.title(f'Average Speed for each trajectory at Start-Target Distance {i}')
        plt.savefig(os.path.join(save_dir, f'AvgSpeedTrajDist_{i}.pdf'))

    for i, distance in enumerate(raw_distance):
        fig = plt.figure()
        # same distance (data point, N, Dim)
        speed = distance.mean(1)
        for d in speed.T:
            t = len(d)
            plt.plot(np.arange(t) - t+1, d, alpha=0.1, color='grey')
        plt.xlabel('Time')
        plt.ylabel('Average Speed')
        plt.title(f'Average Speed for each Neuron at Start-Target Distance {i}')
        plt.savefig(os.path.join(save_dir, f'AvgSpeedDist_{i}.pdf'))
    # visualize raw distance

    # visualize based on distance.
    fig = plt.figure(figsize=(16,4))
    for d in avg_distance:
        t = len(d)
        plt.plot(np.arange(t) - t+1, d, label=f'Distance: {int(t/12)}', alpha=0.75)
    plt.ylabel('Average Speed')
    plt.xlabel('Time')
    plt.title('Average Speed for each Start-Target Distance')
    plt.legend()
    fig.savefig(os.path.join(save_dir, 'distance.pdf'))

    fig = plt.figure()
    for d in avg_distance:
        d = d.reshape(-1, 12).mean(-1)
        t = len(d)
        plt.plot(np.arange(t) - t+1, d, label=f'Distance: {t}', alpha=0.75)
    plt.ylabel('Average Speed')
    plt.xlabel('Chunk (Image-Image)')
    plt.title('Average Speed for each Start-Target Distance')
    plt.legend()
    fig.savefig(os.path.join(save_dir, 'AvgSpeedChunk.pdf'))




