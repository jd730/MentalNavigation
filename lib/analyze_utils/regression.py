"""Regression scatter for learned vs target distances."""
import os
from glob import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def run_regression(save_dir: str, epochs: int, _format: str = 'pdf') -> None:
    """Save a 2x5 scatter grid of learned-vs-target distance over training.

    Args:
        save_dir: A directory under the parent run dump; the run's
            ``verbose_info_*.pkl`` files are read from `dirname(save_dir)`.
        epochs: Per-environment epoch count, used to slice the right
            checkpoints out of the dump (steps ``target + N*epochs``).
        _format: Figure extension. Default ``'pdf'``.

    Returns:
        None. Writes ``<dirname(save_dir)>/scatter.<_format>``: a 2x5 grid
        with rows = (visual, mental) and columns = the 5 environments,
        each scatter showing predicted vs ground-truth distance (in steps).
    """
    path = '/'.join(save_dir.split('/')[:-1])
    verbose_info_paths = sorted(glob(os.path.join(path, 'verbose_info*.pkl')))
    targets =[49, 99]

    fig = plt.figure(figsize=(5*3, 2*3))
    gs = gridspec.GridSpec(2, 5, figure=fig)
    for N in range(5):
        for target in targets:
            _path = f'{path}/verbose_info_{target + N*epochs:05}.pkl'
            if not os.path.exists(_path):
                print("Wrong", _path)
                continue
            with open(_path, 'rb') as f:
                data = pickle.load(f)
            for i, (mode, c) in enumerate(zip(['visual', 'mental'], ['black', 'red'])):
                seen = data[f'{mode}/seen/{N}/']
                unseen = data[f'{mode}/unseen/{N}/']
                merged = np.concatenate((seen, unseen))
                X = merged[:,1] - merged[:,0]
                Y = merged[:,2] - merged[:,0]

                ax = fig.add_subplot(gs[i,N])
                ax.set_xlim(-6*12, 6*12)
                ax.set_ylim(-6*12, 6*12)
                lims = [
                    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                ]

                # now plot both limits against eachother
                ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, linewidth=1)
                ax.scatter(X*12, Y*12, color=c, marker='.')

    plt.savefig(path + f'/scatter.{_format}') 
    plt.close()

