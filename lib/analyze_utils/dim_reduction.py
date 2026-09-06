"""Non-linear dimensionality reduction helpers.

Wraps scikit-learn PCA / Isomap / MDS / t-SNE so the analysis pipeline can
switch reduction methods via a single function call.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn.manifold as skmanifold
import sklearn.decomposition as skdecomposition
import torch
from scipy.io import loadmat

import glob

def jaedong_plot(Y, C, traj_ids, name: str) -> None:
    """Scatter the first 2 (or 3) embedding coords coloured by ``C``.

    Args:
        Y: Embedding coordinates, shape ``(n_samples, n_components)``. If
            ``Y.shape[1] > 2`` the output gets a 3D scatter alongside the
            2D one.
        C: Per-sample colour values (1D array of floats).
        traj_ids: Optional 1D array of trajectory IDs (one per sample).
            When given, samples sharing an ID are connected by a grey line
            on top of the scatter; otherwise just the scatter is drawn.
        name: Output path prefix; the figure is written to ``<name>.png``.

    Returns:
        None.
    """
    if Y.shape[1] > 2: # and False:
        fig = plt.figure(figsize=(9,4))
        ax = fig.add_subplot(121, projection='3d')
        if traj_ids is None:
            ax.scatter(Y[:,0], Y[:,1], Y[:,2], c=C, alpha=0.5, cmap='viridis')
        else:
            for i in np.unique(traj_ids):
                mask = traj_ids == i
                ax.plot3D(Y[mask,0], Y[mask,1], Y[mask,2], alpha=0.5, color='grey')
            ax.scatter(Y[:,0], Y[:,1], Y[:,2], c=C, alpha=0.5, cmap='viridis')
        ax = fig.add_subplot(122) #1, projection='3d')
    else:
        fig = plt.figure()
        fig, ax= plt.subplots()
    if traj_ids is None:
        result = ax.scatter(Y[:,0], Y[:,1], c=C, alpha=0.5) 
    else:
        for i in np.unique(traj_ids):
            mask = traj_ids == i
            ax.plot(Y[mask,0], Y[mask,1], alpha=0.3, c='grey')
        result = ax.scatter(Y[:,0], Y[:,1], c=C, alpha=0.75) 
    fig.colorbar(result, ax=ax)
    plt.savefig(f'{name}.png')


def run_non_linear_reduction(X, labels, hidden_name: str, dirname: str = 'jaedong', traj_ids=None):
    """Run locally-linear embedding on ``X`` and write per-label scatter plots.

    The disabled ``if False`` block below additionally fits t-SNE, MDS, modified
    LLE, Hessian LLE, LTSA, Spectral Embedding, Isomap, and PCA for reference;
    flip it on to regenerate the full reduction sweep.

    Args:
        X: Sample matrix, shape ``(n_samples, n_features)``. Accepts a torch
            Tensor; will be moved to CPU and converted to numpy.
        labels: Single colour array (when ``hidden_name`` is a str) or a list
            of colour arrays (when ``hidden_name`` is a list of names).
        hidden_name: Single name for the colour-coded variable, or a list of
            names paired one-to-one with ``labels``.
        dirname: Output directory; created if missing.
        traj_ids: Optional per-sample trajectory IDs; when given, samples
            sharing an ID are line-connected in the scatter.

    Returns:
        None. Calls ``jaedong_plot`` once per (name, label) pair, writing
        ``<dirname>/<name>_locally_linear.png``.
    """
    if len(X) == 0:
        print("X is empty")
        return
    os.makedirs(dirname, exist_ok=True)
    if type(hidden_name) is str:
        hidden_names = [hidden_name]
        label_list = [labels]
    else:
        hidden_names = hidden_name
        label_list = labels

    if type(X) is torch.Tensor:
        X = X.cpu().numpy()
    n_neighbors = min(100, X.shape[0]-1)
    n_components = 10
    Ylle, squared_error = skmanifold.locally_linear_embedding(X, n_neighbors=n_neighbors, n_components=n_components) # Y \in (Nsample, N_component)

    if False:
        tsne = skmanifold.TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3)
        Ytsne = tsne.fit_transform(X)
        try:
            Ymds = skmanifold.MDS(n_components=2).fit_transform(X)
        except Exception:
            Ymds = None
        try:
            Ymds3d = skmanifold.MDS(n_components=3).fit_transform(X)
        except Exception:
            Ymds3d = None
        Yllem, squared_error = skmanifold.locally_linear_embedding(X, n_neighbors=n_neighbors, n_components=n_components, method='modified') # Y \in (Nsample, N_component)
        Yhessian, squared_error = skmanifold.locally_linear_embedding(X, n_neighbors=n_neighbors, n_components=n_components, method='hessian') # Y \in (Nsample, N_component)
        Yltsa, squared_error = skmanifold.locally_linear_embedding(X, n_neighbors=n_neighbors, n_components=n_components, method='ltsa') # Y \in (Nsample, N_component)

        Yse = skmanifold.SpectralEmbedding(n_components=2).fit_transform(X)
        Yisomap = skmanifold.Isomap(n_components=2).fit_transform(X)
        pca = skdecomposition.PCA(n_components=X.shape[-1]//2) #_hiddens.shape[-1])
        Ypca = pca.fit_transform(X)
        var = pca.explained_variance_
        var /= var.sum()
        var = np.cumsum(var)
        print(var)

    for hidden_name, labels in zip(hidden_names, label_list):
        print(hidden_name)
        jaedong_plot(Ylle, labels, traj_ids, f'{dirname}/{hidden_name}_locally_linear')
        if False:
            jaedong_plot(Ytsne, labels, traj_ids, f'{dirname}/{hidden_name}_tsne')
            if Ymds is not None:
                jaedong_plot(Ymds, labels, traj_ids, f'{dirname}/{hidden_name}_mds')
            if Ymds3d is not None:
                jaedong_plot(Ymds3d, labels, traj_ids, f'{dirname}/{hidden_name}_mds3d')

            jaedong_plot(Yllem, labels, traj_ids, f'{dirname}/{hidden_name}_modified_locally_linear')
            jaedong_plot(Yhessian, labels, traj_ids, f'{dirname}/{hidden_name}_locally_linear_hessian')
            jaedong_plot(Yltsa, labels, traj_ids, f'{dirname}/{hidden_name}_locally_tangent_space')
            jaedong_plot(Yse, labels, traj_ids, f'{dirname}/{hidden_name}_spectral')
            jaedong_plot(Yisomap, labels, traj_ids, f'{dirname}/{hidden_name}_isomap')
            jaedong_plot(Ypca, labels, traj_ids, f'{dirname}/{hidden_name}_pca')
            
            fig = plt.figure()
            plt.plot(np.arange(len(var)), var, marker='o')
            plt.title('scree')
            plt.savefig(f'{dirname}/{hidden_name}_scree.png')

def reverse_time(time, end_points):
    """Convert per-step time stamps into per-trial *time-to-end* stamps.

    Args:
        time: 1D array of per-step time values, concatenated across trials.
        end_points: 1D array of indices into ``time`` marking each trial's
            last step (inclusive).

    Returns:
        np.ndarray: Same shape as ``time``. Each entry ``time[i]`` is
        replaced by ``time[i] - time[end_of_its_trial]``, so the value at
        each trial's end becomes 0 and earlier steps are negative.
    """
    reverse_time = np.zeros_like(time)
    reverse_time[:end_points[0]+1] = time[:end_points[0]+1] - time[end_points[0]]
    for i in range(len(end_points)-1):
        reverse_time[end_points[i]+1:end_points[i+1]+1] = time[end_points[i]+1:end_points[i+1]+1] - time[end_points[i+1]]

    return reverse_time
 
def run_with_mask(mask, X, start_img, targ_img, dist, time, curr_state_img, curr_state_dist, traj_ids, _dirname):
    """Filter all parallel arrays by ``mask`` and dispatch to ``run_non_linear_reduction``.

    Args:
        mask: Boolean array selecting which samples to keep.
        X: Sample feature matrix, shape ``(n_samples, n_features)``.
        start_img / targ_img: Per-sample start / target image indices.
        dist: Per-sample target distance.
        time: Per-sample timestep.
        curr_state_img / curr_state_dist: Per-sample current-image and
            current-distance labels.
        traj_ids: Per-sample trajectory IDs.
        _dirname: Output directory passed straight through to
            ``run_non_linear_reduction``.

    Returns:
        None. Runs LLE once on the masked subset and writes one scatter
        per label (dist, curr_state_img, curr_state_dist, time, start_img,
        targ_img) into ``_dirname``.
    """
    _time = time[mask]
    _start_img = start_img[mask]
    _targ_img = targ_img[mask]
    _dist = dist[mask]
    _curr_state_img = curr_state_img[mask]
    _curr_state_dist = curr_state_dist[mask]
    _traj_ids = traj_ids[mask]
    _X = X[mask]
    print(_X.shape)

    hidden_names = ['dist', 'curr_state_img', 'curr_state_dist', 'time', 'start_img', 'targ_img']
    labels = [_dist, _curr_state_img, _curr_state_dist, _time, _start_img, _targ_img]
    run_non_linear_reduction(_X, labels, hidden_names, _dirname, _traj_ids)


if __name__ == '__main__':
#    flist = glob.glob('model_data/*_data.mat')
    flist = glob.glob('model_data/*alignedto_onset.mat')
    print(flist)
    for f in flist:
        # read mat files.
        if 'ama' not in f or 'HC' not in f:
            continue
        breakpoint()
        mat = loadmat(f)
        print(f)
        dirname = f.replace('.mat', '')
        X = mat['Xx']
        info = mat['labels']
        time = info[0,0][0].reshape(-1)
        start_img = info[0,0][1].reshape(-1)
        targ_img = info[0,0][2].reshape(-1)
        dist = info[0,0][3].reshape(-1)
        curr_state_img = info[0,0][4].reshape(-1)
        curr_state_dist = info[0,0][5].reshape(-1)
        
        end_points = []
        for i in range(len(time)-1):
            if time[i] > time[i+1]:
                end_points.append(i)
        end_points.append(len(time)-1)
        end_points = np.array(end_points)

        traj_ids = np.zeros_like(time)
        for i in range(len(end_points)-1):
            traj_ids[end_points[i]+1:end_points[i+1]+1] = i+1
        offset_time = reverse_time(time, end_points)
        
        # entire data
        if True:
#            mask = np.ones_like(time, dtype=bool)
#            run_with_mask(mask, X, start_img, targ_img, dist, time, curr_state_img, curr_state_dist, traj_ids, dirname)
#            for t in [-0.5, 0, 0.5, 1, 1.5, 2, 2.5]:
#                mask = (time > t) & (time <= t+0.5)
#                _dirname = dirname + '/time/onset_' + str(t) + '_' + str(t+0.5)
#                run_with_mask(mask, X, start_img, targ_img, dist, time, curr_state_img, curr_state_dist, traj_ids, _dirname)
            for t in [-3.5, -3, -2.5, -2, -1.5, -1, -0.5]:
                _dirname = dirname + '/time/offset_' + str(t) + '_' + str(t+0.5)
                mask = (offset_time> t) & (offset_time <= t+0.5)
                run_with_mask(mask, X, start_img, targ_img, dist, offset_time, curr_state_img, curr_state_dist, traj_ids, _dirname)

        if True:
            _dirname = dirname + '/backward'
#            mask = start_img > targ_img
#            run_with_mask(mask, X, start_img, targ_img, dist, time, curr_state_img, curr_state_dist, traj_ids, _dirname)
#            for t in [-0.5, 0, 0.5, 1, 1.5, 2, 2.5]:
#                mask = (time > t) & (time <= t+0.5) & mask
#                _dirname = dirname + '/backward/time/onset_' + str(t) + '_' + str(t+0.5)
#                run_with_mask(mask, X, start_img, targ_img, dist, time, curr_state_img, curr_state_dist, traj_ids, _dirname)
            for t in [-3.5, -3, -2.5, -2, -1.5, -1, -0.5]:
                _dirname = dirname + '/backward/time/offset_' + str(t) + '_' + str(t+0.5)
                mask = (offset_time> t) & (offset_time <= t+0.5)
                run_with_mask(mask, X, start_img, targ_img, dist, offset_time, curr_state_img, curr_state_dist, traj_ids, _dirname)

            _dirname = dirname + '/forward'
            mask = start_img < targ_img
#            run_with_mask(mask, X, start_img, targ_img, dist, time, curr_state_img, curr_state_dist, traj_ids, _dirname)
#            for t in [-0.5, 0, 0.5, 1, 1.5, 2, 2.5]:
#                mask = (time > t) & (time <= t+0.5) & mask
#                _dirname = dirname + '/forward/time/onset_' + str(t) + '_' + str(t+0.5)
#                run_with_mask(mask, X, start_img, targ_img, dist, time, curr_state_img, curr_state_dist, traj_ids, _dirname)
            for t in [-3.5, -3, -2.5, -2, -1.5, -1, -0.5]:
                _dirname = dirname + '/forward/time/offset_' + str(t) + '_' + str(t+0.5)
                mask = (offset_time> t) & (offset_time <= t+0.5)
                run_with_mask(mask, X, start_img, targ_img, dist, offset_time, curr_state_img, curr_state_dist, traj_ids, _dirname)



