import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
import mat73
from classification import process_data
from filtering import smooth_tensor_joyon
from scipy.ndimage import gaussian_filter1d
import pickle
from run_visualization import visualize_grid
import torch.nn.functional as F
import torch


def split(hidden: np.ndarray, distance: np.ndarray, mask=None) -> dict:
    """
    Group hidden states by absolute distance and truncate time according to distance.


    Parameters
    ----------
    hidden : np.ndarray
        Array of hidden states with shape (num_trials, T, ...).
    distance : np.ndarray
        Per-trial distances (same length as num_trials). Can be float but assumed integer-valued.


    Returns
    -------
    dict[int, np.ndarray]
        Mapping distance d -> hidden states for trials with that distance,
        truncated to the last |d| + 1 time steps along the time axis.
    """
    data = {}
    unique_distances = np.unique(distance)
    for d in unique_distances:
        trial_mask = distance == d
        hidden_for_d = hidden[trial_mask]  # shape: (n_trials_d, T, ...)
#        hidden_for_d = hidden_for_d[:16]


        if mask is not None:
            _mask = mask[trial_mask] #[:16]
            s = _mask.sum(axis=0) # sum over trial
            d_int = d
            hidden_for_d = (hidden_for_d * _mask[:, :, None]).sum(axis=0)
            hidden_for_d = hidden_for_d / s[:, None]  # (T, D)
            st, ed = np.nonzero(s)[0][0], np.nonzero(s)[0][-1]
            hidden_for_d = hidden_for_d[st:ed+1][np.newaxis][np.newaxis] # (T, D)
            # subsample
            hidden_for_d = hidden_for_d
            print(st, ed, d)
        else:
            # Remove padding: keep only final |d|+1 time steps
            d_int = int(d)
            hidden_for_d = hidden_for_d[:, -abs(d_int) - 1:]
        data[d_int] = hidden_for_d #[:,:, ::10] #[:,:,::10]
        


    return data




def update_dict(base_dict: dict, new_dict: dict) -> dict:
    """
    Append new hidden-state arrays into an existing dictionary keyed by distance.


    Parameters
    ----------
    base_dict : dict[int, list[np.ndarray]]
        Accumulator dictionary mapping distance -> list of arrays.
    new_dict : dict[int, np.ndarray]
        Dictionary mapping distance -> array for a new seed/file.


    Returns
    -------
    dict[int, list[np.ndarray]]
        Updated accumulator.
    """
    for key, value in new_dict.items():
        key_int = int(key)
        if key_int not in base_dict:
            base_dict[key_int] = []
        base_dict[key_int].append(value)
    return base_dict




def aggregate(data: dict, option: str = 'mean') -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate hidden states across seeds/files and flatten distance labels.


    Parameters
    ----------
    data : dict[int, np.ndarray]
        Mapping distance -> array of shape (num_seeds, ..., T, D) or similar.
    option : {'mean', 'none'}
        - 'mean': average over all non-time dimensions (axis 0 and 1) to get
          one trajectory per distance.
        - 'none': keep all trajectories, flatten all non-time dims into batch.


    Returns
    -------
    output : np.ndarray
        Aggregated data of shape (N_total, T_or_D), suitable for embedding.
    distance : np.ndarray
        Distance labels of length N_total (one per row in `output`).
    """
    output_chunks = []
    distance_labels = []
    for key, value in data.items():
        if option == 'mean':
            # Average over trajectories / seeds (axes 0,1), keep time/feature axis.
            value_agg = value.mean(axis=(0, 1))
        elif option == 'none':
            # Collapse all non-time dims into batch.
            value_agg = value.reshape(-1, value.shape[-1])
        elif option == 'first':
            value_agg = value.mean(0)[0]
        elif option == 'random':
            n_trials = value.shape[1]
            rand_idx = np.random.randint(n_trials)
            value_agg = value.mean(0)[rand_idx]
        elif option == 'jaed1':
            # use the first seed.
            value_agg = value[5:].mean(0).mean(0)
        else:
            raise ValueError(f"Unsupported option: {option}")
        
        value_agg = value_agg#[-12:]
        output_chunks.append(value_agg)
        distance_labels.append([key] * value_agg.shape[0])


    output = np.concatenate(output_chunks, axis=0)
    distance = np.concatenate(distance_labels, axis=0)
    return output, distance




def nonlinear_analysis(
    data: np.ndarray,
    distance: np.ndarray,
    mode: str = 'tsne',
    n_components: int = 3,
    random_state: int = 42
) -> list[np.ndarray]:
    """
    Project hidden states into a low-dimensional space and group by distance.


    Parameters
    ----------
    data : np.ndarray
        Input data of shape (N_samples, D_features).
    distance : np.ndarray
        Distance labels for each sample (length N_samples).
    mode : {'tsne', 'isomap'}
        Nonlinear embedding method.
    n_components : int
        Number of embedding dimensions (default 3 for 3D plot).
    random_state : int
        Random seed for stochastic methods (t-SNE).


    Returns
    -------
    list[np.ndarray]
        List of embedded arrays, one per unique distance value, in ascending
        order of distance.
    """
    if mode == 'tsne':
        model = TSNE(n_components=n_components, random_state=random_state)
    elif mode == 'isomap':
        model = Isomap(n_components=n_components)
    elif mode == 'pca':
        model = PCA(n_components=n_components)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    all_embedded = model.fit_transform(data)


    # Group embedded points by distance
    embedded_by_distance = []
    unique_distances = np.unique(distance)
    for d in unique_distances:
        mask = distance == d
        embedded_by_distance.append(all_embedded[mask])


    print(f"Number of distance groups embedded: {len(embedded_by_distance)}")
    return embedded_by_distance




def visualize(embedded_groups: list[np.ndarray], title: str = '', ax = None) -> None:
    """
    Visualize embedded trajectories in 3D, with one color per distance group.


    Parameters
    ----------
    embedded_groups : list[np.ndarray]
        List of arrays, each of shape (N_group, 3), representing a group
        of points (same distance).
    title : str
        Title for the figure.
    """
    use_existing_ax = ax is not None
    if use_existing_ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


    cmap = plt.get_cmap('Reds')
    colors = cmap(np.linspace(0, 1, len(embedded_groups)*2))[len(embedded_groups):]


    for i, embedded_data in enumerate(embedded_groups):
        c = colors[i]
        # Plot the first point as a scatter for emphasis
        ax.scatter(embedded_data[0, 0], embedded_data[0, 1], embedded_data[0, 2], c=[c])
        # Plot the full trajectory / cloud
        ax.plot(
            embedded_data[:, 0],
            embedded_data[:, 1],
            embedded_data[:, 2],
            c=c
        )


    ax.set_title(title)
    if not use_existing_ax:
        plt.show()
    return ax


def main() -> None:
    """
    Load .mat files, group hidden states by distance, embed them, and visualize.
    """
#    paths = glob.glob('dist_ratio_double_run_50/random*.mat')
    paths = glob.glob('abs/*dist_norm*.mat')


    paths = glob.glob('dist_ratio_double_run_50/random*S11*.mat')
    paths = glob.glob('reg/*.mat')


#    paths = glob.glob('vHMN_run_50/*nodetach*S11*.mat')
    paths = glob.glob('testtt/*nodetach*25*S11*.mat')
#    paths = glob.glob('CONT/*cont50_*.mat')
#    paths = glob.glob('skip_connection/*skipmult*.mat')
    paths = glob.glob('skip_connection/*vector_double*20_abs*.mat')
    paths = glob.glob('skip_connection/*re_*.mat')
    paths = glob.glob('skip_connection/*100_abs*.mat')
    paths = glob.glob('skip_connection/*kipmult_norm5_abs10*.mat')
    paths = glob.glob('skip_connection/*_norm20_abs10*.mat')
    paths = glob.glob('skip_connection/*kipgrid_norm5_abs10*.mat')
    paths = glob.glob('norm_sweep/*detach_norm20*.mat')
    paths = glob.glob('norm_sweep/*norm15*.mat')
    paths = glob.glob('abs/*nodetach*S22*.mat')
    paths = glob.glob('dist_ratio_double_run_50/random*S11*.mat')
    paths = glob.glob('vHMN2_log100/random*S11*.mat')
#    paths = glob.glob('norm25_skipmult/random*S5*.mat')
    paths = glob.glob('norm10_vHMN/random*S5*.mat')
    paths = glob.glob('norm5_vHMN2/random*S7*.mat')
    paths = glob.glob('norm1_vHMN2/random*S5*.mat')
    paths = glob.glob('norm10_skipmult/random*S5*.mat')
    paths = glob.glob('norm20_vHMN2_rev/random*S10*.mat')
    paths = glob.glob('norm10_vHMN2_rev_abs/random*S0*.mat')
    paths = glob.glob('norm1_vHMN2_rev/random*S29*.mat')
    paths = glob.glob('norm5_vHMN2_rev/random*S5*.mat')
    paths = glob.glob('norm5_vHMN2_rev_abs/random*S30*.mat') # 128
    paths = glob.glob('norm5_vHMN2_rev_abs/random*S23*.mat') # 128
    N = 220
    N = 200
    N = 256
    N = 64
    print(N)
    print(paths)


    mode = 'pca'
    mode = 'isomap'
    mode = 'tsne'


#    option = 'random'
    option = 'first'
    option = 'jaed1'
    option = 'mean'


    hiddens = {}
    base_hiddens = {}
    logits = {}


    a7 = '7a_amadeus06242019_a_neur_tensor_joyon.mat'
    ec = 'ec_amadeus08292019_a_neur_tensor_joyon.mat'
    embeddings = {}
    for name, path in zip(['7a', 'ec'], [a7, ec]):
        data = mat73.loadmat(f'monkey_raw_data/{path}')
        binwidth = data['joyon']['binwidth']
        conditions, neural_data, mask = process_data(data) # dictionary, (Neuron, Time, Trial)
        # gaussian smoothing neural data.
        edges = conditions['time'] # (T, Trials)
        sigma = 0.2 / binwidth #/ 10#* 10 #* 10 #/ 10
        sigma = 400
        print(sigma)
        radius = 800
        neural_data /= 1000


        neural_data = gaussian_filter1d(neural_data, sigma=sigma, axis=1, radius=radius) #truncate=2*int(200/sigma))


        distance = conditions['ta']
        num_trial = neural_data.shape[2]
        hidden_dict = split(neural_data.transpose(2,1,0), distance, mask.T)
        # drop all < 0 distance.
        for direction in ['L', 'R', 'A']:
            if 'R' in direction:
                _hidden_dict = {k:v for k,v in hidden_dict.items() if k >=0}
            elif 'L' in direction:
                _hidden_dict = {k:v for k,v in hidden_dict.items() if k <=0}
            else:
                _hidden_dict = {k:v for k,v in hidden_dict.items()}
            hidden, distance = aggregate(_hidden_dict, option)
        
            for mode in ['pca', 'isomap']:
                hidden_embedding = nonlinear_analysis(hidden, distance, mode)
                embeddings[f'{mode}_{name}'] = hidden_embedding
        embeddings[name] = (hidden, distance)
        if hidden.shape[0] != distance.shape[0]:
            breakpoint()


    with open(f'monkey_neural_embeddings_Std{int(sigma)}Rad{radius}.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
        


if __name__ == '__main__':
    main()

