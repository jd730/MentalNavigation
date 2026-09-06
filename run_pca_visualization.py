import glob
#from  nonlinear import visualize
import pickle


from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


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




def visualize_grid(datasets: list[list[np.ndarray]], titles: list[str] = None, prefix='') -> None:
    """
    Visualize 4 sets of embedded trajectories in a 2x2 3D grid.


    Parameters
    ----------
    datasets : list[list[np.ndarray]]
        A list containing 4 elements. Each element is a list of arrays 
        (the 'embedded_groups' from the original function).
    titles : list[str], optional
        A list of 4 titles, one for each subplot.
    """
    # Create a 2x2 grid with 3D projection
    
    N = len(datasets)
    if len(datasets) == 0:
        print("No datasets to visualize.")
        return
    
    X = int(N**0.5 * 0.75)
#    X = 4
    Y = (N+1) // X
    
    if type(datasets) is not list:
        datasets = list(datasets)
    if datasets[0][0].shape[1] == 2:
        fig, axes = plt.subplots(X, Y, figsize=(4*Y, 3*X))
    else:
        fig, axes = plt.subplots(X, Y, subplot_kw={'projection': '3d'}, figsize=(4*Y, 3*X))
    axes = axes.flatten()  # Flatten to 1D array for easy iteration


    # specific logic to handle cases where fewer than 4 datasets are passed
    if titles is None:
        titles = [f"Figure {i+1}" for i in range(len(datasets))]


    # Zip allows us to iterate over the axes, data, and titles simultaneously
    for ax, embedded_groups, title in zip(axes, datasets, titles):
        if True :        
            # Original Color Logic
            cmap = plt.get_cmap('Reds')
            n_groups = len(embedded_groups)
            # Ensure distinct colors even for small groups
            colors = cmap(np.linspace(0, 1, n_groups * 2))[n_groups:]


            for i, embedded_data in enumerate(embedded_groups):
                c = colors[i]
                # Plot the first point as a scatter for emphasis
                if len(embedded_data.shape) < 2:
                    continue
                if embedded_data.shape[1] == 3:
                    ax.scatter(embedded_data[0, 0], embedded_data[0, 1], embedded_data[0, 2], c=[c])
                    ax.scatter(embedded_data[-1, 0], embedded_data[-1, 1], embedded_data[-1, 2], c=[c], marker='x')
                    if len(embedded_data) < 100: # model
                        for i, (x, y, z) in enumerate(embedded_data[::12]):
                            ax.scatter(x, y, z, c=[c], marker='.')
        #                    ax.scatter(embedded_data[-1, 0], embedded_data[-1, 1], embedded_data[-1, 2], c=[c], marker='x')


                    # Plot the full trajectory / cloud
                    ax.plot(
                        embedded_data[:, 0],
                        embedded_data[:, 1],
                        embedded_data[:, 2],
                        c=c
                    )
                    # end
                else:
                    ax.scatter(embedded_data[0, 0], embedded_data[0, 1], c=[c])
                    ax.scatter(embedded_data[-1, 0], embedded_data[-1, 1], c=[c], marker='x')
                    if len(embedded_data) < 100:
                        for i, (x, y) in enumerate(embedded_data[::12]):
                            ax.scatter(x, y, c=[c], marker='.')
                    ax.plot(
                        embedded_data[:, 0],
                        embedded_data[:, 1],
                        c=c
                    )


            
            ax.set_title(title)


    plt.tight_layout()
    plt.suptitle(prefix)
    plt.show()


if __name__ == "__main__":
    
    flists = sorted(glob.glob('monkey_neural_embeddings_*.pkl'))
    for fname in flists:
        print(fname)
        data = pickle.load(open(fname, 'rb'))
        embeddings = []
        titles = []
        for k, v in data.items():
            print(type(v))
            if type(v) is not tuple:
#                pass
                embeddings.append(v)
                titles.append(k)
            elif False:
                mode = 'pca'
                hidden_embedding = nonlinear_analysis(v[0], v[1], mode, n_components=2)
                embeddings.append(hidden_embedding)
                titles.append(k)


        visualize_grid(embeddings, titles=titles, prefix=fname.replace('.pkl',''))
#    for k, v in data.items():
#        visualize(v, k)
    breakpoint()





