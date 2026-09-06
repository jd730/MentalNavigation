import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt


def visualize_specific_view(
    datasets: list[list[np.ndarray]], 
    titles: list[str] = None, 
    elev: int = 30, 
    azim: int = 45,
    prefix=''
) -> None:
    """
    Visualizes embeddings in a standard Matplotlib window with a specific angle.
    """
    print(f"Visualizing with Elevation={elev} and Azimuth={azim} for {prefix}")
    if not datasets:
        return


    is_3d = datasets[0][0].shape[1] == 3
    N = len(datasets)
    
    # Grid logic for the UI window
    X = max(1, int(N**0.5 * 0.75))
    Y = (N + X - 1) // X
    
    fig, axes = plt.subplots(
        X, Y, 
        subplot_kw={'projection': '3d'} if is_3d else {}, 
        figsize=(6, 4),
        facecolor='white'
    )
    
    axes = np.atleast_1d(axes).flatten()
    if titles is None:
        titles = [f"Dataset {i+1}" for i in range(N)]


    for idx, (ax, embedded_groups) in enumerate(zip(axes, datasets)):
        cmap = plt.get_cmap('Reds')
        colors = cmap(np.linspace(0.4, 1, len(embedded_groups)))


        if is_3d:
            ax.view_init(elev=elev, azim=azim)
            # Clean Style: Transparent panes, no grid
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.grid(False)


        for i, embedded_data in enumerate(embedded_groups):
            if embedded_data.ndim < 2: continue
            c = colors[i]
            
            if is_3d:
                # Trajectory
                ax.plot(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2], 
                        c=c) #, alpha=0.8, lw=1.5)
                # Start (Circle)
                ax.scatter(embedded_data[0, 0], embedded_data[0, 1], embedded_data[0, 2], c=[c])
                ax.scatter(embedded_data[-1, 0], embedded_data[-1, 1], embedded_data[-1, 2], c=[c], marker='x')
            else:
                ax.plot(embedded_data[:, 0], embedded_data[:, 1], c=c)


#        ax.set_title(titles[idx], fontsize=12)


    # Remove unused axes
    for j in range(len(datasets), len(axes)):
        fig.delaxes(axes[j])


#    plt.suptitle(f"{prefix} | View: Elev={elev}, Azim={azim}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{prefix}_view_elev{elev}_azim{azim}.pdf", dpi=1000)


if __name__ == "__main__":
    # SET YOUR CHOSEN ANGLE HERE


    viewpoint ={
            'ec': [(90, 30)],
            '7a': [(150,120)],
            'distRNN': [(90, 150), (330, 60)],
            'actionRNN': [(150, 270), (90, 300)],
            'logit': [(150, 270), (90, 300)],
            'softmax': [(150, 270), (90, 300)],
            }


    flists = sorted(glob.glob('monkey_neural_embeddings_*.pkl')) + sorted(glob.glob('vHMN2*.pkl'))
    for fname in flists:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        print(data.keys())
        for title, embedding in data.items():
            elev = 0
            azim = 0
            if isinstance(embedding, tuple) or 'isomap' in title:
                continue
            for k, v in viewpoint.items():
                if k in title:
                    for _v in v:
                        print(k, _v)
                        elev, azim = _v
                        visualize_specific_view(
                            [embedding], [title], elev, azim,
                            prefix=k
                        )

