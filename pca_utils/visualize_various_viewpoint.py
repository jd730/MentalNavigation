import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def visualize_rotation_grid(
    datasets: list[list[np.ndarray]], 
    titles: list[str] = None, 
    prefix=''
) -> None:
    """
    Saves a PDF where each page is a giant grid for one embedding.
    Rows = Elevation (0:360:10), Columns = Azimuth (0:360:10).
    """
    if not datasets:
        print("No datasets to visualize.")
        return


    pdf_filename = f"{prefix}_rotation_matrix.pdf"
    is_3d = datasets[0][0].shape[1] == 3
    
    # 0, 10, ..., 360 (37 steps)
    angles = np.arange(0, 360, 30)
    n_angles = len(angles)
    
    if titles is None:
        titles = [f"Dataset {i+1}" for i in range(len(datasets))]


    with PdfPages(pdf_filename) as pdf:
        for dataset_idx, (embedded_groups, title) in enumerate(zip(datasets, titles)):
            print(f"Creating rotation grid for: {title}...")
            
            # Create a large figure to hold the 37x37 grid
            # Each subplot is roughly 2x2 inches
            fig, axes = plt.subplots(
                n_angles, n_angles, 
                subplot_kw={'projection': '3d'} if is_3d else {}, 
                figsize=(n_angles * 1.5, n_angles * 1.5),
                facecolor='white'
            )
            
            cmap = plt.get_cmap('Reds')
            colors = cmap(np.linspace(0.3, 1, len(embedded_groups)))


            for r, elev in enumerate(angles):
                for c, azim in enumerate(angles):
                    ax = axes[r, c]
                    
                    # --- STYLE: AXIS ON, GRID OFF ---
                    if is_3d:
                        ax.view_init(elev=elev, azim=azim)
                        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                        ax.grid(False)
                        
                        # Remove tick labels to save space in the giant grid
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_zticklabels([])
                    
                    # Plot the trajectories
                    for i, embedded_data in enumerate(embedded_groups):
                        if embedded_data.ndim < 2: continue
                        color = colors[i]
                        ax.plot(
                            embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2], 
                            c=color, alpha=0.7, lw=0.8
                        )
                        # 2. Start marker (Circle)
                        ax.scatter(
                            embedded_data[0, 0], embedded_data[0, 1], embedded_data[0, 2], 
                            c=[color], s=8, marker='o'
                        )
                        # 3. Target marker (X)
                        ax.scatter(
                            embedded_data[-1, 0], embedded_data[-1, 1], embedded_data[-1, 2], 
                            c=[color], s=12, marker='x'
                        )
                    
                    # Label the outer edges of the grid
                    if r == 0:
                        ax.set_title(f"Azim: {azim}°", fontsize=10)
                    if c == 0:
                        ax.set_ylabel(f"Elev: {elev}°", fontsize=10, labelpad=20)


            fig.suptitle(f"Rotation Matrix: {title}\n(Rows: Elevation | Cols: Azimuth)", fontsize=40, y=1.02)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
    print(f"Saved giant grid PDF to {pdf_filename}")


if __name__ == "__main__":
    flists = sorted(glob.glob('monkey_neural_embeddings_*.pkl'))
    flists = sorted(glob.glob('vHMN2*.pkl'))
    
    for fname in flists:
        try:
            with open(fname, 'rb') as f:
                data = pickle.load(f)
        except:
            continue


        embeddings = []
        titles = []
        for k, v in data.items():
            if not isinstance(v, tuple):
                embeddings.append(v)
                titles.append(k)


        if embeddings:
            visualize_rotation_grid(embeddings, titles, fname.replace('.pkl', ''))

