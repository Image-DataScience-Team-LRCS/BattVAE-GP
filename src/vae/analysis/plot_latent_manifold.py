import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Tuple
from pathlib import Path
import numpy as np

# from umap import UMAP
import matplotlib

matplotlib.use("Agg")
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"


def visualize_latent_space(
    latent_file: Path, vis_range: Optional[Tuple[int, int]] = None
) -> None:
    # Load latent space data
    data = torch.load(latent_file, weights_only=True)

    latent_space = data["latent_space_mu"].numpy()
    cycle_numbers = np.array(data["cycle_numbers"])

    # Sort cycle numbers and latent space
    sort_indices = np.argsort(cycle_numbers)
    cycle_numbers = cycle_numbers[sort_indices]
    latent_space = latent_space[sort_indices]  # Fixed indexing here

    if vis_range is not None:
        cycle_numbers = cycle_numbers[vis_range[0] : vis_range[1]]
        latent_space = latent_space[vis_range[0] : vis_range[1]]

        # Print
        for i, j in zip(cycle_numbers, latent_space):
            print(i, j)

    # 0. Direct Visualization
    if latent_space.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(
            latent_space[:, 0],
            latent_space[:, 1],
            c=cycle_numbers,
            cmap="viridis",
            alpha=0.7,
        )
        plt.title("Direct Visualization of 2D Latent Space with Cycle Numbers")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.colorbar(label="Cycle Numbers")
        plt.savefig("artifacts/visualizations/DirectLatentSpace.png", dpi=300)
        plt.close()

    elif latent_space.shape[1] == 3:
        fig = px.scatter_3d(
            x=latent_space[:, 0],
            y=latent_space[:, 1],
            z=latent_space[:, 2],
            color=cycle_numbers,
            labels={"x": "Dim 1", "y": "Dim 2", "z": "Dim 3", "color": "Cycle Numbers"},
            title="Direct Visualisation 3D Latent space with cycle numbers",
            opacity=0.8,
        )

        fig.update_traces(
            marker=dict(
                size=4,
                # line=dict(width=0.1, color='White'),
                colorscale="Rainbow",
                colorbar=dict(
                    title="Cycle Number",
                    len=1.5,
                ),
            )
        )
        fig.write_html("artifacts/visualizations/3D_Plotly.html")
        #fig.show()

    # 1. t-SNE Visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(latent_space)
    plt.figure(figsize=(10, 8))
    plt.scatter(
        tsne_result[:, 0],
        tsne_result[:, 1],
        c=cycle_numbers,
        cmap="viridis",
        s=30,
        alpha=0.7,
    )
    plt.title("t-SNE Visualization of Latent Space")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.colorbar(label="Cycle Numbers")
    plt.savefig("artifacts/visualizations/TSNE.png", dpi=300)
    plt.close()

    # 2. PCA Visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(latent_space)
    plt.figure(figsize=(10, 8))
    plt.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        c=cycle_numbers,
        cmap="viridis",
        s=30,
        alpha=0.7,
    )
    plt.title("PCA Visualization of Latent Space")
    plt.xlabel("PCA Dim 1")
    plt.ylabel("PCA Dim 2")
    plt.colorbar(label="Cycle Numbers")
    plt.savefig("artifacts/visualizations/PCA.png", dpi=300)
    plt.close()

    # 2. PCA Statistics
    pca = PCA()
    pca_stat = pca.fit_transform(latent_space)
    plt.figure(figsize=(10, 8))
    plt.plot(
        range(1, len(pca.explained_variance_ratio_) + 1),
        np.cumsum(pca.explained_variance_ratio_),
    )
    plt.title("Choix du nombre de dimensions")
    plt.xlabel("Nombre de dimensions")
    plt.ylabel("Variance expliquée cumulée")
    # plt.colorbar(label="Cycle Numbers")
    plt.savefig("artifacts/visualizations/PCA_Dimensions.png", dpi=300)
    plt.close()

    # # 3. UMAP Visualization
    # umap = UMAP(n_components=2, random_state=42)
    # umap_result = umap.fit_transform(latent_space)
    # plt.figure(figsize=(10, 8))
    # plt.scatter(
    #     umap_result[:, 0],
    #     umap_result[:, 1],
    #     c=cycle_numbers,
    #     cmap="viridis",
    #     s=30,
    #     alpha=0.7,
    # )
    # plt.title("UMAP Visualization of Latent Space")
    # plt.xlabel("UMAP Dim 1")
    # plt.ylabel("UMAP Dim 2")
    # plt.colorbar(label="Cycle Numbers")
    # plt.savefig("artifacts/visualizations/UMAP.png", dpi=300)
    # plt.close()

    # 4. Truncated SVD Visualization
    svd = TruncatedSVD(n_components=2, random_state=42)
    svd_result = svd.fit_transform(latent_space)
    plt.figure(figsize=(10, 8))
    plt.scatter(
        svd_result[:, 0],
        svd_result[:, 1],
        c=cycle_numbers,
        cmap="viridis",
        s=30,
        alpha=0.7,
    )
    plt.title("Truncated SVD Visualization of Latent Space")
    plt.xlabel("SVD Dim 1")
    plt.ylabel("SVD Dim 2")
    plt.colorbar(label="Cycle Numbers")
    plt.savefig("artifacts/visualizations/SVD.png", dpi=300)
    plt.close()

    # 5. K-Means Clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_labels = kmeans.fit_predict(latent_space)
    pca_result = PCA(n_components=2).fit_transform(latent_space)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        hue=kmeans_labels,
        palette="viridis",
        s=30,
        alpha=0.7,
    )
    plt.title("K-Means Clustering of Latent Space (PCA for 2D)")
    plt.xlabel("PCA Dim 1")
    plt.ylabel("PCA Dim 2")
    plt.legend(title="Cluster")
    plt.savefig("artifacts/visualizations/KMeans.png", dpi=300)
    plt.close()

    # # 6. Heatmap Visualization (for correlation matrix of latent space features)
    # corr_matrix = np.abs(np.corrcoef(latent_space, rowvar=False))
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, cbar=True, vmax=1, vmin=0)
    # plt.title("Heatmap of Latent Space Correlation Matrix")
    # plt.savefig('artifacts/visualizations/Heatmap.png', dpi=300)
    # plt.close()

    # 6. 3D PCA with Plotly
    if latent_space.shape[1] > 3:
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(latent_space)

        fig = px.scatter_3d(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            z=pca_result[:, 2],
            color=cycle_numbers,
            labels={
                "x": "PCA Dim 1",
                "y": "PCA Dim 2",
                "z": "PCA Dim 3",
                "color": "Cycle Numbers",
            },
            title="Visualisation 3D du latent space avec PCA (Plotly)",
            opacity=0.8,
        )

        fig.update_traces(
            marker=dict(
                size=4,
                line=dict(width=0.1, color="DarkSlateGrey"),
                colorscale="Viridis",
                colorbar=dict(
                    title="Cycle Number",
                    len=0.99,
                ),
            )
        )
        fig.write_html("artifacts/visualizations/PCA_3D_Plotly_PCA.html")
        fig.show()


def latent_space_correlation(
    latent_file: Path, vis_range: Optional[Tuple[int, int]] = None
) -> None:
    """
    Visualize the correlation matrix of the latent space dimensions.

    Args:
        latent_file (str): Path to the latent space file.
        vis_range (tuple, optional): Range of indices to visualize. Defaults to None.
    """
    data = torch.load(latent_file, weights_only=True)
    latent_space = data["latent_space_mu"].numpy()
    cycle_numbers = np.array(data["cycle_numbers"])

    # Sort by cycle number
    sort_indices = np.argsort(cycle_numbers)
    cycle_numbers = cycle_numbers[sort_indices]
    latent_space = latent_space[sort_indices]

    if vis_range is not None:
        cycle_numbers = cycle_numbers[vis_range[0] : vis_range[1]]
        latent_space = latent_space[vis_range[0] : vis_range[1]]


def animate_latent_space(latent_file: Path, save_path: Path) -> None:
    """
    Animate the evolution of the latent space over cycles and save it as a GIF.

    Args:
        latent_file (str): Path to the latent space file.
    """
    try:
        # Load latent space data
        data = torch.load(latent_file, weights_only=True)
        latent_space = data["latent_space_mu"].numpy()
        cycle_numbers = np.array(data["cycle_numbers"])
    except KeyError as e:
        raise KeyError(f"Missing key in latent space file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading latent space data: {e}")

    # Ensure latent space is 2D
    # Apply PCA if dimension > 2
    if latent_space.shape[1] > 2:
        print(f"Reducing latent space from {latent_space.shape[1]}D to 2D using PCA")
        pca = PCA(n_components=2)
        latent_space = pca.fit_transform(latent_space)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2%}")

    # Sort the latent space and cycle numbers by cycle_numbers to get a smooth progression
    sorted_indices = np.argsort(cycle_numbers)
    latent_space_sorted = latent_space[sorted_indices]
    cycle_numbers_sorted = cycle_numbers[sorted_indices]

    # Normalize cycle numbers to range [0, 1]
    if cycle_numbers.max() == cycle_numbers.min():
        raise ValueError("Cycle numbers must vary to create an animation.")
    normalized_cycles = (cycle_numbers_sorted - cycle_numbers_sorted.min()) / (
        cycle_numbers_sorted.max() - cycle_numbers_sorted.min()
    )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter([], [], c=[], cmap="viridis", s=30, alpha=0.7)

    # Set axis limits dynamically
    ax.set_xlim(
        latent_space_sorted[:, 0].min() - 0.1, latent_space_sorted[:, 0].max() + 0.1
    )
    ax.set_ylim(
        latent_space_sorted[:, 1].min() - 0.1, latent_space_sorted[:, 1].max() + 0.1
    )

    # Add labels and title
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    ax.set_title("Evolution of Latent Space Over Cycles")
    colorbar = plt.colorbar(sc, ax=ax)

    # Set colorbar to display original cycle numbers
    colorbar.set_label("Cycle Numbers")
    colorbar.set_ticks(np.linspace(0, 1, num=6))  # Adjust tick positions
    colorbar.set_ticklabels(
        np.linspace(
            cycle_numbers_sorted.min(), cycle_numbers_sorted.max(), num=6, dtype=int
        )
    )

    # Initialize function for animation
    def init() -> tuple:
        sc.set_offsets(np.empty((0, 2)))
        sc.set_array([])
        return (sc,)

    # Update function for animation
    def update(frame: int) -> tuple:
        current_data = latent_space_sorted[: frame + 1]
        current_cycles = normalized_cycles[: frame + 1]
        sc.set_offsets(current_data)
        sc.set_array(current_cycles)
        return (sc,)

    # Create the animation
    ani = FuncAnimation(
        fig,
        update,
        frames=len(cycle_numbers_sorted),
        init_func=init,
        blit=True,
        interval=100,
    )

    try:
        ani.save(save_path, writer="ffmpeg", fps=10)
        print(f"Animation saved successfully at {save_path}")
    except Exception as e:
        raise IOError(f"Failed to save animation: {e}")


def plot_latent_space_distributions(
    latent_file: Path, vis_range: Optional[Tuple[int, int]] = None
) -> None:
    """
    Plot the distribution of each latent space dimension in one graph.

    Args:
        latent_space (numpy.ndarray): The latent space array of shape (n_samples, n_dimensions).
    """

    data = torch.load(latent_file, weights_only=True)
    latent_space = data["latent_space_mu"].numpy()
    latent_space_logvar = data["latent_space_logvar"].numpy()
    cycle_numbers = np.array(data["cycle_numbers"])

    # Sort by cycle number
    sort_indices = np.argsort(cycle_numbers)
    cycle_numbers = cycle_numbers[sort_indices]
    latent_space = latent_space[sort_indices]
    latent_space_logvar = latent_space_logvar[sort_indices]

    # Apply optional visualization range
    if vis_range is not None:
        cycle_numbers = cycle_numbers[vis_range[0] : vis_range[1]]
        latent_space = latent_space[vis_range[0] : vis_range[1]]
        latent_space_logvar = latent_space_logvar[vis_range[0] : vis_range[1]]

    n_dimensions = latent_space.shape[1]

    # Create subplots: 2 columns, n_dimensions rows
    fig, axes = plt.subplots(
        nrows=n_dimensions,
        ncols=2,
        figsize=(12, 3 * n_dimensions),
        sharex="col",  # Share x-axis only within each column
    )

    # Handle edge case where n_dimensions == 1 (axes won't be 2D)
    if n_dimensions == 1:
        axes = np.array([axes])  # Ensure 2D array shape: (1, 2)

    for i in range(n_dimensions):
        # Left column: latent space
        sns.histplot(
            latent_space[:, i],
            bins=100,
            kde=True,
            ax=axes[i, 0],
            stat="density",
            color="blue",
            label="Data",
        )
        sns.kdeplot(
            latent_space[:, i], ax=axes[i, 0], color="red", label="Gaussian Fit"
        )
        axes[i, 0].set_title(f"Latent mu dim {i+1}")
        axes[i, 0].set_ylabel("Density")
        axes[i, 0].legend()

        # Right column: log-variance
        sns.histplot(
            latent_space_logvar[:, i],
            bins=100,
            kde=True,
            ax=axes[i, 1],
            stat="density",
            color="blue",
            label="Data",
        )
        sns.kdeplot(
            latent_space_logvar[:, i], ax=axes[i, 1], color="red", label="Gaussian Fit"
        )
        axes[i, 1].set_title(f"Logvar dim {i+1}")
        axes[i, 1].set_ylabel("Density")
        axes[i, 1].legend()

    # Set x-labels only on the bottom row
    axes[-1, 0].set_xlabel("Value")
    axes[-1, 1].set_xlabel("Value")

    plt.tight_layout()
    plt.savefig("artifacts/visualizations/Latent_vs_Logvar_Distributions.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # Visualize
    # visualize_latent_space("../artifacts/latent_space/latent_space.pt")
    output_path = "../artifacts/visualizations/LatentSpaceEvolution.gif"
    animate_latent_space(
        "../artifacts/latent_space/latent_space.pt", save_path=output_path
    )
    # plot_latent_space_distributions("../artifacts/latent_space/latent_space.pt")
