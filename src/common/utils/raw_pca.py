import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot_cycle_pca(csv_path):
    # Read CSV
    df = pd.read_csv(csv_path)

    # Check required columns
    assert {"Cycle", "Voltage", "Current"}.issubset(
        df.columns
    ), "Missing required columns"

    df = df.sort_values(by=["Cycle"])
    grouped = df.groupby("Cycle")

    Voltage_vectors = []
    Current_vectors = []
    cycle_ids = []

    max_len = 1000  # fixed time-series length for consistency

    for i, (cycle, group) in enumerate(grouped):

        Voltage = group["Voltage"].values
        Current = group["Current"].values

        # Truncate or pad Voltage
        if len(Voltage) > max_len:
            Voltage = Voltage[:max_len]
            Current = Current[:max_len]
        else:
            pad_len = max_len - len(Voltage)
            Voltage = np.pad(Voltage, (0, pad_len), mode="constant")
            Current = np.pad(Current, (0, pad_len), mode="constant")

        Voltage_vectors.append(Voltage)
        Current_vectors.append(Current)
        cycle_ids.append(cycle)

    Voltage_matrix = np.array(Voltage_vectors)  # (N, T)
    Current_matrix = np.array(Current_vectors)  # (N, T)

    pca_Voltage = PCA(n_components=2).fit_transform(Voltage_matrix)
    pca_Current = PCA(n_components=2).fit_transform(Current_matrix)

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    scatter1 = axs[0].scatter(
        pca_Voltage[:, 0], pca_Voltage[:, 1], c=cycle_ids, cmap="viridis", s=30
    )
    axs[0].set_title("PCA of Voltage per Cycle")
    axs[0].set_xlabel("PC1")
    axs[0].set_ylabel("PC2")
    axs[0].grid(True)

    scatter2 = axs[1].scatter(
        pca_Current[:, 0], pca_Current[:, 1], c=cycle_ids, cmap="plasma", s=30
    )
    axs[1].set_title("PCA of Current per Cycle")
    axs[1].set_xlabel("PC1")
    axs[1].set_ylabel("PC2")
    axs[1].grid(True)

    # Add colorbar
    cbar = fig.colorbar(scatter1, ax=axs.ravel().tolist(), shrink=0.95)
    cbar.set_label("Cycle Number")

    plt.tight_layout()
    plt.show()


# Example usage:
file = sys.argv[1]
plot_cycle_pca(file)
