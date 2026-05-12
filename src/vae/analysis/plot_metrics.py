import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import ast


def plot_all_metrics(metrics_path):
    """
    Plot all training and validation metrics on one graph.

    Parameters:
    - metrics_path (str): Path to the CSV file containing metrics data.

    The CSV file should have the following columns:
    - epoch: The epoch number.
    - val_loss: Validation loss for each epoch.
    - kl_loss: KL divergence loss for each epoch.
    - reconstruction_loss: Reconstruction loss for each epoch.
    - train_loss: Training loss for each epoch.
    """
    # Load metrics from the CSV file
    data = pd.read_csv(metrics_path)

    # Extract data
    epochs = data["epoch"]
    val_loss = data["val_loss"]
    kl_loss = data["kl_loss"]
    reconstruction_loss = data["reconstruction_loss"]
    train_loss = data["train_loss"]
    soh_loss = data["soh_loss"]
    val_soh_loss = data["val_soh_loss"]

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, val_loss, label="Validation Loss", marker="o")
    plt.plot(epochs, kl_loss, label="KL Loss", marker="o")
    plt.plot(epochs, reconstruction_loss, label="Reconstruction Loss", marker="o")
    plt.plot(epochs, soh_loss, label="SOH Loss", marker="o")
    plt.plot(epochs, val_soh_loss, label="Validation SOH Loss", marker="o")

    # Add labels and title
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Metrics")
    plt.legend()
    plt.grid(True)

    plt.yscale("log")

    # Save and show plot
    plt.savefig("artifacts/visualizations/metrics_plot.png", dpi=300)


def plot_kl_per_dimension(metrics_path: Path) -> None:
    """
    Plot KL divergence per dimension over epochs.

    Parameters:
    ----------
    metrics_path : str
        Path to the CSV file containing metrics data with kl_per_dimension column
    """
    # Create visualization directory if it doesn't exist

    # Load and process data
    data = pd.read_csv(metrics_path)
    epochs = data["epoch"]

    # Convert string representations of lists to numpy arrays
    kl_per_dim = data["kl_per_dimension"].apply(ast.literal_eval)
    kl_per_dim_array = np.array(kl_per_dim.tolist())

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each dimension
    for dim in range(kl_per_dim_array.shape[1]):
        plt.plot(
            epochs,
            kl_per_dim_array[:, dim],
            label=f"Dimension {dim+1}",
            marker="o",
            markersize=3,
        )

    # Add labels and title
    plt.xlabel("Epochs")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence per Latent Dimension")
    plt.legend()
    plt.grid(True)

    # # Set y-axis to log scale if values vary significantly
    # if np.max(kl_per_dim_array) / np.min(kl_per_dim_array[kl_per_dim_array > 0]) > 100:
    #     plt.yscale('log')

    # Add horizontal line at y=0 to show collapse threshold
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.3, label="Collapse Threshold")

    # Save plot
    plt.savefig(
        "artifacts/visualizations/kl_per_dimension.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":
    plot_all_metrics("metrics.csv")
