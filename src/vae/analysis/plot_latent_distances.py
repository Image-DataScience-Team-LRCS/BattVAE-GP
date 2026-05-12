import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


def analyze_latent_distances(
    file_path: Path, save_path: Path, distance_threshold: float = 0.0001
) -> pd.DataFrame:
    """
    Analyze distances between latent points with dynamic dimensionality.

    Args:
        file_path: Path to the latent space CSV file
        distance_threshold: Threshold for considering points as "close"
    """
    # Load data
    df = pd.read_csv(file_path)

    # Identify all z-dimension columns dynamically
    z_cols = [col for col in df.columns if col.startswith("z")]
    print(f"Found {len(z_cols)} latent dimensions: {z_cols}")

    # Sort by cycle and reset index
    df = df[["Cycle"] + z_cols].sort_values(by="Cycle").reset_index(drop=True)

    # Extract latent vectors
    vectors = df[z_cols].values

    # Calculate pairwise distances
    close_pairs = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dist = np.linalg.norm(vectors[i] - vectors[j])
            if dist < distance_threshold:
                close_pairs.append((df["Cycle"].iloc[i], df["Cycle"].iloc[j], dist))

    # Convert to DataFrame
    df_pairs = pd.DataFrame(close_pairs, columns=["Cycle_i", "Cycle_j", "Distance"])

    # Create visualization
    plt.figure(figsize=(12, 10))

    # Create heatmap-style scatter plot
    scatter = plt.scatter(
        df_pairs["Cycle_i"],
        df_pairs["Cycle_j"],
        c=df_pairs["Distance"],
        cmap="viridis",
        s=20,
        alpha=0.6,
    )

    # Add colorbar and labels
    plt.colorbar(scatter, label="Euclidean Distance")
    plt.title(
        f"Cycle Pairs with Latent Distance < {distance_threshold}\n"
        f"(Using {len(z_cols)} latent dimensions)"
    )
    plt.xlabel("Cycle i")
    plt.ylabel("Cycle j")

    # Add grid and improve layout
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Print statistics
    print(f"\nDistance Statistics:")
    print(f"Number of close pairs: {len(df_pairs)}")
    print(f"Min distance: {df_pairs['Distance'].min():.6f}")
    print(f"Max distance: {df_pairs['Distance'].max():.6f}")
    print(f"Mean distance: {df_pairs['Distance'].mean():.6f}")

    return df_pairs


if __name__ == "__main__":
    # Example usage
    latent_space_file = "latent_space.csv"
    pairs = analyze_latent_distances(latent_space_file, distance_threshold=0.0001)
