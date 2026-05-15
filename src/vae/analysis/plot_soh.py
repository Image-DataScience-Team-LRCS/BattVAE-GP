import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from typing import Optional
import argparse


def plot_soh_evolution(
    predictions_path: Path, save_path: Optional[Path] = None
) -> None:
    """
    Create a beautiful plot showing SOH evolution over cycles with metrics.

    Args:
        predictions_path: Path to the SOH predictions CSV file
        save_path: Optional path to save the plot
    """
    plt.style.use("default")
    sns.set_context("paper", font_scale=1.1)

    # Read predictions
    df = pd.read_csv(predictions_path)
    df = df.dropna(subset=["soh_computed", "soh_predicted"]).copy()
    if df.empty:
        raise ValueError(
            f"No rows with both soh_computed and soh_predicted are available in {predictions_path}."
        )

    r2 = r2_score(df["soh_computed"], df["soh_predicted"])
    mse = mean_squared_error(df["soh_computed"], df["soh_predicted"])
    mae = mean_absolute_error(df["soh_computed"], df["soh_predicted"])
    rmse = np.sqrt(mse)

    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Plot data
    ax.plot(
        df["cycle_numbers"],
        df["soh_computed"],
        label="True SOH",
        linewidth=1.8,
        color="#1f1f1f",
        alpha=0.95,
    )
    uncertainty_cols = {"soh_predicted_lower_90", "soh_predicted_upper_90"}
    if uncertainty_cols.issubset(df.columns):
        uncertainty_df = df.dropna(subset=list(uncertainty_cols))
        if not uncertainty_df.empty:
            ax.fill_between(
                uncertainty_df["cycle_numbers"].to_numpy(dtype=float),
                uncertainty_df["soh_predicted_lower_90"].to_numpy(dtype=float),
                uncertainty_df["soh_predicted_upper_90"].to_numpy(dtype=float),
                label="Predicted SOH 90% interval",
                color="#c44e52",
                alpha=0.18,
                linewidth=0.0,
            )
    ax.plot(
        df["cycle_numbers"],
        df["soh_predicted"],
        label="Predicted SOH",
        linewidth=1.8,
        color="#c44e52",
        alpha=0.95,
    )

    ax.set_xlabel("Cycle Number", fontsize=22)
    ax.set_ylabel("State of Health (SOH)", fontsize=22)
    ax.set_xlim(0, 5000)
    ax.set_ylim(0.92, 1.0)
    ax.grid(False)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(axis="both", labelsize=18, direction="out", length=4, width=0.8)

    ax.legend(loc="lower left", frameon=False, fontsize=18)

    plt.tight_layout()

    # Save or show
    if save_path:
        metrics_path = save_path.with_name(f"{save_path.stem}_metrics.csv")
        pd.DataFrame(
            [
                {
                    "r2_score": r2,
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                }
            ]
        ).to_csv(metrics_path, index=False)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
        print(f"Metrics saved to: {metrics_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_path", type=Path, help="Path to the SOH CSV file.")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional output image path. Defaults to the CSV path with .png suffix.",
    )
    args = parser.parse_args()

    save_path = args.save_path or args.predictions_path.with_suffix(".png")
    plot_soh_evolution(args.predictions_path, save_path)
