from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the first sustained V_ch normalized-threshold crossing index "
            "for each interpolation cycle and plot it versus cycle number."
        )
    )
    parser.add_argument(
        "--input-csv",
        default="data/processed/interpolation/gp_interpolation_070/gp_interpolation_reconstructions.csv",
        help="Interpolation reconstruction CSV containing V_ch_reconstruction_norm.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/interpolation/gp_interpolation_070",
        help="Directory where the summary CSV and plot will be written.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Normalized V_ch threshold used to define the crossing point.",
    )
    parser.add_argument(
        "--min-consecutive",
        type=int,
        default=3,
        help="Minimum number of consecutive samples above threshold.",
    )
    return parser.parse_args()


def _first_sustained_crossing_index(
    values: np.ndarray,
    threshold: float,
    min_consecutive: int,
) -> int | None:
    above = values >= threshold
    run_length = 0
    for idx, is_above in enumerate(above):
        if is_above:
            run_length += 1
            if run_length >= min_consecutive:
                return idx - min_consecutive + 1
        else:
            run_length = 0
    return None


def build_summary(
    reconstruction_df: pd.DataFrame,
    threshold: float,
    min_consecutive: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = reconstruction_df.sort_values(["Cycle", "q_index"]).groupby("Cycle", sort=True)

    for cycle, cycle_df in grouped:
        vch = cycle_df["V_ch_reconstruction_norm"].to_numpy(dtype=float)
        q_cap = cycle_df["q_cap"].to_numpy(dtype=float)
        q_index = cycle_df["q_index"].to_numpy(dtype=int)
        charging_rate = float(cycle_df["charging_rate"].iloc[0])

        crossing_pos = _first_sustained_crossing_index(
            values=vch,
            threshold=threshold,
            min_consecutive=min_consecutive,
        )

        row = {
            "Cycle": int(cycle),
            "charging_rate": charging_rate,
            "threshold": threshold,
            "min_consecutive": int(min_consecutive),
            "crossing_found": crossing_pos is not None,
            "crossing_q_index": np.nan,
            "crossing_q_cap": np.nan,
            "crossing_vch_norm": np.nan,
        }
        if crossing_pos is not None:
            row["crossing_q_index"] = int(q_index[crossing_pos])
            row["crossing_q_cap"] = float(q_cap[crossing_pos])
            row["crossing_vch_norm"] = float(vch[crossing_pos])

        rows.append(row)

    return pd.DataFrame(rows)


def plot_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=300)

    valid = summary_df[summary_df["crossing_found"]].copy()
    if not valid.empty:
        ax.plot(
            valid["Cycle"],
            valid["crossing_q_cap"],
            color="#1f4e79",
            lw=1.6,
            marker="o",
            markersize=2.0,
            markerfacecolor="#5ba3d0",
            markeredgewidth=0.0,
        )

    ax.set_xlabel("Cycle number")
    ax.set_ylabel("q_cap at first threshold crossing")
    ax.set_title("First sustained V_ch threshold crossing")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reconstruction_df = pd.read_csv(
        input_csv,
        usecols=[
            "Cycle",
            "charging_rate",
            "q_index",
            "q_cap",
            "V_ch_reconstruction_norm",
        ],
    )

    summary_df = build_summary(
        reconstruction_df=reconstruction_df,
        threshold=float(args.threshold),
        min_consecutive=int(args.min_consecutive),
    )

    csv_path = output_dir / "vch_threshold_index_by_cycle.csv"
    plot_path = output_dir / "vch_threshold_index_by_cycle.png"

    summary_df.to_csv(csv_path, index=False)
    plot_summary(summary_df, plot_path)

    print(csv_path)
    print(plot_path)


if __name__ == "__main__":
    main()
