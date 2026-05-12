from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "vae.yaml"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.logger.logging import get_logger, setup_logging
from src.vae.preprocessing.data_preprocessing import DataProcessor
from src.common.utils.config_schema import FullConfig
from src.common.utils.utils import load_config, load_datasets

logger = get_logger(__name__)


def find_zero_tail_cycles(
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    fill_value: float,
    tol: float = 1e-8,
    value_prefix: str = "v_dis",
) -> pd.DataFrame:
    rows = []

    for idx in range(features.shape[0]):
        q_cap = features[idx, 1, :]
        v_dis = features[idx, 3, :]
        valid = masks[idx, 3, :] > 0.5

        fill_valid = valid & np.isclose(v_dis, fill_value, atol=tol)
        early_fill = fill_valid & (q_cap > tol)
        if not early_fill.any():
            continue

        first_fill_idx = int(np.flatnonzero(early_fill)[0])
        first_valid_idx = np.flatnonzero(valid)
        first_five = [np.nan] * 5
        for j, pos in enumerate(first_valid_idx[:5]):
            first_five[j] = float(v_dis[int(pos)])

        rows.append(
            {
                "sample_index": idx,
                "cycle": int(labels[idx, 1]),
                "q_fill_start": float(q_cap[first_fill_idx]),
                "num_fill_points": int(fill_valid.sum()),
                f"{value_prefix}_1": first_five[0],
                f"{value_prefix}_2": first_five[1],
                f"{value_prefix}_3": first_five[2],
                f"{value_prefix}_4": first_five[3],
                f"{value_prefix}_5": first_five[4],
            }
        )

    return pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)


def plot_zero_tail_cycles(
    features: np.ndarray,
    masks: np.ndarray,
    zero_tail_df: pd.DataFrame,
    dataset_name: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)

    if zero_tail_df.empty:
        ax.text(0.5, 0.5, "No V_dis left-tail fill cycles found.", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    cycles = zero_tail_df["cycle"].to_numpy(dtype=int)
    cyc_min = int(cycles.min())
    cyc_max = int(cycles.max())
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=cyc_min, vmax=cyc_max if cyc_max > cyc_min else cyc_min + 1)

    for row in zero_tail_df.itertuples(index=False):
        sample_idx = int(row.sample_index)
        q_cap = features[sample_idx, 1, :]
        v_dis = features[sample_idx, 3, :]
        valid = masks[sample_idx, 3, :] > 0.5
        color = cmap(norm(int(row.cycle)))
        ax.plot(q_cap[valid], v_dis[valid], color=color, lw=1.2, alpha=0.9)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Cycle number")

    ax.set_title(f"V_dis curves with left-tail fill before q=0: {dataset_name}")
    ax.set_xlabel("q_cap")
    ax.set_ylabel("V_dis [V]")
    ax.grid(True, ls=":", lw=0.6)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot V_dis curves whose processed discharge reaches the left-tail fill value before q=0.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the VAE config file.")
    parser.add_argument("--data", default="data5", help="Dataset key from the VAE config.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/visualizations",
        help="Directory to save outputs.",
    )
    args = parser.parse_args()

    config = FullConfig(**load_config(args.config))
    setup_logging(filename_prefix="vdis_zero_tail", console_output=True)
    v_dis_fill_value = float(config.NORMALIZATION.voltage.get("v_min", 2.5))

    df = load_datasets(config, data=args.data, split="train")
    processor = DataProcessor(config)
    processor.prepare_cycle_data(df)
    features_raw, masks_raw, labels_raw, token_masks, feature_names, label_names, extras = (
        processor.create_capacity_features_and_masks(
            normalize=False,
            norm_cfg=config.NORMALIZATION.model_dump(),
        )
    )
    del token_masks, feature_names, label_names, extras

    features_norm, masks_norm, labels_norm, _, _, _, _ = processor.create_capacity_features_and_masks(
        normalize=True,
        norm_cfg=config.NORMALIZATION.model_dump(),
    )

    zero_tail_df = find_zero_tail_cycles(
        features_raw,
        masks_raw,
        labels_raw,
        fill_value=v_dis_fill_value,
        value_prefix="v_dis",
    )
    fill_value_norm = 0.0
    zero_tail_norm_df = find_zero_tail_cycles(
        features_norm,
        masks_norm,
        labels_norm,
        fill_value=fill_value_norm,
        value_prefix="v_dis_norm",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"vdis_zero_tail_{args.data}.png"
    csv_path = output_dir / f"vdis_zero_tail_{args.data}.csv"
    csv_norm_path = output_dir / f"vdis_zero_tail_{args.data}_normalized.csv"

    plot_zero_tail_cycles(
        features=features_raw,
        masks=masks_raw,
        zero_tail_df=zero_tail_df,
        dataset_name=args.data,
        output_path=plot_path,
    )
    zero_tail_df.to_csv(csv_path, index=False)
    zero_tail_norm_df.to_csv(csv_norm_path, index=False)

    logger.info("Saved zero-tail plot to %s", plot_path)
    logger.info("Saved zero-tail summary to %s", csv_path)
    logger.info("Saved normalized zero-tail summary to %s", csv_norm_path)
    logger.info(
        "Found %s cycles with V_dis reaching fill value %.3f before q=0",
        len(zero_tail_df),
        v_dis_fill_value,
    )
    logger.info(
        "Found %s cycles with normalized V_dis reaching fill value %.3f before q=0",
        len(zero_tail_norm_df),
        fill_value_norm,
    )


if __name__ == "__main__":
    main()
