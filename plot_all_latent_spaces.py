from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
CONFIG_PATH = ROOT / "configs" / "datasets.yaml"
OUTPUT_PATH = ARTIFACTS_DIR / "latent_spaces_overlay_with_gp_interpolation.png"

GP_TEXT_STYLE = {
    "fontsize": 12,
    "weight": "bold",
    "color": "#3f2305",
    "bbox": {
        "boxstyle": "round,pad=0.22",
        "fc": "#fff1a8",
        "ec": "#cc8b00",
        "alpha": 0.98,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gp-data",
        default=None,
        nargs="*",
        help="GP interpolation dataset ids to plot. Omit to skip GP data, use `all` for every GP_INTERPOLATION_DATASETS entry, or pass specific dataset ids.",
    )
    return parser.parse_args()


def publication_style() -> Dict[str, object]:
    return {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }


def load_rate_labels(config_path: Path) -> Dict[str, str]:
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    labels: Dict[str, str] = {}
    vae_datasets = dict(config.get("VAE_DATASETS", {}) or {})
    gp_interpolation = dict(config.get("GP_INTERPOLATION_DATASETS", {}) or {})

    for name, dataset_cfg in vae_datasets.items():
        if name == "gp_interpolation":
            continue
        dataset_id = dataset_cfg.get("id")
        if dataset_id is None:
            charging_rate = dataset_cfg.get("charging_rate")
            if charging_rate is not None:
                dataset_id = f"{float(charging_rate):.2f}C"
        if dataset_id is not None:
            labels[str(name)] = str(dataset_id)

    for name, dataset_cfg in gp_interpolation.items():
        dataset_id = dataset_cfg.get("id")
        if dataset_id is None:
            charging_rate = dataset_cfg.get("charging_rate")
            if charging_rate is not None:
                dataset_id = f"{float(charging_rate):.2f}C"
        if dataset_id is not None:
            labels[str(name)] = str(dataset_id)
    return labels


def find_latent_csvs(artifacts_dir: Path) -> List[Tuple[str, Path]]:
    results: List[Tuple[str, Path]] = []
    for folder in sorted(artifacts_dir.glob("latent_space_data*")):
        if not folder.is_dir():
            continue
        match = re.fullmatch(r"latent_space_(data\d+)", folder.name)
        if not match:
            continue
        csv_path = folder / "latent_space.csv"
        if csv_path.exists():
            results.append((match.group(1), csv_path))
    return results


def find_gp_interpolation_csvs(
    config_path: Path,
    gp_data: Optional[list[str]] = None,
) -> List[Tuple[str, Path]]:
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    gp_datasets = dict(config.get("GP_INTERPOLATION_DATASETS", {}) or {})
    requested = [item.lower() for item in gp_data] if gp_data is not None else []
    include_all = "all" in requested
    results: List[Tuple[str, Path]] = []
    for dataset_key, dataset_cfg in sorted(gp_datasets.items()):
        if not include_all and dataset_key.lower() not in requested:
            continue

        csv_path = Path(str(dataset_cfg.get("interpolation_latent_path", "")))
        if not csv_path:
            continue
        if not csv_path.is_absolute():
            csv_path = ROOT / csv_path
        if not csv_path.exists():
            continue
        results.append((str(dataset_key), csv_path))
    return results


def load_latent_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(csv_path)
    latent_cols = [col for col in frame.columns if re.fullmatch(r"z\d+", str(col))]
    latent_cols.sort(key=lambda name: int(name[1:]))
    if len(latent_cols) < 2:
        raise ValueError(f"Expected at least two latent columns in {csv_path}, found {latent_cols}")

    ordered = frame.sort_values("Cycle")
    cycles = ordered["Cycle"].to_numpy(dtype=float)
    latents = ordered[[latent_cols[0], latent_cols[1]]].to_numpy(dtype=float)
    return cycles, latents


def gp_label_from_csv(csv_path: Path) -> str:
    match = re.fullmatch(r"latent_space_(\d+\.\d+)C\.csv", csv_path.name)
    if not match:
        return csv_path.stem
    return f"GP interpolated {float(match.group(1)):.2f}C"


def annotate_standard_label(ax: plt.Axes, xy: np.ndarray, label: str, dx: float, dy: float) -> None:
    label_loc = {
        "1.00C": (xy[0] - 5 * dx, xy[1] + 6 * dy),
        "0.85C": (xy[0] -10 * dx, xy[1] + 6 * dy),
        "0.75C": (xy[0] - 5 * dx, xy[1] + 7 * dy),
        "0.60C": (xy[0] - 5 * dx, xy[1] + 7 * dy),
        "0.50C": (xy[0] - 3 * dx, xy[1] + 6 * dy),
        "0.30C": (xy[0] - 8 * dx, xy[1] + 3 * dy),
        "0.20C": (xy[0] - 8 * dx, xy[1] + 5 * dy),
    }
    text_x, text_y = label_loc.get(label, (xy[0] - 8 * dx, xy[1] + min(6 * dy, 1.5)))

    ax.text(
        text_x,
        text_y,
        label,
        fontsize=14,
        weight="bold",
        color="black",
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "0.80", "alpha": 0.92},
        zorder=5,
    )


def annotate_gp_label(ax: plt.Axes, xy: np.ndarray, label: str, dx: float, dy: float) -> None:
    ax.annotate(
        label,
        xy=(xy[0], xy[1]),
        xytext=(xy[0] - dx * 2, xy[1] -  dy * 2),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#8a5300"},
        ha="left",
        va="center",
        zorder=7,
        **GP_TEXT_STYLE,
    )


def main() -> None:
    args = parse_args()
    rate_labels = load_rate_labels(CONFIG_PATH)
    latent_files = find_latent_csvs(ARTIFACTS_DIR)
    gp_files = find_gp_interpolation_csvs(CONFIG_PATH, gp_data=args.gp_data)
    if not latent_files:
        raise SystemExit("No artifacts/latent_space_data*/latent_space.csv files found.")
    if args.gp_data and not gp_files:
        requested = ", ".join(args.gp_data)
        raise SystemExit(f"No GP interpolation CSV found for requested GP_INTERPOLATION_DATASETS dataset(s): {requested}.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(publication_style()):
        fig, ax = plt.subplots(figsize=(7.4, 5.8), dpi=300)
        cmap = plt.get_cmap("turbo")
        all_series: List[Tuple[str, np.ndarray, np.ndarray, bool, str]] = []
        global_cycle_min = float("inf")
        global_cycle_max = float("-inf")

        for dataset_key, csv_path in latent_files:
            cycles, latents = load_latent_csv(csv_path)
            label = rate_labels.get(dataset_key, dataset_key)
            all_series.append((dataset_key, cycles, latents, False, label))
            global_cycle_min = min(global_cycle_min, float(1))
            # global_cycle_min = min(global_cycle_min, float(np.min(cycles)))

            global_cycle_max = max(global_cycle_max, float(np.max(cycles)))

        for dataset_key, csv_path in gp_files:
            cycles, latents = load_latent_csv(csv_path)
            label = gp_label_from_csv(csv_path)
            all_series.append((dataset_key, cycles, latents, True, label))
            global_cycle_min = min(global_cycle_min, float(np.min(cycles)))
            global_cycle_max = max(global_cycle_max, float(np.max(cycles)))

        norm = Normalize(vmin=global_cycle_min, vmax=global_cycle_max)

        ax.minorticks_on()
        ax.tick_params(which="both", top=True, right=True)

        for dataset_key, cycles, latents, is_gp, label in all_series:
            if is_gp:
                ax.plot(
                    latents[:, 0],
                    latents[:, 1],
                    color="#6b6b6b",
                    linewidth=1.8,
                    alpha=0.85,
                    zorder=4,
                )
                sample_step = max(1, len(latents) // 220)
                sampled = slice(None, None, sample_step)
                ax.scatter(
                    latents[sampled, 0],
                    latents[sampled, 1],
                    s=24,
                    c=cycles[sampled],
                    cmap=cmap,
                    norm=norm,
                    alpha=0.95,
                    edgecolors="white",
                    linewidths=0.25,
                    zorder=5,
                )
            else:
                ax.scatter(
                    latents[:, 0],
                    latents[:, 1],
                    s=18,
                    c=cycles,
                    cmap=cmap,
                    norm=norm,
                    alpha=0.90,
                    edgecolors="none",
                    zorder=3,
                )

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        dx = 0.012 * (x_max - x_min)
        dy = 0.010 * (y_max - y_min)

        for dataset_key, _, latents, is_gp, label in all_series:
            start = latents[-1]
            if is_gp:
                gp_dx = 0.03 * (x_max - x_min)
                gp_dy = 0.04 * (y_max - y_min) if "0.55" in label else 0.075 * (y_max - y_min)
                annotate_gp_label(ax, start, label, gp_dx, gp_dy)
            else:
                annotate_standard_label(ax, start, label, dx, dy)

        ax.set_xlabel(r"Latent Dimension 1 ($z_1$)")
        ax.set_ylabel(r"Latent Dimension 2 ($z_2$)")

        ax.grid(False)
        ax.margins(x=0.08, y=0.08)

        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("Cycle Number")
        tick_values = np.linspace(global_cycle_min, global_cycle_max, 5)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f"{int(round(value))}" for value in tick_values])
        cbar.outline.set_linewidth(0.8)

        fig.tight_layout()
        fig.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=600)
        plt.close(fig)

    print(f"Saved overlay plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
