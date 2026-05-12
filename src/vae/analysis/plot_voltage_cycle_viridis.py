"""
Create a publication-style voltage-versus-time plot colored by cycle number.

The script reads one raw dataset file, converts each cycle to a relative-time
trace, and plots the full cycle family with a viridis colormap where color
encodes the cycle index.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd
try:
    import yaml
except ImportError:
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "vae.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "analysis"
REQUIRED_COLUMNS = ["Cycle", "Time", "Voltage"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot voltage curves for one dataset with cycle number encoded by a "
            "viridis colormap."
        )
    )
    parser.add_argument(
        "--dataset-key",
        default="data5",
        help="Dataset key defined in the VAE config, for example 'data5'.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Direct path to a CSV file. Overrides --dataset-key when provided.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the figure files will be written.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Base filename without extension. Defaults to the dataset label.",
    )
    parser.add_argument(
        "--cycle-min",
        type=int,
        default=None,
        help="Minimum cycle number to include.",
    )
    parser.add_argument(
        "--cycle-max",
        type=int,
        default=None,
        help="Maximum cycle number to include.",
    )
    parser.add_argument(
        "--cycle-stride",
        type=int,
        default=1,
        help="Keep every n-th cycle. Use values > 1 for lighter figures.",
    )
    parser.add_argument(
        "--point-stride",
        type=int,
        default=1,
        help="Keep every n-th point inside each cycle trace.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Optional cap on the number of displayed cycles after filtering.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI for the PNG export.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title. Leave unset for a cleaner journal-style panel.",
    )
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is not installed.")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_scalar(value: str) -> str | int | float | bool | None:
    if value == "":
        return None

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        return value[1:-1]

    try:
        if any(token in value for token in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_dataset_config_fallback(config_path: Path) -> dict[str, dict]:
    datasets: dict[str, dict] = {}
    current_dataset: str | None = None
    in_datasets_block = False

    with config_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(line.lstrip(" "))

            if indent == 0:
                in_datasets_block = stripped == "DATASETS:"
                current_dataset = None
                continue

            if not in_datasets_block:
                continue

            if indent == 2 and stripped.endswith(":"):
                current_dataset = stripped[:-1]
                datasets[current_dataset] = {}
                continue

            if current_dataset is None or indent < 4 or ":" not in stripped:
                continue

            key, raw_value = stripped.split(":", 1)
            value = raw_value.split("#", 1)[0].strip()
            datasets[current_dataset][key.strip()] = parse_scalar(value)

    return datasets


def load_dataset_entry(config_path: Path, dataset_key: str) -> dict:
    if yaml is not None:
        config = load_yaml_config(config_path)
        return config.get("DATASETS", {}).get(dataset_key)

    datasets = load_dataset_config_fallback(config_path)
    return datasets.get(dataset_key)


def resolve_data_path(config_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    config_relative = (config_path.parent / candidate).resolve()
    if config_relative.exists():
        return config_relative

    repo_relative = (REPO_ROOT / candidate).resolve()
    if repo_relative.exists():
        return repo_relative

    return config_relative


def resolve_dataset(args: argparse.Namespace) -> tuple[Path, str]:
    if args.csv_path is not None:
        csv_path = args.csv_path.expanduser().resolve()
        return csv_path, csv_path.stem

    config_path = args.config.expanduser().resolve()
    dataset_info = load_dataset_entry(config_path, args.dataset_key)
    if dataset_info is None:
        available = ", ".join(sorted(load_dataset_config_fallback(config_path)))
        raise KeyError(
            f"Dataset key '{args.dataset_key}' was not found in {config_path}. "
            f"Available keys: {available}"
        )

    csv_path = resolve_data_path(config_path, str(dataset_info["path"]))
    dataset_label = dataset_info.get("id", args.dataset_key)
    return csv_path, f"{args.dataset_key}_{dataset_label}"


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.major.size": 5,
            "ytick.minor.size": 3,
            "xtick.major.width": 1.0,
            "xtick.minor.width": 0.8,
            "ytick.major.width": 1.0,
            "ytick.minor.width": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_voltage_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {csv_path}")

    dataframe = pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS)
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    dataframe = dataframe.dropna(subset=REQUIRED_COLUMNS).copy()
    dataframe["Cycle"] = dataframe["Cycle"].astype(int)
    dataframe = dataframe.sort_values(["Cycle", "Time"], kind="mergesort")
    return dataframe


def select_cycles(
    cycles: np.ndarray,
    cycle_min: int | None,
    cycle_max: int | None,
    cycle_stride: int,
    max_cycles: int | None,
) -> np.ndarray:
    if cycle_stride < 1:
        raise ValueError("--cycle-stride must be >= 1")
    if max_cycles is not None and max_cycles < 1:
        raise ValueError("--max-cycles must be >= 1")

    selected = cycles
    if cycle_min is not None:
        selected = selected[selected >= cycle_min]
    if cycle_max is not None:
        selected = selected[selected <= cycle_max]

    selected = selected[::cycle_stride]

    if max_cycles is not None and len(selected) > max_cycles:
        step = math.ceil(len(selected) / max_cycles)
        selected = selected[::step]

    if len(selected) == 0:
        raise ValueError("No cycles remain after applying the requested filters.")

    return selected


def downsample_trace(trace: np.ndarray, point_stride: int) -> np.ndarray:
    if point_stride < 1:
        raise ValueError("--point-stride must be >= 1")
    if point_stride == 1 or len(trace) <= 2:
        return trace

    reduced = trace[::point_stride]
    if not np.array_equal(reduced[-1], trace[-1]):
        reduced = np.vstack([reduced, trace[-1]])
    return reduced


def build_cycle_segments(
    dataframe: pd.DataFrame,
    selected_cycles: Iterable[int],
    point_stride: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    segments: list[np.ndarray] = []
    cycle_values: list[int] = []

    for cycle, cycle_frame in dataframe.groupby("Cycle", sort=True):
        if cycle not in selected_cycles:
            continue

        elapsed_time = cycle_frame["Time"].to_numpy(dtype=np.float32)
        elapsed_time = elapsed_time - elapsed_time[0]
        voltage = cycle_frame["Voltage"].to_numpy(dtype=np.float32)

        trace = np.column_stack((elapsed_time, voltage))
        trace = downsample_trace(trace, point_stride)

        if len(trace) < 2:
            continue

        segments.append(trace)
        cycle_values.append(int(cycle))

    if not segments:
        raise ValueError("No valid cycle traces were prepared for plotting.")

    return segments, np.asarray(cycle_values, dtype=np.float32)


def plot_voltage_map(
    segments: list[np.ndarray],
    cycle_values: np.ndarray,
    output_base: Path,
    dataset_label: str,
    dpi: int,
    title: str | None = None,
) -> None:
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)

    vmin = float(cycle_values.min())
    vmax = float(cycle_values.max())
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    collection = LineCollection(
        segments,
        cmap="viridis",
        norm=norm,
        linewidths=0.55,
        alpha=0.9,
        rasterized=True,
        capstyle="round",
        joinstyle="round",
    )
    collection.set_array(cycle_values)
    ax.add_collection(collection)
    ax.autoscale()
    ax.margins(x=0.02, y=0.04)

    ax.set_xlabel("Relative time, $t$ (s)")
    ax.set_ylabel("Voltage (V)")
    if title:
        ax.set_title(title)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(top=True, right=True)

    colorbar = fig.colorbar(collection, ax=ax, pad=0.02)
    colorbar.set_label("Cycle number")

    ax.text(
        0.02,
        0.98,
        dataset_label.replace("_", " "),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.5},
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.parent / f"{output_base.name}.png"
    pdf_path = output_base.parent / f"{output_base.name}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path, dataset_label = resolve_dataset(args)
    dataframe = load_voltage_data(csv_path)

    available_cycles = np.sort(dataframe["Cycle"].unique())
    selected_cycles = select_cycles(
        available_cycles,
        cycle_min=args.cycle_min,
        cycle_max=args.cycle_max,
        cycle_stride=args.cycle_stride,
        max_cycles=args.max_cycles,
    )
    selected_cycle_set = set(int(cycle) for cycle in selected_cycles.tolist())
    segments, cycle_values = build_cycle_segments(
        dataframe,
        selected_cycles=selected_cycle_set,
        point_stride=args.point_stride,
    )

    output_name = args.output_name or f"voltage_cycle_viridis_{dataset_label}"
    output_base = args.output_dir / output_name
    plot_voltage_map(
        segments=segments,
        cycle_values=cycle_values,
        output_base=output_base,
        dataset_label=dataset_label,
        dpi=args.dpi,
        title=args.title,
    )

    print(f"Saved figure to {output_base.parent / f'{output_base.name}.png'}")
    print(f"Saved figure to {output_base.parent / f'{output_base.name}.pdf'}")


if __name__ == "__main__":
    main()
