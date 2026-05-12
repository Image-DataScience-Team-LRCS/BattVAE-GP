from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

REPO_ROOT = Path(__file__).resolve().parents[3]
if __package__ is None or __package__ == "":
    sys.path.append(str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "gp_outputs"

from src.common.logger.logging import get_logger, setup_logging


logger = get_logger(__name__)

TARGET_COLORS = {
    "z1": "#1f7a8c",
    "z2": "#c06c2b",
}

FOLD_COLORS = {
    "0.60C": "#3b5b92",
    "0.75C": "#b85c38",
}

PUBLICATION_STYLE = {
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 0.9,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.titlesize": 20,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "axes.grid": False,
}

SELECTION_METRIC_COLUMNS = (
    "val_mae",
    "val_rmse",
    "val_nll",
    "val_mean_std",
    "val_coverage_50",
    "val_width_50",
    "val_coverage_90",
    "val_width_90",
    "val_coverage_95",
    "val_width_95",
)


def crate_sort_key(label: str) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(label))
    return float(match.group(1)) if match else float("inf")


def ordered_crate_labels(values: pd.Series) -> list[str]:
    labels = list(dict.fromkeys(values.astype(str).tolist()))
    return sorted(labels, key=crate_sort_key)


def format_target_label(target_column: str) -> str:
    if re.fullmatch(r"z\d+", str(target_column)):
        return rf"$z_{{{target_column[1:]}}}$"
    return str(target_column)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate nested GP analysis figures from analysis tables.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="GP output directory.",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    plt.style.use("bmh")
    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "font.family": "DejaVu Sans",
            "axes.grid": False,
        }
    )
    if sns is not None:
        sns.set_theme(style="ticks", context="talk", rc={"axes.grid": False})


def apply_publication_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(PUBLICATION_STYLE)
    if sns is not None:
        sns.set_theme(
            style="ticks",
            context="paper",
            rc=PUBLICATION_STYLE,
        )


def require_seaborn(plot_name: str) -> None:
    if sns is None:
        raise ModuleNotFoundError(f"seaborn is required for '{plot_name}'.")


def load_table(tables_dir: Path, filename: str) -> pd.DataFrame:
    path = tables_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Required analysis table not found: {path}")
    return pd.read_csv(path)


def load_optional_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def disable_figure_grids(fig: plt.Figure) -> None:
    for ax in fig.axes:
        ax.grid(False)
        ax.xaxis.grid(False, which="both")
        ax.yaxis.grid(False, which="both")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    disable_figure_grids(fig)
    # fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", path)


def save_publication_figure(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    disable_figure_grids(fig)
    # fig.tight_layout()
    for suffix, dpi in ((".png", 400), (".pdf", None)):
        output_path = base_path.with_suffix(suffix)
        save_kwargs: dict[str, object] = {"bbox_inches": "tight"}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)
        logger.info("Saved publication figure to %s", output_path)
    plt.close(fig)


def _resolved_config(output_dir: Path) -> dict[str, object] | None:
    config_path = output_dir / "resolved_config.json"
    if not config_path.exists():
        return None
    import json

    return json.loads(config_path.read_text(encoding="utf-8"))


def _selection_details_from_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if "inner_fold_file" not in summary_df.columns:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for inner_path_raw, group_df in summary_df.groupby("inner_fold_file", dropna=False):
        if pd.isna(inner_path_raw) or not inner_path_raw:
            continue
        path = Path(str(inner_path_raw))
        if not path.exists():
            continue
        inner_df = pd.read_csv(path)
        if "outer_test_key" not in inner_df.columns:
            inner_df["outer_test_key"] = group_df["outer_test_key"].iloc[0]
        if "outer_test_label" not in inner_df.columns:
            inner_df["outer_test_label"] = group_df["outer_test_label"].iloc[0]
        if "target_column" in inner_df.columns:
            frames.append(inner_df)
            continue

        target_columns = [str(value) for value in group_df["target_column"].dropna().unique().tolist()]
        expanded_frames: list[pd.DataFrame] = []
        for target_column in target_columns:
            target_df = inner_df.copy()
            target_df["target_column"] = target_column
            copied_metric = False
            for metric_column in SELECTION_METRIC_COLUMNS:
                source_column = f"{metric_column}_{target_column}"
                if source_column in target_df.columns:
                    target_df[metric_column] = target_df[source_column]
                    copied_metric = True
            if copied_metric:
                expanded_frames.append(target_df)

        if expanded_frames:
            frames.extend(expanded_frames)
            continue

        if target_columns:
            inner_df["target_column"] = target_columns[0]
        frames.append(inner_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _selection_metric_column(selection_df: pd.DataFrame, target_column: str, metric_key: str) -> str:
    if metric_key in selection_df.columns:
        return metric_key
    target_metric_key = f"{metric_key}_{target_column}"
    if target_metric_key in selection_df.columns:
        return target_metric_key
    raise KeyError(metric_key)


def add_panel_labels(axes: np.ndarray | list[plt.Axes]) -> None:
    flat_axes = np.array(axes, dtype=object).ravel()
    for index, ax in enumerate(flat_axes):
        ax.text(
            -0.12,
            1.05,
            f"({chr(97 + index)}).",
            transform=ax.transAxes,
            fontsize=30,
            fontweight="bold",
            va="bottom",
        )


def downsample_frame(frame: pd.DataFrame, max_points: int = 700) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame.copy()
    indices = np.linspace(0, len(frame) - 1, max_points, dtype=int)
    return frame.iloc[indices].copy()


def prediction_std_band(predictions_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    lower = predictions_df["prediction_mean"] - predictions_df["prediction_std"]
    upper = predictions_df["prediction_mean"] + predictions_df["prediction_std"]
    return lower, upper


def plot_fold_performance(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    ordered_labels = ordered_crate_labels(summary_df["outer_test_label"])
    plot_df = summary_df.copy()
    plot_df["outer_test_label"] = pd.Categorical(plot_df["outer_test_label"], categories=ordered_labels, ordered=True)
    plot_df = plot_df.sort_values(["target_column", "outer_test_label"])

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, metric, title in zip(axes, ["rmse", "nll"], ["RMSE", "Test NLL"], strict=False):
        for target_column, target_df in plot_df.groupby("target_column", sort=False):
            ax.plot(
                target_df["outer_test_label"].astype(str),
                target_df[metric],
                marker="o",
                linewidth=2.0,
                label=target_column,
            )
        ax.set_ylabel(title)
        ax.set_xlabel("")
        ax.legend()
    axes[1].set_xlabel("Held-Out C-rate")
    save_figure(fig, figures_dir / "fold_performance_metrics.png")


def plot_coverage_by_fold(coverage_df: pd.DataFrame, figures_dir: Path) -> None:
    ordered_labels = ordered_crate_labels(coverage_df["outer_test_label"])
    nominal_order = {"50%": 0, "90%": 1, "95%": 2}
    plot_df = coverage_df.copy()
    plot_df["coverage_label"] = plot_df["outer_test_label"].astype(str) + "\n" + plot_df["nominal_label"].astype(str)
    ordered_pairs = sorted(
        plot_df[["outer_test_label", "nominal_label", "coverage_label"]].drop_duplicates().itertuples(index=False),
        key=lambda row: (crate_sort_key(row.outer_test_label), nominal_order.get(row.nominal_label, 99)),
    )
    ordered_coverage_labels = [row.coverage_label for row in ordered_pairs]
    fig, ax = plt.subplots(figsize=(13, 7))
    targets = list(dict.fromkeys(coverage_df["target_column"].tolist()))
    bar_width = 0.35
    x_positions = np.arange(len(ordered_coverage_labels))
    for index, target_column in enumerate(targets):
        target_df = plot_df.loc[plot_df["target_column"] == target_column].copy()
        target_df["coverage_label"] = pd.Categorical(
            target_df["coverage_label"],
            categories=ordered_coverage_labels,
            ordered=True,
        )
        target_df = target_df.sort_values("coverage_label")
        ax.bar(
            x_positions + (index - (len(targets) - 1) / 2) * bar_width,
            target_df["empirical_coverage"],
            width=bar_width,
            label=target_column,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ordered_coverage_labels)
    ax.set_xlabel("Held-Out C-rate and nominal interval")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Empirical 90/95% Coverage by Held-Out C-rate and Target")
    ax.legend()
    save_figure(fig, figures_dir / "coverage_by_fold.png")


def plot_prediction_intervals(predictions_df: pd.DataFrame, figures_dir: Path) -> None:
    labels = ordered_crate_labels(predictions_df["outer_test_label"])
    targets = list(dict.fromkeys(predictions_df["target_column"].tolist()))
    n_rows = len(targets)
    n_cols = len(labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]

    for row_index, target_column in enumerate(targets):
        row_axes = axes[row_index] if n_cols > 1 else [axes[row_index]]
        for ax, held_out_label in zip(row_axes, labels, strict=False):
            fold_df = predictions_df.loc[
                (predictions_df["outer_test_label"] == held_out_label)
                & (predictions_df["target_column"] == target_column)
            ].sort_values("Cycle")
            lower_std, upper_std = prediction_std_band(fold_df)
            ax.plot(fold_df["Cycle"], fold_df[target_column], label="True", linewidth=1.8)
            ax.plot(fold_df["Cycle"], fold_df["prediction_mean"], label="GP mean", linewidth=1.8)
            ax.fill_between(
                fold_df["Cycle"],
                lower_std,
                upper_std,
                alpha=0.25,
                label="Mean ± std",
            )
            ax.set_title(f"{target_column} | {held_out_label}")
            ax.set_xlabel("Cycle")
            ax.set_ylabel(target_column)
            if row_index == 0 and held_out_label == labels[0]:
                ax.legend()

    save_figure(fig, figures_dir / "prediction_intervals_by_fold.png")


def plot_uncertainty_vs_error(predictions_df: pd.DataFrame, figures_dir: Path) -> None:
    sampled = predictions_df.copy()
    if len(sampled) > 10000:
        sampled = sampled.sample(10000, random_state=42)
    fig, ax = plt.subplots(figsize=(9, 7))
    for (target_column, held_out_label), fold_df in sampled.groupby(["target_column", "outer_test_label"], sort=False):
        ax.scatter(
            fold_df["prediction_std"],
            fold_df["abs_error"],
            alpha=0.35,
            s=20,
            label=f"{target_column} | {held_out_label}",
        )
    ax.set_xlabel("Predicted Standard Deviation")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Uncertainty vs Absolute Error")
    ax.legend()
    save_figure(fig, figures_dir / "uncertainty_vs_absolute_error.png")


def plot_loss_curves(history_df: pd.DataFrame, figures_dir: Path) -> None:
    if history_df.empty:
        logger.warning("No history data available for plotting.")
        return

    labels = ordered_crate_labels(history_df["outer_test_label"])
    targets = list(dict.fromkeys(history_df["target_column"].tolist()))
    n_rows = len(targets)
    n_cols = len(labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]

    for row_index, target_column in enumerate(targets):
        row_axes = axes[row_index] if n_cols > 1 else [axes[row_index]]
        for ax, held_out_label in zip(row_axes, labels, strict=False):
            fold_df = history_df.loc[
                (history_df["target_column"] == target_column)
                & (history_df["outer_test_label"] == held_out_label)
            ].sort_values("epoch")
            ax.plot(fold_df["epoch"], fold_df["train_loss"], label="Train", linewidth=1.8)
            if fold_df["test_nll"].notna().any():
                ax.plot(fold_df["epoch"], fold_df["test_nll"], label="Test NLL", linewidth=1.8)
            ax.set_title(f"{target_column} | {held_out_label}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            if row_index == 0 and held_out_label == labels[0]:
                ax.legend()

    save_figure(fig, figures_dir / "loss_curves_final.png")


def plot_validation_stability_summary(
    selection_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    publication_dir: Path,
) -> None:

    outer_labels = ordered_crate_labels(selection_df["outer_test_label"])
    target_order = list(dict.fromkeys(selection_df["target_column"].tolist()))
    metric_specs = [
        ("rmse", "RMSE", "val_rmse"),
        ("nll", "NLL", "val_nll"),
    ]

    fig, axes = plt.subplots(
        len(metric_specs),
        len(outer_labels),
        figsize=(11, 9),
        sharex=False,
        constrained_layout=False,
    )

    # Ensure axes is iterable (works for 1 or many subplots)
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    if len(metric_specs) == 1:
        axes = np.array([axes])
    if len(outer_labels) == 1:
        axes = axes[:, np.newaxis]

    for col_index, outer_label in enumerate(outer_labels):
        outer_selection = selection_df.loc[selection_df["outer_test_label"] == outer_label].copy()
        validation_labels = ordered_crate_labels(outer_selection["validation_label"])
        x_labels = ordered_crate_labels(pd.Series(validation_labels + [outer_label]))
        x_positions = np.arange(len(x_labels))
        x_lookup = {label: idx for idx, label in enumerate(x_labels)}

        for row_index, (summary_metric_key, metric_label, validation_metric_key) in enumerate(metric_specs):
            ax = axes[row_index, col_index]
            for target_column in target_order:
                color = TARGET_COLORS.get(target_column, "#4c78a8")
                target_selection = outer_selection.loc[outer_selection["target_column"] == target_column].copy()
                validation_metric_column = _selection_metric_column(
                    target_selection,
                    target_column,
                    validation_metric_key,
                )
                target_selection["validation_label"] = pd.Categorical(
                    target_selection["validation_label"],
                    categories=validation_labels,
                    ordered=True,
                )
                target_selection = target_selection.sort_values("validation_label")
                validation_x = [x_lookup[label] for label in target_selection["validation_label"].astype(str)]
                validation_y = target_selection[validation_metric_column].to_numpy()

                ax.plot(validation_x, validation_y, color=color, linewidth=2, alpha=0.95)
                ax.scatter(
                    validation_x,
                    validation_y,
                    color=color,
                    s=90,
                    marker="o",
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=3,
                )

                test_row = summary_df.loc[
                    (summary_df["outer_test_label"] == outer_label)
                    & (summary_df["target_column"] == target_column)
                ].iloc[0]
                ax.scatter(
                    x_lookup[outer_label],
                    test_row[summary_metric_key],
                    color=color,
                    s=90,
                    marker="s",
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=4,
                )

            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=30)
            ax.tick_params(axis="y", which="major", labelsize=26)
            ax.set_ylabel(metric_label, fontsize=30)
            if row_index == 0:
                ax.set_title(f"Outer test held-out = {outer_label}", fontsize=30)
            if row_index == len(metric_specs) - 1:
                ax.set_xlabel("C-rate", fontsize=30)


    target_handles = [
        Line2D(
            [0],
            [0],
            color=TARGET_COLORS.get(target_column, "#4c78a8"),
            marker="o",
            linewidth=2,
            label=format_target_label(target_column),
        )
        for target_column in target_order
    ]
    split_handles = [
        Line2D([0], [0], color="#444444", marker="o", linestyle="None", markersize=8, label="Held-out validation set"),
        Line2D([0], [0], color="#444444", marker="s", linestyle="None", markersize=8, label="Test set"),
    ]
    legend_handles = target_handles + split_handles
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=30,
    )
    # fig.subplots_adjust(left=0.09, right=0.985, bottom=0.14, top=0.84, wspace=0.4, hspace=0.7)
    fig.subplots_adjust(left=0.0, right=1.1, bottom=0.01, top=0.84, wspace=0.25, hspace=0.7)

    add_panel_labels(axes)
    save_publication_figure(fig, publication_dir / "figure_01_validation_stability")


def plot_publication_trajectory_fits(
    predictions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    publication_dir: Path,
) -> None:
    labels = ordered_crate_labels(predictions_df["outer_test_label"])
    targets = list(dict.fromkeys(summary_df["target_column"].tolist()))
    fig, axes = plt.subplots(
        len(targets),
        len(labels),
        figsize=(11, 7),
        sharex=True,
        constrained_layout=False,
    )
    if len(targets) == 1:
        axes = np.array([axes])

    for row_index, target_column in enumerate(targets):
        for col_index, held_out_label in enumerate(labels):
            ax = axes[row_index, col_index]
            fold_df = predictions_df.loc[
                (predictions_df["target_column"] == target_column)
                & (predictions_df["outer_test_label"] == held_out_label)
            ].sort_values("Cycle")
            fold_df = downsample_frame(fold_df, max_points=750)
            lower_std, upper_std = prediction_std_band(fold_df)
            metrics_row = summary_df.loc[
                (summary_df["target_column"] == target_column)
                & (summary_df["outer_test_label"] == held_out_label)
            ].iloc[0]

            ax.fill_between(
                fold_df["Cycle"],
                lower_std,
                upper_std,
                color=TARGET_COLORS.get(target_column, "#4c78a8"),
                alpha=0.18,
                linewidth=0,
                label="Mean ± std" if row_index == 0 and col_index == 0 else None,
            )
            ax.plot(
                fold_df["Cycle"],
                fold_df[target_column],
                color="#1d1d1d",
                linewidth=1.7,
                label="Observed" if row_index == 0 and col_index == 0 else None,
            )
            ax.plot(
                fold_df["Cycle"],
                fold_df["prediction_mean"],
                color=TARGET_COLORS.get(target_column, "#4c78a8"),
                linewidth=1.8,
                label="GP mean" if row_index == 0 and col_index == 0 else None,
            )
            ax.set_title(f"{format_target_label(target_column)} | held-out {held_out_label}")
            ax.set_xlabel("Cycle")
            ax.set_ylabel(format_target_label(target_column))
            ax.text(
                0.03,
                0.97,
                f"RMSE = {metrics_row['rmse']:.3f}\nMAE = {metrics_row['mae']:.3f}\nNLL = {metrics_row['nll']:.2f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "#cfcfcf", "boxstyle": "round,pad=0.25"},
            )

    handles, labels_text = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_text, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    add_panel_labels(axes)
    save_publication_figure(fig, publication_dir / "figure_01_trajectory_fits")


def plot_publication_fold_metrics(summary_df: pd.DataFrame, publication_dir: Path) -> None:
    ordered_labels = ordered_crate_labels(summary_df["outer_test_label"])
    plot_df = summary_df.copy()
    plot_df["outer_test_label"] = pd.Categorical(plot_df["outer_test_label"], categories=ordered_labels, ordered=True)
    plot_df = plot_df.sort_values(["outer_test_label", "target_column"])

    metrics = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("nll", "Test NLL"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(11.5, 3.7), sharey=True, constrained_layout=False)
    y_positions = np.arange(len(ordered_labels))

    for ax, (metric_key, metric_label) in zip(axes, metrics, strict=False):
        for y_index, held_out_label in enumerate(ordered_labels):
            row_df = plot_df.loc[plot_df["outer_test_label"] == held_out_label].sort_values("target_column")
            metric_values = row_df[metric_key].to_numpy()
            ax.plot(metric_values, [y_index, y_index], color="#c7c7c7", linewidth=1.2, zorder=1)
            for _, row in row_df.iterrows():
                ax.scatter(
                    row[metric_key],
                    y_index,
                    s=62,
                    color=TARGET_COLORS.get(row["target_column"], "#4c78a8"),
                    edgecolors="white",
                    linewidths=0.7,
                    zorder=2,
                    label=format_target_label(row["target_column"]),
                )
        ax.set_title(metric_label)
        ax.set_xlabel(metric_label)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(ordered_labels)
    axes[0].set_ylabel("Held-out C-rate")

    handles, labels_text = axes[0].get_legend_handles_labels()
    dedup_handles: list[object] = []
    dedup_labels: list[str] = []
    for handle, label in zip(handles, labels_text, strict=False):
        if label not in dedup_labels:
            dedup_labels.append(label)
            dedup_handles.append(handle)
    fig.legend(dedup_handles, dedup_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    add_panel_labels(axes)
    save_publication_figure(fig, publication_dir / "figure_02_fold_metrics")


def plot_publication_parity(predictions_df: pd.DataFrame, publication_dir: Path) -> None:
    targets = list(dict.fromkeys(predictions_df["target_column"].tolist()))
    labels = ordered_crate_labels(predictions_df["outer_test_label"])
    fig, axes = plt.subplots(1, len(targets), figsize=(10.5, 4.3), constrained_layout=False)
    if len(targets) == 1:
        axes = np.array([axes])

    for ax, target_column in zip(axes, targets, strict=False):
        target_df = predictions_df.loc[predictions_df["target_column"] == target_column].copy()
        if len(target_df) > 3500:
            sampled_frames = []
            for _, frame in target_df.groupby("outer_test_label", sort=False):
                sampled_frames.append(frame.sample(min(len(frame), 1750), random_state=42))
            target_df = pd.concat(sampled_frames, ignore_index=True)

        actual = target_df[target_column]
        predicted = target_df["prediction_mean"]
        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())
        padding = 0.04 * (upper - lower if upper > lower else 1.0)

        for held_out_label in labels:
            fold_df = target_df.loc[target_df["outer_test_label"] == held_out_label]
            ax.scatter(
                fold_df[target_column],
                fold_df["prediction_mean"],
                s=12,
                alpha=0.45,
                color=FOLD_COLORS.get(held_out_label, "#4c78a8"),
                edgecolors="none",
                rasterized=True,
                label=held_out_label,
            )

        ax.plot(
            [lower - padding, upper + padding],
            [lower - padding, upper + padding],
            color="#222222",
            linestyle="--",
            linewidth=1.1,
        )
        ax.set_xlim(lower - padding, upper + padding)
        ax.set_ylim(lower - padding, upper + padding)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Observed {format_target_label(target_column)}")
        ax.set_ylabel(f"Predicted {format_target_label(target_column)}")
        ax.set_title(f"Parity for {format_target_label(target_column)}")

    handles, labels_text = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_text, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.03))
    add_panel_labels(axes)
    save_publication_figure(fig, publication_dir / "figure_02_parity")


def plot_publication_calibration(
    calibration_df: pd.DataFrame,
    cycle_df: pd.DataFrame,
    publication_dir: Path,
) -> None:
    targets = list(dict.fromkeys(calibration_df["target_column"].tolist()))
    labels = ordered_crate_labels(cycle_df["outer_test_label"])
    fig, axes = plt.subplots(2, len(targets), figsize=(10.5, 7.2), constrained_layout=False)
    if len(targets) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col_index, target_column in enumerate(targets):
        calibration_ax = axes[0, col_index]
        target_calibration = calibration_df.loc[calibration_df["target_column"] == target_column].sort_values("nominal_coverage")
        calibration_ax.errorbar(
            target_calibration["nominal_coverage"],
            target_calibration["empirical_coverage"],
            yerr=target_calibration["empirical_coverage_std"],
            color=TARGET_COLORS.get(target_column, "#4c78a8"),
            marker="o",
            markersize=5,
            linewidth=1.8,
            capsize=3,
        )
        calibration_ax.plot([0.45, 1.0], [0.45, 1.0], color="#444444", linestyle="--", linewidth=1.0)
        calibration_ax.set_xlim(0.45, 0.98)
        calibration_ax.set_ylim(0.0, 1.05)
        calibration_ax.set_title(f"Calibration: {format_target_label(target_column)}")
        calibration_ax.set_ylabel("Empirical coverage")

        coverage_ax = axes[1, col_index]
        target_cycle = cycle_df.loc[cycle_df["target_column"] == target_column].copy()
        target_cycle["cycle_mid"] = (target_cycle["cycle_start"] + target_cycle["cycle_end"]) / 2.0
        for held_out_label in labels:
            fold_df = target_cycle.loc[target_cycle["outer_test_label"] == held_out_label].sort_values("cycle_mid")
            coverage_ax.plot(
                fold_df["cycle_mid"],
                fold_df["coverage_50"],
                color=FOLD_COLORS.get(held_out_label, "#4c78a8"),
                linewidth=1.8,
                marker="o",
                markersize=3.5,
                label=held_out_label,
            )
        coverage_ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1.0)
        coverage_ax.set_ylim(0.0, 1.05)
        coverage_ax.set_xlabel("Cycle")
        coverage_ax.set_ylabel("50% interval coverage")
        coverage_ax.set_title(f"Cycle-resolved coverage: {format_target_label(target_column)}")

    axes[0, 0].set_xlabel("Nominal coverage")
    if len(targets) > 1:
        axes[0, 1].set_xlabel("Nominal coverage")

    handles, labels_text = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels_text, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.02))
    add_panel_labels(axes)
    save_publication_figure(fig, publication_dir / "figure_04_calibration")


def plot_publication_test_uncertainty(
    predictions_df: pd.DataFrame,
    publication_dir: Path,
) -> None:
    labels = ordered_crate_labels(predictions_df["outer_test_label"])
    targets = list(dict.fromkeys(predictions_df["target_column"].tolist()))
    cycle_min = 1.0
    cycle_max = float(predictions_df["Cycle"].max())
    cycle_ticks = np.unique(np.r_[cycle_min, np.linspace(1000, cycle_max, 5)]).astype(int)
    fig, axes = plt.subplots(
        len(targets),
        len(labels),
        figsize=(12.8, 7.6),
        sharex=True,
        constrained_layout=False,
    )
    if len(targets) == 1:
        axes = np.array([axes])
    if len(labels) == 1:
        axes = axes[:, np.newaxis]

    for row_index, target_column in enumerate(targets):
        for col_index, held_out_label in enumerate(labels):
            ax = axes[row_index, col_index]
            fold_df = predictions_df.loc[
                (predictions_df["target_column"] == target_column)
                & (predictions_df["outer_test_label"] == held_out_label)
            ].sort_values("Cycle")
            fold_df = downsample_frame(fold_df, max_points=800)
            lower_std, upper_std = prediction_std_band(fold_df)
            color = TARGET_COLORS.get(target_column, "#4c78a8")
            ax.fill_between(
                fold_df["Cycle"],
                lower_std,
                upper_std,
                color=color,
                alpha=0.20,
                linewidth=0,
                label="Mean ± std" if row_index == 0 and col_index == 0 else None,
            )
            ax.plot(
                fold_df["Cycle"],
                fold_df["prediction_mean"],
                color=color,
                linewidth=2.1,
                label="GP mean" if row_index == 0 and col_index == 0 else None,
            )
            ax.plot(
                fold_df["Cycle"],
                fold_df[target_column],
                color="#1f1f1f",
                linewidth=1.7,
                label="Observed (VAE latent space)" if row_index == 0 and col_index == 0 else None,
            )
            ax.set_xlim(cycle_min, cycle_max)
            ax.set_xticks(cycle_ticks)
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(which="major", direction="in", top=True, right=True, length=5.0, width=0.9)
            ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.8, width=0.7)
            ax.tick_params(axis="x", which="major", labelsize=22)

            if row_index == 0:
                ax.set_title(f"Test {held_out_label}", pad=8)
            if col_index == 0:
                ax.set_ylabel(format_target_label(target_column), labelpad=8)
            else:
                ax.set_ylabel("")
            if row_index < len(targets) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Cycle", fontsize=24, labelpad=8)

    handles, labels_text = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_text, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.12, top=0.88, wspace=0.18, hspace=0.12)
    add_panel_labels(axes)
    save_publication_figure(fig, publication_dir / "figure_03_test_uncertainty")


def _load_interpolation_predictions(output_dir: Path) -> pd.DataFrame:
    resolved_config = _resolved_config(output_dir)
    if resolved_config is None:
        return pd.DataFrame()

    interpolation_cfg = dict(resolved_config.get("interpolation", {}) or {})
    dataset_keys = [str(key) for key in interpolation_cfg.get("interpolation_datasets", [])]
    datasets_file_path = resolved_config.get("_datasets_file_path")
    if not dataset_keys or not datasets_file_path:
        return pd.DataFrame()

    try:
        import yaml
    except ModuleNotFoundError:
        return pd.DataFrame()

    datasets_config = yaml.safe_load(Path(str(datasets_file_path)).read_text(encoding="utf-8")) or {}
    interpolation_datasets = datasets_config.get("GP_INTERPOLATION_DATASETS", {}) or {}
    frames: list[pd.DataFrame] = []
    for dataset_key in dataset_keys:
        dataset_meta = interpolation_datasets.get(dataset_key)
        if not isinstance(dataset_meta, dict):
            continue
        output_csv = dataset_meta.get("interpolation_latent_path")
        if not output_csv:
            continue
        path = Path(str(output_csv))
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["dataset_key"] = dataset_key
        frame["dataset_label"] = str(dataset_meta.get("id", dataset_key))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_publication_interpolation_uncertainty(output_dir: Path, publication_dir: Path) -> None:
    interpolation_df = _load_interpolation_predictions(output_dir)
    if interpolation_df.empty:
        logger.warning("No GP interpolation predictions with uncertainty were found.")
        return

    dataset_labels = ordered_crate_labels(interpolation_df["dataset_label"])
    target_columns = [column for column in ("z1", "z2") if column in interpolation_df.columns]
    fig, axes = plt.subplots(
        len(target_columns),
        len(dataset_labels),
        figsize=(11.0, 6.8),
        sharex=True,
        constrained_layout=False,
    )
    if len(target_columns) == 1:
        axes = np.array([axes])
    if len(dataset_labels) == 1:
        axes = axes[:, np.newaxis]

    for row_index, target_column in enumerate(target_columns):
        color = TARGET_COLORS.get(target_column, "#4c78a8")
        for col_index, dataset_label in enumerate(dataset_labels):
            ax = axes[row_index, col_index]
            fold_df = interpolation_df.loc[
                interpolation_df["dataset_label"] == dataset_label
            ].sort_values("Cycle")
            lower_col = f"{target_column}_lower_95"
            upper_col = f"{target_column}_upper_95"
            if lower_col not in fold_df.columns or upper_col not in fold_df.columns:
                continue
            ax.fill_between(
                fold_df["Cycle"],
                fold_df[lower_col],
                fold_df[upper_col],
                color=color,
                alpha=0.18,
                linewidth=0,
                label="95% interval" if row_index == 0 and col_index == 0 else None,
            )
            ax.plot(
                fold_df["Cycle"],
                fold_df[target_column],
                color=color,
                linewidth=1.8,
                label="GP mean" if row_index == 0 and col_index == 0 else None,
            )
            ax.set_title(f"{format_target_label(target_column)} | interp. {dataset_label}")
            ax.set_xlabel("Cycle")
            ax.set_ylabel(format_target_label(target_column))

    handles, labels_text = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_text, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    add_panel_labels(axes)
    save_publication_figure(fig, publication_dir / "figure_05_interpolation_uncertainty")


def run(output_dir: Path) -> None:
    tables_dir = output_dir / "analysis" / "tables"
    figures_dir = output_dir / "analysis" / "figures"
    publication_dir = figures_dir / "publication"
    figures_dir.mkdir(parents=True, exist_ok=True)
    publication_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        log_dir=output_dir / "analysis" / "logs",
        filename_prefix="gp_analysis_plots_",
        console_output=True,
        level=logging.INFO,
    )
    logger.info("Generating nested GP analysis figures from %s", tables_dir)

    summary_df = load_table(tables_dir, "fold_metrics.csv")
    coverage_df = load_table(tables_dir, "coverage_long.csv")
    predictions_df = load_table(tables_dir, "prediction_diagnostics.csv")
    history_df = load_table(tables_dir, "training_history_long.csv")
    selection_df = load_optional_table(output_dir / "selection_details.csv")
    if selection_df is None:
        selection_df = _selection_details_from_summary(summary_df)

    configure_plot_style()
    plot_fold_performance(summary_df, figures_dir)
    plot_coverage_by_fold(coverage_df.loc[coverage_df["nominal_label"].isin(["90%", "95%"])], figures_dir)
    plot_prediction_intervals(predictions_df, figures_dir)
    plot_uncertainty_vs_error(predictions_df, figures_dir)
    plot_loss_curves(history_df, figures_dir)

    apply_publication_style()
    if not selection_df.empty:
        plot_validation_stability_summary(selection_df, summary_df, publication_dir)
    plot_publication_parity(predictions_df, publication_dir)
    plot_publication_test_uncertainty(predictions_df, publication_dir)
    plot_publication_interpolation_uncertainty(output_dir, publication_dir)


def main() -> None:
    args = parse_args()
    run(Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
