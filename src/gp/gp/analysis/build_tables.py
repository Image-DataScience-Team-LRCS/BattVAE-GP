from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if __package__ is None or __package__ == "":
    sys.path.append(str(REPO_ROOT))

from src.common.logger.logging import get_logger, setup_logging
from src.gp.training.loss import interval_bounds


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build nested GP analysis tables from outputs.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs_latent_gp"),
        help="GP output directory.",
    )
    parser.add_argument(
        "--cycle-bins",
        type=int,
        default=10,
        help="Number of cycle bins for per-cycle analysis tables.",
    )
    return parser.parse_args()


def ensure_analysis_directories(output_dir: Path) -> tuple[Path, Path]:
    analysis_dir = output_dir / "analysis"
    tables_dir = analysis_dir / "tables"
    figures_dir = analysis_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def load_summary(output_dir: Path) -> pd.DataFrame:
    summary_path = output_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"GP summary file not found: {summary_path}")
    summary_df = pd.read_csv(summary_path)
    logger.info("Loaded summary table from %s with %d outer folds", summary_path, len(summary_df))
    return summary_df


def ensure_prediction_diagnostics(prediction_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    prediction_df = prediction_df.copy()
    required_columns = {"prediction_mean", "prediction_std", target_column}
    missing_columns = required_columns.difference(prediction_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise KeyError(f"Prediction diagnostics require columns: {missing_list}")

    prediction_df["residual"] = prediction_df["prediction_mean"] - prediction_df[target_column]
    if "abs_error" not in prediction_df.columns:
        prediction_df["abs_error"] = np.abs(prediction_df["residual"])
    if "sq_error" not in prediction_df.columns:
        prediction_df["sq_error"] = prediction_df["residual"] ** 2

    mean = prediction_df["prediction_mean"].to_numpy()
    std = prediction_df["prediction_std"].to_numpy()
    for coverage, suffix in ((0.50, "50"), (0.90, "90"), (0.95, "95")):
        lower_col = f"lower_{suffix}"
        upper_col = f"upper_{suffix}"
        width_col = f"width_{suffix}"
        hit_col = f"hit_{suffix}"
        if lower_col not in prediction_df.columns or upper_col not in prediction_df.columns:
            lower, upper = interval_bounds(mean, std, coverage)
            if lower_col not in prediction_df.columns:
                prediction_df[lower_col] = lower
            if upper_col not in prediction_df.columns:
                prediction_df[upper_col] = upper
        if width_col not in prediction_df.columns:
            prediction_df[width_col] = prediction_df[upper_col] - prediction_df[lower_col]
        if hit_col not in prediction_df.columns:
            prediction_df[hit_col] = (
                (prediction_df[target_column] >= prediction_df[lower_col])
                & (prediction_df[target_column] <= prediction_df[upper_col])
            ).astype(int)
    return prediction_df


def load_prediction_diagnostics(summary_df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, row in summary_df.iterrows():
        prediction_path = Path(str(row["prediction_file"]))
        if not prediction_path.exists():
            logger.warning("Prediction file missing for target=%s outer_test=%s: %s", row["target_column"], row["outer_test_key"], prediction_path)
            continue
        prediction_df = ensure_prediction_diagnostics(
            pd.read_csv(prediction_path),
            target_column=str(row["target_column"]),
        )
        prediction_df["run_tag"] = row["run_tag"]
        prediction_df["target_column"] = row["target_column"]
        prediction_df["outer_test_key"] = row["outer_test_key"]
        prediction_df["outer_test_label"] = row["outer_test_label"]
        prediction_df["feature_normalization"] = row["feature_normalization"]
        prediction_df["target_normalization"] = row["target_normalization"]
        prediction_df["prediction_file"] = str(prediction_path)
        frames.append(prediction_df)
    if not frames:
        return pd.DataFrame()
    diagnostics_df = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded %d prediction rows across %d outer folds",
        len(diagnostics_df),
        diagnostics_df[["target_column", "outer_test_key"]].drop_duplicates().shape[0],
    )
    return diagnostics_df


def load_saved_histories(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, summary_row in summary_df.iterrows():
        history_path_raw = summary_row.get("final_history_file")
        if pd.isna(history_path_raw) or not history_path_raw:
            continue
        history_path = Path(str(history_path_raw))
        if not history_path.exists():
            continue
        history_df = pd.read_csv(history_path)
        history_df["stage"] = "final"
        history_df["run_tag"] = summary_row["run_tag"]
        history_df["target_column"] = summary_row["target_column"]
        history_df["outer_test_key"] = summary_row["outer_test_key"]
        history_df["outer_test_label"] = summary_row["outer_test_label"]
        history_df["history_file"] = str(history_path)
        rows.append(history_df)
    if not rows:
        return pd.DataFrame()
    history_df = pd.concat(rows, ignore_index=True)
    logger.info("Loaded %d epoch records from saved history CSVs", len(history_df))
    return history_df


def build_coverage_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in summary_df.iterrows():
        for nominal in (50, 90, 95):
            empirical = float(row[f"coverage_{nominal}"])
            width = float(row[f"width_{nominal}"])
            nominal_fraction = nominal / 100.0
            rows.append(
                {
                    "run_tag": row["run_tag"],
                    "target_column": row["target_column"],
                    "outer_test_key": row["outer_test_key"],
                    "outer_test_label": row["outer_test_label"],
                    "nominal_coverage": nominal_fraction,
                    "nominal_label": f"{nominal}%",
                    "empirical_coverage": empirical,
                    "coverage_gap": empirical - nominal_fraction,
                    "interval_width": width,
                }
            )
    return pd.DataFrame(rows)


def build_cycle_bin_table(prediction_df: pd.DataFrame, cycle_bins: int) -> pd.DataFrame:
    if prediction_df.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for (_, target_column, outer_test_key), fold_df in prediction_df.groupby(
        ["run_tag", "target_column", "outer_test_key"],
        sort=False,
    ):
        ordered = fold_df.copy().sort_values("Cycle")
        ordered["cycle_bin_index"] = pd.qcut(
            ordered["Cycle"].rank(method="first"),
            q=cycle_bins,
            labels=False,
            duplicates="drop",
        )
        grouped = ordered.groupby("cycle_bin_index", as_index=False).agg(
            run_tag=("run_tag", "first"),
            target_column=("target_column", "first"),
            outer_test_key=("outer_test_key", "first"),
            outer_test_label=("outer_test_label", "first"),
            cycle_start=("Cycle", "min"),
            cycle_end=("Cycle", "max"),
            n_points=("Cycle", "size"),
            mean_abs_error=("abs_error", "mean"),
            rmse=("sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            mean_prediction_std=("prediction_std", "mean"),
            coverage_50=("hit_50", "mean"),
            coverage_90=("hit_90", "mean"),
            coverage_95=("hit_95", "mean"),
            width_50=("width_50", "mean"),
            width_90=("width_90", "mean"),
            width_95=("width_95", "mean"),
        )
        grouped["cycle_bin_label"] = grouped.apply(
            lambda row: f"{int(row['cycle_start'])}-{int(row['cycle_end'])}",
            axis=1,
        )
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def build_calibration_summary(coverage_df: pd.DataFrame) -> pd.DataFrame:
    if coverage_df.empty:
        return coverage_df
    calibration = (
        coverage_df.groupby(["run_tag", "target_column", "nominal_coverage", "nominal_label"], as_index=False)
        .agg(
            empirical_coverage=("empirical_coverage", "mean"),
            empirical_coverage_std=("empirical_coverage", "std"),
            interval_width=("interval_width", "mean"),
            mean_gap=("coverage_gap", "mean"),
        )
    )
    calibration["absolute_gap"] = calibration["mean_gap"].abs()
    return calibration


def build_best_epoch_table(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return history_df

    rows: list[dict[str, object]] = []
    for (run_tag, target_column, outer_test_key), stage_df in history_df.groupby(
        ["run_tag", "target_column", "outer_test_key"],
        dropna=False,
    ):
        ordered = stage_df.sort_values("epoch")
        if ordered["test_nll"].notna().any():
            best_row = ordered.loc[ordered["test_nll"].astype(float).idxmin()]
            selection_metric = "test_nll"
        else:
            best_row = ordered.iloc[-1]
            selection_metric = "train_loss"

        rows.append(
            {
                "run_tag": run_tag,
                "target_column": target_column,
                "outer_test_key": outer_test_key,
                "outer_test_label": ordered["outer_test_label"].iloc[0],
                "best_epoch_by_history": int(best_row["epoch"]),
                "selection_metric": selection_metric,
                "best_train_loss": float(best_row["train_loss"]),
                "best_test_nll": None if pd.isna(best_row.get("test_nll")) else float(best_row["test_nll"]),
                "epochs_recorded": int(ordered["epoch"].max()),
            }
        )
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved analysis table to %s", path)


def run(output_dir: Path, cycle_bins: int = 10) -> None:
    tables_dir, figures_dir = ensure_analysis_directories(output_dir)
    setup_logging(
        log_dir=output_dir / "analysis" / "logs",
        filename_prefix="gp_analysis_tables_",
        console_output=True,
        level=logging.INFO,
    )
    logger.info("Building nested GP analysis tables from %s", output_dir)
    logger.info("Plots will later be written to %s", figures_dir)

    summary_df = load_summary(output_dir)
    history_df = load_saved_histories(summary_df)
    predictions_df = load_prediction_diagnostics(summary_df)
    coverage_df = build_coverage_table(summary_df)
    calibration_df = build_calibration_summary(coverage_df)
    cycle_bin_df = build_cycle_bin_table(predictions_df, cycle_bins)
    best_epoch_df = build_best_epoch_table(history_df)

    save_table(summary_df, tables_dir / "fold_metrics.csv")
    save_table(coverage_df, tables_dir / "coverage_long.csv")
    save_table(calibration_df, tables_dir / "calibration_summary.csv")
    save_table(predictions_df, tables_dir / "prediction_diagnostics.csv")
    save_table(cycle_bin_df, tables_dir / "cycle_bin_metrics.csv")
    save_table(history_df, tables_dir / "training_history_long.csv")
    save_table(best_epoch_df, tables_dir / "best_epoch_summary.csv")

    logger.info("Finished building nested GP analysis tables")


def main() -> None:
    args = parse_args()
    run(
        output_dir=Path(args.output_dir).resolve(),
        cycle_bins=args.cycle_bins,
    )


if __name__ == "__main__":
    main()
