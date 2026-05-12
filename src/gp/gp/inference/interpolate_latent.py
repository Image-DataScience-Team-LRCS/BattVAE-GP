from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import gpytorch
import torch
from src.gp.training.loss import interval_bounds

REPO_ROOT = Path(__file__).resolve().parents[3]
if __package__ is None or __package__ == "":
    sys.path.append(str(REPO_ROOT))

from src.gp.data.data_loader import ArrayScaler, inverse_transform_array, transform_array
from src.gp.data.data_preparation import load_experiment_config
from src.gp.models.gp_model import SparseGaussianProcessRegressor
from src.gp.models.gp_model_2d import SparseGaussianProcessRegressor2D
from src.common.logger.logging import get_logger, setup_logging


logger = get_logger(__name__)


def _format_c_rate_tag(c_rate: Any) -> str:
    value = float(c_rate)
    return f"{value:g}"


def _parse_c_rate_from_label(value: Any) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value))
    if match is None:
        return None
    return float(match.group(1))


def _merge_config(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    if not override:
        return base

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _load_interpolation_dataset_specs(config: dict[str, Any], interpolation_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_keys = [str(key) for key in interpolation_cfg.get("interpolation_datasets", [])]
    if not dataset_keys:
        return []

    datasets_file_path = config.get("_datasets_file_path")
    if not datasets_file_path:
        raise ValueError("GP config is missing _datasets_file_path for interpolation dataset lookup.")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to resolve interpolation datasets.") from exc

    datasets_config = yaml.safe_load(Path(datasets_file_path).read_text(encoding="utf-8")) or {}
    interpolation_datasets = datasets_config.get("GP_INTERPOLATION_DATASETS", {}) or {}

    resolved_specs: list[dict[str, Any]] = []
    for dataset_key in dataset_keys:
        dataset_meta = interpolation_datasets.get(dataset_key)
        if not isinstance(dataset_meta, dict):
            raise ValueError(f"Interpolation dataset '{dataset_key}' was not found in GP_INTERPOLATION_DATASETS.")

        interpolation_latent_path = dataset_meta.get("interpolation_latent_path")
        if not interpolation_latent_path:
            raise ValueError(
                f"Interpolation dataset '{dataset_key}' is missing interpolation_latent_path in datasets.yaml."
            )

        resolved_specs.append(
            {
                "dataset_key": dataset_key,
                "charging_rate": float(dataset_meta["charging_rate"]),
                "cycle_start": int(dataset_meta.get("cycle_min", 1)),
                "cycle_end": int(dataset_meta["cycle_max"]),
                "output_csv": Path(str(interpolation_latent_path)),
            }
        )
    return resolved_specs


def _resolve_output_dir(config: dict[str, Any]) -> Path:
    return Path(str(config["PATHS"]["gp_output_dir"]))


def _interpolation_config(config: dict[str, Any]) -> dict[str, Any]:
    interpolation_cfg = dict(config.get("interpolation", {}) or {})
    if not bool(interpolation_cfg.get("enabled", False)):
        raise ValueError("Interpolation is disabled in the GP config.")
    if not bool(interpolation_cfg.get("use_saved_deployment_models", True)):
        raise ValueError("Interpolation currently requires use_saved_deployment_models=true.")
    return interpolation_cfg


def _build_query_frame(interpolation_cfg: dict[str, Any]) -> pd.DataFrame:
    cycle_start = int(interpolation_cfg["cycle_start"])
    cycle_end = int(interpolation_cfg["cycle_end"])
    if cycle_end < cycle_start:
        raise ValueError("interpolation.cycle_end must be greater than or equal to interpolation.cycle_start")

    cycles = np.arange(cycle_start, cycle_end + 1, dtype=int)
    c_rate = float(interpolation_cfg["c_rate"])
    return pd.DataFrame(
        {
            "Cycle": cycles,
            "c_rate": np.full(len(cycles), c_rate, dtype=float),
        }
    )


def _build_query_frame_from_dataset(dataset_spec: dict[str, Any]) -> pd.DataFrame:
    cycles = np.arange(dataset_spec["cycle_start"], dataset_spec["cycle_end"] + 1, dtype=int)
    return pd.DataFrame(
        {
            "Cycle": cycles,
            "c_rate": np.full(len(cycles), dataset_spec["charging_rate"], dtype=float),
        }
    )


def _scaler_from_metadata(payload: dict[str, Any]) -> ArrayScaler:
    return ArrayScaler(
        method=str(payload["method"]),
        center=np.asarray(payload["center"], dtype=float),
        scale=np.asarray(payload["scale"], dtype=float),
        column_methods=payload.get("column_methods"),
        column_names=payload.get("column_names"),
    )


def _load_model_registry(config: dict[str, Any], interpolation_cfg: dict[str, Any]) -> pd.DataFrame:
    registry_path = Path(str(config["PATHS"]["gp_model_summary_csv"]))
    if not registry_path.exists():
        raise FileNotFoundError(f"Deployment model summary not found: {registry_path}")
    registry_df = pd.read_csv(registry_path)
    if registry_df.empty:
        raise ValueError(f"No deployment models found in {registry_path}")
    return registry_df


def _select_registry_rows(
    registry_df: pd.DataFrame,
    dataset_spec: dict[str, Any],
    interpolation_cfg: dict[str, Any],
    *,
    model_type: str | None = None,
    target_column: str | None = None,
) -> pd.DataFrame:
    selected = registry_df.copy()
    if model_type is not None:
        selected = selected.loc[selected["model_type"].astype(str).str.lower().eq(model_type.lower())]
    if target_column is not None and "target_column" in selected.columns:
        selected = selected.loc[selected["target_column"].astype(str) == str(target_column)]
    if selected.empty:
        raise ValueError("No deployment models matched the interpolation selection criteria.")

    configured_outer_test_key = interpolation_cfg.get("deployment_outer_test_key")
    if configured_outer_test_key:
        keyed = selected.loc[selected["outer_test_key"].astype(str) == str(configured_outer_test_key)]
        if keyed.empty:
            available_keys = ", ".join(sorted(selected["outer_test_key"].astype(str).unique().tolist()))
            raise ValueError(
                f"Configured interpolation deployment_outer_test_key='{configured_outer_test_key}' was not found. "
                f"Available keys: {available_keys}"
            )
        return keyed

    if len(selected) == 1:
        return selected

    if "outer_test_label" not in selected.columns:
        raise ValueError("Multiple deployment models are available, but outer_test_label is missing for selection.")

    dataset_c_rate = float(dataset_spec["charging_rate"])
    candidate_rates = selected["outer_test_label"].map(_parse_c_rate_from_label)
    if candidate_rates.isna().any():
        available_labels = ", ".join(selected["outer_test_label"].astype(str).unique().tolist())
        raise ValueError(
            "Multiple deployment models are available, but at least one outer_test_label "
            f"does not contain a numeric C-rate. Available labels: {available_labels}"
        )

    selected = selected.assign(_selection_distance=(candidate_rates.astype(float) - dataset_c_rate).abs())
    min_distance = float(selected["_selection_distance"].min())
    closest = selected.loc[selected["_selection_distance"] == min_distance].copy()
    if len(closest) > 1:
        candidate_labels = ", ".join(closest["outer_test_label"].astype(str).unique().tolist())
        raise ValueError(
            f"Interpolation dataset '{dataset_spec['dataset_key']}' matched multiple deployment models at the same "
            f"C-rate distance ({min_distance:.4f}). Candidates: {candidate_labels}. "
            "Set INTERPOLATION.deployment_outer_test_key to disambiguate."
        )

    logger.info(
        "Selected deployment model for interpolation dataset=%s at c_rate=%.3f using held-out model=%s (%s)",
        dataset_spec["dataset_key"],
        dataset_c_rate,
        str(closest.iloc[0].get("outer_test_key", "unknown")),
        str(closest.iloc[0].get("outer_test_label", "unknown")),
    )
    return closest.drop(columns="_selection_distance")


def _resolve_artifact_path(raw_path: str, config: dict[str, Any]) -> Path:
    path = Path(str(raw_path))
    if path.exists():
        return path

    marker = "artifacts/"
    raw_text = str(raw_path)
    marker_index = raw_text.find(marker)
    if marker_index != -1:
        candidate = Path(raw_text[marker_index:])
        if candidate.exists():
            return candidate

    output_dir = _resolve_output_dir(config)
    for artifact_subdir in ("deployment_models", "models", "histories", "predictions"):
        candidate = output_dir / artifact_subdir / path.name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Artifact not found: {raw_path}")


def _predict_target(
    model: SparseGaussianProcessRegressor,
    feature_scaler: ArrayScaler,
    target_scaler: ArrayScaler,
    feature_columns: list[str],
    query_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    x_query = transform_array(query_df[feature_columns].to_numpy(), feature_scaler)
    posterior = model.predict(x_query, include_noise=True)
    mean = inverse_transform_array(posterior.mean.reshape(-1, 1), target_scaler).reshape(-1)
    variance = posterior.variance * float(target_scaler.scale[0] ** 2)
    std = np.sqrt(np.maximum(variance, 1e-12))
    return mean, std


def _predict_targets_2d(
    model: SparseGaussianProcessRegressor2D,
    feature_scaler: ArrayScaler,
    target_scalers: dict[str, ArrayScaler],
    feature_columns: list[str],
    target_columns: list[str],
    query_df: pd.DataFrame,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[tuple[str, str], np.ndarray]]:
    x_query = transform_array(query_df[feature_columns].to_numpy(), feature_scaler)
    scaled_mean, scaled_variance, scaled_covariance = _predict_2d_distribution_with_covariance(
        model,
        x_query,
        include_noise=True,
    )
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for target_index, target_column in enumerate(target_columns):
        scaler = target_scalers[target_column]
        target_mean = inverse_transform_array(
            scaled_mean[:, target_index].reshape(-1, 1),
            scaler,
        ).reshape(-1)
        target_variance = scaled_variance[:, target_index] * float(scaler.scale[0] ** 2)
        std = np.sqrt(np.maximum(target_variance, 1e-12))
        predictions[target_column] = (target_mean, std)

    covariances: dict[tuple[str, str], np.ndarray] = {}
    for first_index, first_column in enumerate(target_columns):
        for second_index in range(first_index + 1, len(target_columns)):
            second_column = target_columns[second_index]
            first_scaler = target_scalers[first_column]
            second_scaler = target_scalers[second_column]
            covariances[(first_column, second_column)] = (
                scaled_covariance[:, first_index, second_index]
                * float(first_scaler.scale[0])
                * float(second_scaler.scale[0])
            )
    return predictions, covariances


def _predict_2d_distribution_with_covariance(
    model: SparseGaussianProcessRegressor2D,
    x: np.ndarray,
    include_noise: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model.model_ is None or model.likelihood_ is None or model.output_dim_ is None:
        raise RuntimeError("Model must be fitted before prediction.")

    x_array = np.asarray(x, dtype=np.float32)
    batch_size = min(int(model.training_config.get("eval_batch_size", 4096)), 512)
    output_dim = int(model.output_dim_)

    mean_parts: list[np.ndarray] = []
    variance_parts: list[np.ndarray] = []
    covariance_parts: list[np.ndarray] = []

    model.model_.eval()
    model.likelihood_.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for start in range(0, len(x_array), batch_size):
            x_batch = model._to_tensor(x_array[start : start + batch_size]).to(model.device)
            posterior = model.model_(x_batch)
            if include_noise:
                posterior = model.likelihood_(posterior)

            batch_mean = posterior.mean.detach().cpu().numpy()
            batch_variance = posterior.variance.detach().cpu().numpy()
            covariance_matrix = posterior.covariance_matrix.detach().cpu().numpy()
            batch_len = batch_mean.shape[0]
            batch_covariance = np.empty((batch_len, output_dim, output_dim), dtype=np.float64)
            for row_index in range(batch_len):
                matrix_offset = row_index * output_dim
                batch_covariance[row_index] = covariance_matrix[
                    matrix_offset : matrix_offset + output_dim,
                    matrix_offset : matrix_offset + output_dim,
                ]

            mean_parts.append(batch_mean)
            variance_parts.append(batch_variance)
            covariance_parts.append(batch_covariance)

    mean = np.concatenate(mean_parts, axis=0)
    variance = np.maximum(np.concatenate(variance_parts, axis=0), 1e-12)
    covariance = np.concatenate(covariance_parts, axis=0)
    return mean, variance, covariance


def run(config_path: str | Path, config_override: dict[str, Any] | None = None) -> Path:
    config = _merge_config(load_experiment_config(config_path), config_override)
    interpolation_cfg = _interpolation_config(config)
    output_dir = _resolve_output_dir(config)
    interpolation_dir = output_dir / "interpolation"
    interpolation_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        log_dir=interpolation_dir / "logs",
        filename_prefix="gp_latent_interpolation_",
        console_output=True,
        level=logging.INFO,
    )

    registry_df = _load_model_registry(config, interpolation_cfg)
    target_columns = [str(column) for column in interpolation_cfg.get("target_columns", ["z1", "z2"])]
    dataset_specs = _load_interpolation_dataset_specs(config, interpolation_cfg)
    if not dataset_specs:
        raise ValueError("No GP interpolation datasets were configured in INTERPOLATION.interpolation_datasets.")

    is_joint_2d = (
        "model_type" in registry_df.columns
        and registry_df["model_type"].astype(str).str.lower().eq("sparse_gp_2d").any()
    )

    last_output_csv: Path | None = None
    for dataset_spec in dataset_specs:
        query_df = _build_query_frame_from_dataset(dataset_spec)
        predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        covariances: dict[tuple[str, str], np.ndarray] = {}
        if is_joint_2d:
            joint_rows = _select_registry_rows(
                registry_df,
                dataset_spec,
                interpolation_cfg,
                model_type="sparse_gp_2d",
            )
            model_path = _resolve_artifact_path(str(joint_rows.iloc[0]["model_file"]), config)
            model, metadata = SparseGaussianProcessRegressor2D.load_artifact(model_path)
            feature_columns = [str(column) for column in metadata["feature_columns"]]
            feature_scaler = _scaler_from_metadata(dict(metadata["feature_scaler"]))
            target_scalers = {
                str(target_column): _scaler_from_metadata(dict(payload))
                for target_column, payload in dict(metadata["target_scalers"]).items()
            }
            predictions, covariances = _predict_targets_2d(
                model=model,
                feature_scaler=feature_scaler,
                target_scalers=target_scalers,
                feature_columns=feature_columns,
                target_columns=target_columns,
                query_df=query_df,
            )
            logger.info(
                "Generated joint 2D interpolation dataset=%s with model=%s rows=%d",
                dataset_spec["dataset_key"],
                model_path,
                len(query_df),
            )
        else:
            for target_column in target_columns:
                row = _select_registry_rows(
                    registry_df,
                    dataset_spec,
                    interpolation_cfg,
                    target_column=target_column,
                )
                if row.empty:
                    raise ValueError(f"No deployment model is available for target '{target_column}'.")
                model_path = _resolve_artifact_path(str(row.iloc[0]["model_file"]), config)
                model, metadata = SparseGaussianProcessRegressor.load_artifact(model_path)
                feature_columns = [str(column) for column in metadata["feature_columns"]]
                feature_scaler = _scaler_from_metadata(dict(metadata["feature_scaler"]))
                target_scaler = _scaler_from_metadata(dict(metadata["target_scaler"]))
                predictions[target_column] = _predict_target(
                    model=model,
                    feature_scaler=feature_scaler,
                    target_scaler=target_scaler,
                    feature_columns=feature_columns,
                    query_df=query_df,
                )
                logger.info(
                    "Generated interpolation dataset=%s target=%s with model=%s rows=%d",
                    dataset_spec["dataset_key"],
                    target_column,
                    model_path,
                    len(query_df),
                )

        result_df = pd.DataFrame(
            {
                "C-rate": query_df["c_rate"].astype(float),
                "Cycle": query_df["Cycle"].astype(int),
            }
        )
        for target_column in target_columns:
            mean, std = predictions[target_column]
            lower_90, upper_90 = interval_bounds(mean, std, 0.90)
            lower_95, upper_95 = interval_bounds(mean, std, 0.95)
            result_df[target_column] = mean
            result_df[f"{target_column}_std"] = std
            result_df[f"{target_column}_lower_90"] = lower_90
            result_df[f"{target_column}_upper_90"] = upper_90
            result_df[f"{target_column}_lower_95"] = lower_95
            result_df[f"{target_column}_upper_95"] = upper_95

        for (first_column, second_column), covariance in covariances.items():
            result_df[f"{first_column}_{second_column}_cov"] = covariance

        output_csv = Path(dataset_spec["output_csv"])
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv, index=False)
        logger.info("Saved latent interpolation CSV for %s to %s", dataset_spec["dataset_key"], output_csv)
        last_output_csv = output_csv

    if last_output_csv is None:
        raise RuntimeError("Interpolation produced no output CSV.")
    return last_output_csv
