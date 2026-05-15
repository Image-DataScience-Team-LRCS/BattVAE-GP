from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.logger.logging import get_logger
from src.gp.data.data_loader import (
    build_nested_fold_data,
    build_refit_fold_data,
    fit_feature_scaler,
    fit_scaler,
    inverse_transform_array,
    transform_array,
)
from src.gp.inference.inference import evaluate_prediction_frame, predict_dataframe
from src.gp.models.gp_model_2d import SparseGaussianProcessRegressor2D
from src.gp.training.loss import interval_bounds, summarize_uncertainty
from src.gp.training.train import select_and_fit_nested_gp


logger = get_logger(__name__)


def _target_columns(config: dict[str, Any]) -> list[str]:
    configured = config["features"].get("target_columns", [])
    if configured:
        return [str(column) for column in configured]
    return [str(config["features"]["target_column"])]


def _validation_keys(frame: pd.DataFrame, config: dict[str, Any], outer_test_key: str) -> list[str]:
    group_column = str(config["features"]["group_column"])
    return [
        str(dataset_key)
        for dataset_key in frame[group_column].drop_duplicates().tolist()
        if str(dataset_key) != outer_test_key
    ]


def _default_target_norm(config: dict[str, Any], target_column: str) -> str:
    target_methods = {
        str(column): str(method)
        for column, method in dict(
            config.get("normalization", {}).get("target_column_methods", {}) or {}
        ).items()
    }
    if target_column in target_methods:
        return target_methods[target_column]
    return str(config["normalization"].get("target_method", "standard"))


def _candidate_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    model_cfg = dict(config["model"])
    params = {
        "kernel": [str(model_cfg["kernel"])],
        "lengthscale": model_cfg["lengthscale"] if isinstance(model_cfg["lengthscale"], list) else [model_cfg["lengthscale"]],
        "variance": model_cfg["variance"] if isinstance(model_cfg["variance"], list) else [model_cfg["variance"]],
        "noise_variance": model_cfg["noise_variance"] if isinstance(model_cfg["noise_variance"], list) else [model_cfg["noise_variance"]],
        "rq_alpha": model_cfg["rq_alpha"] if isinstance(model_cfg["rq_alpha"], list) else [model_cfg["rq_alpha"]],
    }
    search_cfg = dict(config.get("search", {}) or {})
    if bool(search_cfg.get("enabled", False)):
        params["lengthscale"] = search_cfg.get("lengthscale_candidates", params["lengthscale"])
        params["variance"] = search_cfg.get("variance_candidates", params["variance"])
        params["noise_variance"] = search_cfg.get("noise_variance_candidates", params["noise_variance"])
        params["rq_alpha"] = search_cfg.get("rq_alpha_candidates", params["rq_alpha"])

    grid: list[dict[str, Any]] = []
    for kernel in params["kernel"]:
        for lengthscale in params["lengthscale"]:
            for variance in params["variance"]:
                for noise_variance in params["noise_variance"]:
                    for rq_alpha in params["rq_alpha"]:
                        grid.append(
                            {
                                "kernel": str(kernel),
                                "lengthscale": lengthscale,
                                "variance": float(variance),
                                "noise_variance": float(noise_variance),
                                "rq_alpha": float(rq_alpha),
                            }
                        )
    return grid


def _save_history(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Finished saving history to %s", path)
    return path


def _artifact_metadata(config: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(extra)
    metadata["config"] = dict(config)
    metadata["checkpoint_metadata"] = {
        "experiment_name": str(config["experiment"]["name"]),
        "model_type": str(config["experiment"].get("model_type", "gp")),
        "random_seed": int(config["experiment"].get("random_seed", 42)),
    }
    return metadata


def _run_tag(model_type: str, outer_test_key: str, target_column: str) -> str:
    return f"{model_type}__test={outer_test_key}__target={target_column}"


def _training_config_with_early_stopping(config: dict[str, Any]) -> dict[str, Any]:
    training_cfg = dict(config.get("training", {}) or {})
    training_cfg["early_stopping"] = dict(config.get("early_stopping", {}) or {})
    return training_cfg


def _feature_scaler_metadata(feature_scaler: Any) -> dict[str, Any]:
    return {
        "method": feature_scaler.method,
        "center": feature_scaler.center.tolist(),
        "scale": feature_scaler.scale.tolist(),
        "column_methods": feature_scaler.column_methods,
        "column_names": feature_scaler.column_names,
        "fourier_enabled": feature_scaler.fourier_enabled,
        "fourier_num_frequencies": feature_scaler.fourier_num_frequencies,
        "fourier_max_frequency": feature_scaler.fourier_max_frequency,
        "fourier_include_original": feature_scaler.fourier_include_original,
        "fourier_frequency_scale": feature_scaler.fourier_frequency_scale,
    }


def _save_inner_fold_rows(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Finished saving inner-fold metrics to %s", path)
    return path


def _summarize_2d_metrics(
    prediction_df: pd.DataFrame,
    target_column: str,
    true_values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, float]:
    prediction_df["prediction_mean"] = mean
    prediction_df["prediction_std"] = std
    prediction_df["abs_error"] = np.abs(true_values - mean)
    prediction_df["sq_error"] = (true_values - mean) ** 2
    for coverage in (0.50, 0.90, 0.95):
        lower, upper = interval_bounds(mean, std, coverage)
        suffix = str(int(coverage * 100))
        prediction_df[f"lower_{suffix}"] = lower
        prediction_df[f"upper_{suffix}"] = upper
        prediction_df[f"width_{suffix}"] = upper - lower
        prediction_df[f"hit_{suffix}"] = (
            (prediction_df[target_column] >= lower) & (prediction_df[target_column] <= upper)
        ).astype(int)
    return summarize_uncertainty(y_true=true_values, mean=mean, std=std)


def _fit_nested_2d_holdout(
    frame: pd.DataFrame,
    config: dict[str, Any],
    test_key: str,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feature_columns = [str(column) for column in config["features"]["input_columns"]]
    target_columns = _target_columns(config)
    if len(target_columns) != 2:
        raise ValueError("sparse_gp_2d requires exactly two target columns.")

    group_column = str(config["features"]["group_column"])
    cycle_column = str(config["features"]["cycle_column"])
    normalization_cfg = dict(config.get("normalization", {}) or {})
    feature_method = str(normalization_cfg.get("feature_method", "standard")).lower()
    feature_column_methods = normalization_cfg.get("feature_column_methods", {})
    fourier_config = dict(config.get("features", {}).get("fourier_features", {}) or {})
    validation_keys = _validation_keys(frame, config, test_key)

    logger.info(
        "Starting nested 2D GP selection for outer_test=%s with %d inner validation dataset(s)",
        test_key,
        len(validation_keys),
    )

    candidate_rows_all: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_score = float("inf")
    best_epochs: list[int] = []
    selection_metric = str(config.get("selection", {}).get("metric", "nll")).lower()

    for candidate_index, params in enumerate(_candidate_grid(config), start=1):
        candidate_scores: list[float] = []
        candidate_best_epochs: list[int] = []
        candidate_rows: list[dict[str, Any]] = []

        for validation_key in validation_keys:
            train_df = frame.loc[~frame[group_column].isin([test_key, validation_key])].copy()
            val_df = frame.loc[frame[group_column] == validation_key].copy()
            if train_df.empty or val_df.empty:
                continue

            feature_scaler = fit_feature_scaler(
                frame=train_df,
                feature_columns=feature_columns,
                default_method=feature_method,
                column_methods=feature_column_methods,
                fourier_config=fourier_config,
            )

            target_scalers: dict[str, Any] = {}
            y_train_cols: list[np.ndarray] = []
            y_val_true: dict[str, np.ndarray] = {}
            y_val_scaled_cols: list[np.ndarray] = []
            for target_column in target_columns:
                scaler = fit_scaler(train_df[[target_column]].to_numpy(), _default_target_norm(config, target_column))
                target_scalers[target_column] = scaler
                y_train_cols.append(transform_array(train_df[[target_column]].to_numpy(), scaler).reshape(-1))
                y_val_scaled_cols.append(transform_array(val_df[[target_column]].to_numpy(), scaler).reshape(-1))
                y_val_true[target_column] = val_df[target_column].to_numpy()

            x_train = transform_array(train_df[feature_columns].to_numpy(), feature_scaler)
            x_val = transform_array(val_df[feature_columns].to_numpy(), feature_scaler)
            y_train = np.column_stack(y_train_cols)
            y_val = np.column_stack(y_val_scaled_cols)

            model_cfg = dict(config["model"])
            model = SparseGaussianProcessRegressor2D(
                kernel=str(params["kernel"]),
                lengthscale=params["lengthscale"],
                variance=float(params["variance"]),
                noise_variance=float(params["noise_variance"]),
                rq_alpha=float(params["rq_alpha"]),
                inducing_points=int(model_cfg["inducing_points"]),
                inducing_method=str(model_cfg["inducing_method"]),
                jitter=float(model_cfg["jitter"]),
                random_seed=int(config["experiment"].get("random_seed", 42)),
                training_config=_training_config_with_early_stopping(config),
                runtime_config=dict(config.get("runtime", {}) or {}),
            )
            model.fit(
                x_train,
                y_train,
                groups=train_df[group_column].to_numpy(),
                cycles=train_df[cycle_column].to_numpy(dtype=float),
                val_x=x_val,
                val_y=y_val,
                run_name=f"inner:test={test_key}:val={validation_key}:candidate={candidate_index}",
            )

            posterior = model.predict(x_val, include_noise=True)
            per_target_metrics: dict[str, dict[str, float]] = {}
            for target_index, target_column in enumerate(target_columns):
                scaler = target_scalers[target_column]
                mean = inverse_transform_array(posterior.mean[:, target_index].reshape(-1, 1), scaler).reshape(-1)
                std = np.sqrt(np.maximum(posterior.variance[:, target_index], 1e-12) * float(scaler.scale[0] ** 2))
                metric_values = summarize_uncertainty(y_true=y_val_true[target_column], mean=mean, std=std)
                per_target_metrics[target_column] = metric_values

            score = float(np.mean([metrics[selection_metric] for metrics in per_target_metrics.values()]))
            candidate_scores.append(score)
            candidate_best_epochs.append(
                0 if model.training_summary_ is None else int(model.training_summary_.best_epoch)
            )
            row: dict[str, Any] = {
                "outer_test_key": test_key,
                "outer_test_label": str(frame.loc[frame[group_column] == test_key, "crate_label"].iloc[0]),
                "validation_key": validation_key,
                "validation_label": str(val_df["crate_label"].iloc[0]),
                "selection_metric": selection_metric,
                "selection_score": score,
                "best_epoch": None if model.training_summary_ is None else model.training_summary_.best_epoch,
                "epochs_completed": None if model.training_summary_ is None else model.training_summary_.epochs_completed,
            }
            for target_column, metrics in per_target_metrics.items():
                for metric_name, metric_value in metrics.items():
                    row[f"val_{metric_name}_{target_column}"] = float(metric_value)
            row.update({f"candidate_{key}": value for key, value in params.items()})
            candidate_rows.append(row)

        candidate_rows_all.extend(candidate_rows)
        if candidate_scores:
            mean_score = float(np.mean(candidate_scores))
            if mean_score < best_score:
                best_score = mean_score
                best_params = dict(params)
                best_epochs = candidate_best_epochs.copy()

    if best_params is None:
        raise RuntimeError(f"No 2D GP candidate could be selected for outer test dataset '{test_key}'.")

    refit_epochs = max(1, int(round(float(np.median(np.asarray(best_epochs, dtype=float))))))
    logger.info(
        "Finished nested 2D model selection outer_test=%s params=%s refit_epochs=%d mean_%s=%.6f",
        test_key,
        best_params,
        refit_epochs,
        selection_metric,
        best_score,
    )

    train_df = frame.loc[frame[group_column] != test_key].copy()
    test_df = frame.loc[frame[group_column] == test_key].copy()
    feature_scaler = fit_feature_scaler(
        frame=train_df,
        feature_columns=feature_columns,
        default_method=feature_method,
        column_methods=feature_column_methods,
        fourier_config=fourier_config,
    )
    target_scalers: dict[str, Any] = {}
    y_train_cols = []
    y_test_true: dict[str, np.ndarray] = {}
    for target_column in target_columns:
        scaler = fit_scaler(train_df[[target_column]].to_numpy(), _default_target_norm(config, target_column))
        target_scalers[target_column] = scaler
        y_train_cols.append(transform_array(train_df[[target_column]].to_numpy(), scaler).reshape(-1))
        y_test_true[target_column] = test_df[target_column].to_numpy()

    x_train = transform_array(train_df[feature_columns].to_numpy(), feature_scaler)
    x_test = transform_array(test_df[feature_columns].to_numpy(), feature_scaler)
    y_train = np.column_stack(y_train_cols)

    model_cfg = dict(config["model"])
    final_model = SparseGaussianProcessRegressor2D(
        kernel=str(best_params["kernel"]),
        lengthscale=best_params["lengthscale"],
        variance=float(best_params["variance"]),
        noise_variance=float(best_params["noise_variance"]),
        rq_alpha=float(best_params["rq_alpha"]),
        inducing_points=int(model_cfg["inducing_points"]),
        inducing_method=str(model_cfg["inducing_method"]),
        jitter=float(model_cfg["jitter"]),
        random_seed=int(config["experiment"].get("random_seed", 42)),
        training_config=_training_config_with_early_stopping(config),
        runtime_config=dict(config.get("runtime", {}) or {}),
    )
    final_model.fit(
        x_train,
        y_train,
        groups=train_df[group_column].to_numpy(),
        cycles=train_df[cycle_column].to_numpy(dtype=float),
        test_x=x_test,
        test_y=None,
        run_name=f"final:test={test_key}:targets={','.join(target_columns)}",
        epochs_override=refit_epochs,
    )

    predictions_dir = output_dir / "predictions"
    histories_dir = output_dir / "histories"
    models_dir = output_dir / "models"
    inner_dir = output_dir / "inner_folds"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    inner_dir.mkdir(parents=True, exist_ok=True)

    history_path = histories_dir / f"history_{test_key}_joint2d.csv"
    _save_history(final_model.training_history_, history_path)
    inner_path = inner_dir / f"inner_{test_key}_joint2d.csv"
    _save_inner_fold_rows(candidate_rows_all, inner_path)

    model_path = models_dir / f"test_{test_key}_joint2d.pt"
    final_model.save_artifact(
        model_path,
        metadata=_artifact_metadata(
            config,
            {
                "outer_test_key": test_key,
                "outer_test_label": str(test_df["crate_label"].iloc[0]),
                "feature_columns": feature_columns,
                "target_columns": target_columns,
                "feature_scaler": _feature_scaler_metadata(feature_scaler),
                "target_scalers": {
                    target_column: {
                        "method": target_scalers[target_column].method,
                        "center": target_scalers[target_column].center.tolist(),
                        "scale": target_scalers[target_column].scale.tolist(),
                    }
                    for target_column in target_columns
                },
            },
        ),
    )

    posterior = final_model.predict(x_test, include_noise=True)
    summary_rows: list[dict[str, Any]] = []
    for target_index, target_column in enumerate(target_columns):
        scaler = target_scalers[target_column]
        mean = inverse_transform_array(posterior.mean[:, target_index].reshape(-1, 1), scaler).reshape(-1)
        std = np.sqrt(np.maximum(posterior.variance[:, target_index], 1e-12) * float(scaler.scale[0] ** 2))
        prediction_df = test_df.copy().reset_index(drop=True)
        metrics = _summarize_2d_metrics(
            prediction_df=prediction_df,
            target_column=target_column,
            true_values=y_test_true[target_column],
            mean=mean,
            std=std,
        )
        prediction_path = predictions_dir / f"test_{test_key}_{target_column}.csv"
        prediction_df.to_csv(prediction_path, index=False)
        run_tag = _run_tag("sparse_gp_2d", test_key, target_column)
        row = {
            "run_tag": run_tag,
            "model_type": "sparse_gp_2d",
            "target_column": target_column,
            "outer_test_key": test_key,
            "outer_test_label": str(test_df["crate_label"].iloc[0]),
            "feature_normalization": feature_method,
            "target_normalization": _default_target_norm(config, target_column),
            "selection_metric": selection_metric,
            "selection_score": best_score,
            "refit_epochs": refit_epochs,
            "prediction_file": str(prediction_path),
            "final_history_file": str(history_path),
            "inner_fold_file": str(inner_path),
            "model_file": str(model_path),
            "epochs_completed": None if final_model.training_summary_ is None else final_model.training_summary_.epochs_completed,
            "best_epoch": None if final_model.training_summary_ is None else final_model.training_summary_.best_epoch,
            "best_test_nll": None if final_model.training_summary_ is None else final_model.training_summary_.best_test_nll,
        }
        row.update(metrics)
        summary_rows.append(row)

    deployment_row = {
        "model_type": "sparse_gp_2d",
        "outer_test_key": test_key,
        "outer_test_label": str(test_df["crate_label"].iloc[0]),
        "model_file": str(model_path),
        "target_columns": json.dumps(target_columns),
    }
    return summary_rows, deployment_row


def run_nested_holdout_training(
    frame: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    outer_test_keys: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    deployment_rows: list[dict[str, Any]] = []
    model_type = str(config.get("model", {}).get("type", "sparse_gp")).lower()

    logger.info(
        "Running nested holdout evaluation for %d outer test dataset(s) with model.type=%s",
        len(outer_test_keys),
        model_type,
    )

    if model_type == "sparse_gp_2d":
        for outer_test_key in outer_test_keys:
            rows, deployment_row = _fit_nested_2d_holdout(
                frame=frame,
                config=config,
                test_key=outer_test_key,
                output_dir=output_dir,
            )
            summary_rows.extend(rows)
            deployment_rows.append(deployment_row)
        return pd.DataFrame(summary_rows), pd.DataFrame(deployment_rows)

    feature_norm = str(config.get("normalization", {}).get("feature_method", "standard"))
    target_columns = _target_columns(config)
    predictions_dir = output_dir / "predictions"
    histories_dir = output_dir / "histories"
    models_dir = output_dir / "models"
    inner_dir = output_dir / "inner_folds"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    inner_dir.mkdir(parents=True, exist_ok=True)

    for outer_test_key in outer_test_keys:
        validation_keys = _validation_keys(frame, config, outer_test_key)
        for target_column in target_columns:
            target_norm = _default_target_norm(config, target_column)
            inner_folds = [
                build_nested_fold_data(
                    frame=frame,
                    config=config,
                    target_column=target_column,
                    test_key=outer_test_key,
                    validation_key=validation_key,
                    feature_method=feature_norm,
                    target_method=target_norm,
                )
                for validation_key in validation_keys
            ]
            refit_fold = build_refit_fold_data(
                frame=frame,
                config=config,
                target_column=target_column,
                test_key=outer_test_key,
                feature_method=feature_norm,
                target_method=target_norm,
            )
            result = select_and_fit_nested_gp(
                inner_folds=inner_folds,
                refit_fold=refit_fold,
                config=config,
                run_context={"outer_test_key": outer_test_key, "target_column": target_column},
            )
            prediction_df = predict_dataframe(result.model, refit_fold, split="test")
            metrics = evaluate_prediction_frame(prediction_df, refit_fold.target_column)

            prediction_path = predictions_dir / f"test_{outer_test_key}_{target_column}.csv"
            prediction_df.to_csv(prediction_path, index=False)
            history_path = histories_dir / f"history_{outer_test_key}_{target_column}.csv"
            _save_history(result.final_history, history_path)
            inner_path = inner_dir / f"inner_{outer_test_key}_{target_column}.csv"
            _save_inner_fold_rows(result.all_inner_fold_rows, inner_path)
            model_path = models_dir / f"test_{outer_test_key}_{target_column}.pt"
            saved_model_path = result.model.save_artifact(
                model_path,
                metadata=_artifact_metadata(
                    config,
                    {
                        "target_column": target_column,
                        "outer_test_key": outer_test_key,
                        "outer_test_label": refit_fold.test_label,
                        "feature_columns": refit_fold.feature_columns,
                        "feature_scaler": _feature_scaler_metadata(refit_fold.feature_scaler),
                        "target_scaler": {
                            "method": refit_fold.target_scaler.method,
                            "center": refit_fold.target_scaler.center.tolist(),
                            "scale": refit_fold.target_scaler.scale.tolist(),
                        },
                        "feature_normalization": feature_norm,
                        "target_normalization": target_norm,
                    },
                ),
            )

            run_tag = _run_tag(model_type, outer_test_key, target_column)
            row = {
                "run_tag": run_tag,
                "model_type": model_type,
                "target_column": target_column,
                "outer_test_key": outer_test_key,
                "outer_test_label": refit_fold.test_label,
                "feature_normalization": feature_norm,
                "target_normalization": target_norm,
                "selection_metric": result.selection_metric,
                "selection_score": result.selection_score,
                "refit_epochs": result.refit_epochs,
                "prediction_file": str(prediction_path),
                "final_history_file": str(history_path),
                "inner_fold_file": str(inner_path),
                "model_file": str(saved_model_path),
            }
            row.update(metrics)
            if result.model.training_summary_ is not None:
                row["epochs_completed"] = result.model.training_summary_.epochs_completed
                row["best_test_nll"] = result.model.training_summary_.best_test_nll
                row["best_epoch"] = result.model.training_summary_.best_epoch
            summary_rows.append(row)
            deployment_rows.append(
                {
                    "target_column": target_column,
                    "outer_test_key": outer_test_key,
                    "outer_test_label": refit_fold.test_label,
                    "model_file": str(saved_model_path),
                    "feature_normalization": feature_norm,
                    "target_normalization": target_norm,
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(deployment_rows)
