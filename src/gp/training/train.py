from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from src.gp.data.data_loader import FoldData, inverse_transform_array
from src.gp.models.gp_model import SparseGaussianProcessRegressor
from src.gp.training.loss import summarize_uncertainty
from src.common.logger.logging import get_logger


logger = get_logger(__name__)


@dataclass
class NestedTrainingResult:
    model: SparseGaussianProcessRegressor
    best_params: dict[str, Any]
    selection_metric: str
    selection_score: float
    refit_epochs: int
    all_inner_fold_rows: list[dict[str, Any]]
    inner_fold_rows: list[dict[str, Any]]
    final_history: list[dict[str, Any]]


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _selection_metric_name(config: dict[str, Any]) -> str:
    selection_cfg = dict(config.get("selection", {}) or {})
    return str(selection_cfg.get("metric", "nll")).lower()


def _refit_epoch_strategy(config: dict[str, Any]) -> str:
    selection_cfg = dict(config.get("selection", {}) or {})
    return str(selection_cfg.get("refit_epoch_strategy", "median")).lower()


def _candidate_grid(config: dict[str, Any], kernel_override: str | None = None) -> list[dict[str, Any]]:
    model_cfg = dict(config["model"])
    if kernel_override is not None:
        model_cfg["kernel"] = kernel_override

    grids = {
        "kernel": [str(model_cfg["kernel"])],
        "lengthscale": _as_list(model_cfg["lengthscale"]),
        "variance": _as_list(model_cfg["variance"]),
        "noise_variance": _as_list(model_cfg["noise_variance"]),
        "rq_alpha": _as_list(model_cfg["rq_alpha"]),
    }
    search_cfg = dict(config.get("search", {}) or {})
    if bool(search_cfg.get("enabled", False)):
        grids["lengthscale"] = _as_list(search_cfg.get("lengthscale_candidates", grids["lengthscale"]))
        grids["variance"] = _as_list(search_cfg.get("variance_candidates", grids["variance"]))
        grids["noise_variance"] = _as_list(
            search_cfg.get("noise_variance_candidates", grids["noise_variance"])
        )
        grids["rq_alpha"] = _as_list(search_cfg.get("rq_alpha_candidates", grids["rq_alpha"]))

    return [
        {
            "kernel": str(kernel),
            "lengthscale": lengthscale,
            "variance": float(variance),
            "noise_variance": float(noise_variance),
            "rq_alpha": float(rq_alpha),
        }
        for kernel, lengthscale, variance, noise_variance, rq_alpha in product(
            grids["kernel"],
            grids["lengthscale"],
            grids["variance"],
            grids["noise_variance"],
            grids["rq_alpha"],
        )
    ]


def _instantiate_model(params: dict[str, Any], config: dict[str, Any]) -> SparseGaussianProcessRegressor:
    model_cfg = dict(config["model"])
    training_cfg = dict(config.get("training", {}) or {})
    training_cfg["early_stopping"] = dict(config.get("early_stopping", {}) or {})
    runtime_cfg = dict(config.get("runtime", {}) or {})
    return SparseGaussianProcessRegressor(
        kernel=str(params["kernel"]),
        lengthscale=params["lengthscale"],
        variance=float(params["variance"]),
        noise_variance=float(params["noise_variance"]),
        rq_alpha=float(params["rq_alpha"]),
        inducing_points=int(model_cfg["inducing_points"]),
        inducing_method=str(model_cfg["inducing_method"]),
        jitter=float(model_cfg["jitter"]),
        random_seed=int(config["experiment"].get("random_seed", 42)),
        training_config=training_cfg,
        runtime_config=runtime_cfg,
    )


def _predict_metrics_on_split(model: SparseGaussianProcessRegressor, fold_data: FoldData, split: str) -> dict[str, float]:
    if split == "val":
        x_data = fold_data.x_val
        split_df = fold_data.val_df
    elif split == "test":
        x_data = fold_data.x_test
        split_df = fold_data.test_df
    else:
        raise ValueError(f"Unsupported split for evaluation: {split}")

    if x_data is None or split_df is None or split_df.empty:
        raise ValueError(f"Split '{split}' is not available for target {fold_data.target_column}.")

    posterior = model.predict(x_data, include_noise=True)
    mean = inverse_transform_array(
        posterior.mean.reshape(-1, 1),
        fold_data.target_scaler,
    ).reshape(-1)
    std = np.sqrt(np.maximum(posterior.variance, 1e-12) * float(fold_data.target_scaler.scale[0] ** 2))
    y_true = split_df[fold_data.target_column].to_numpy()
    return summarize_uncertainty(y_true=y_true, mean=mean, std=std)


def _score_from_metrics(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name not in metrics:
        raise ValueError(f"Selection metric '{metric_name}' is not available. Found: {sorted(metrics)}")
    return float(metrics[metric_name])


def _aggregate_refit_epochs(best_epochs: list[int], strategy: str) -> int:
    if not best_epochs:
        raise ValueError("At least one best epoch is required for refit aggregation.")
    values = np.asarray(best_epochs, dtype=float)
    if strategy == "mean":
        return max(1, int(round(float(np.mean(values)))))
    if strategy == "median":
        return max(1, int(round(float(np.median(values)))))
    raise ValueError(f"Unsupported refit_epoch_strategy: {strategy}")


def _evaluate_candidate_on_inner_fold(
    fold_data: FoldData,
    config: dict[str, Any],
    params: dict[str, Any],
    run_name: str,
) -> tuple[float, dict[str, float], SparseGaussianProcessRegressor]:
    logger.info(
        "⏳ Starting: inner validation round target=%s test=%s validation=%s params=%s",
        fold_data.target_column,
        fold_data.test_key,
        fold_data.validation_key,
        params,
    )
    train_groups = fold_data.train_df[fold_data.group_column].to_numpy()
    train_cycles = fold_data.train_df[fold_data.cycle_column].to_numpy(dtype=float)

    model = _instantiate_model(params, config)
    model.fit(
        fold_data.x_train,
        fold_data.y_train,
        groups=train_groups,
        cycles=train_cycles,
        val_x=fold_data.x_val,
        val_y=fold_data.y_val,
        run_name=run_name,
    )
    val_metrics = _predict_metrics_on_split(model, fold_data, split="val")
    score = _score_from_metrics(val_metrics, _selection_metric_name(config))
    logger.info(
        "✅ Finished: inner validation round target=%s test=%s validation=%s score[%s]=%.6f rmse=%.6f mae=%.6f best_epoch=%s",
        fold_data.target_column,
        fold_data.test_key,
        fold_data.validation_key,
        _selection_metric_name(config),
        score,
        val_metrics["rmse"],
        val_metrics["mae"],
        None if model.training_summary_ is None else model.training_summary_.best_epoch,
    )
    return score, val_metrics, model


def fit_gp_with_params(
    fold_data: FoldData,
    params: dict[str, Any],
    config: dict[str, Any],
    run_name: str,
    epochs_override: int | None = None,
) -> SparseGaussianProcessRegressor:
    train_groups = fold_data.train_df[fold_data.group_column].to_numpy()
    train_cycles = fold_data.train_df[fold_data.cycle_column].to_numpy(dtype=float)
    model = _instantiate_model(params, config)
    model.fit(
        fold_data.x_train,
        fold_data.y_train,
        groups=train_groups,
        cycles=train_cycles,
        val_x=fold_data.x_val,
        val_y=fold_data.y_val,
        test_x=fold_data.x_test,
        test_y=fold_data.y_test,
        run_name=run_name,
        epochs_override=epochs_override,
    )
    return model


def select_and_fit_nested_gp(
    inner_folds: list[FoldData],
    refit_fold: FoldData,
    config: dict[str, Any],
    kernel_override: str | None = None,
    run_context: dict[str, str] | None = None,
) -> NestedTrainingResult:
    if not inner_folds:
        raise ValueError("At least one inner fold is required for nested GP selection.")

    selection_metric = _selection_metric_name(config)
    epoch_strategy = _refit_epoch_strategy(config)
    candidate_params = _candidate_grid(config, kernel_override=kernel_override)

    best_params: dict[str, Any] | None = None
    best_score = float("inf")
    best_epoch_values: list[int] = []
    all_inner_rows: list[dict[str, Any]] = []
    best_inner_rows: list[dict[str, Any]] = []

    context = dict(run_context or {})
    outer_test_key = context.get("outer_test_key", refit_fold.test_key or "unknown")
    target_column = context.get("target_column", refit_fold.target_column)

    for candidate_index, params in enumerate(candidate_params, start=1):
        logger.info(
            "⏳ Starting: candidate evaluation target=%s outer_test=%s candidate=%d/%d params=%s",
            target_column,
            outer_test_key,
            candidate_index,
            len(candidate_params),
            params,
        )
        candidate_rows: list[dict[str, Any]] = []
        candidate_scores: list[float] = []
        candidate_best_epochs: list[int] = []

        for inner_fold in inner_folds:
            validation_key = inner_fold.validation_key or "unknown"
            run_name = (
                f"inner:test={outer_test_key}:val={validation_key}:target={target_column}:"
                f"candidate={candidate_index}"
            )
            score, val_metrics, model = _evaluate_candidate_on_inner_fold(
                fold_data=inner_fold,
                config=config,
                params=params,
                run_name=run_name,
            )
            summary = model.training_summary_
            candidate_scores.append(score)
            candidate_best_epochs.append(int(summary.best_epoch if summary is not None else 0))

            row = {
                "target_column": target_column,
                "outer_test_key": outer_test_key,
                "outer_test_label": refit_fold.test_label,
                "validation_key": validation_key,
                "validation_label": inner_fold.validation_label,
                "selection_metric": selection_metric,
                "selection_score": score,
                "best_epoch": None if summary is None else summary.best_epoch,
                "epochs_completed": None if summary is None else summary.epochs_completed,
            }
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            row.update({f"candidate_{key}": value for key, value in params.items()})
            candidate_rows.append(row)

        mean_score = float(np.mean(candidate_scores))
        all_inner_rows.extend(deepcopy(candidate_rows))
        logger.info(
            "✅ Finished: candidate evaluation target=%s outer_test=%s candidate=%d/%d mean_%s=%.6f best_inner_epoch_median=%d",
            target_column,
            outer_test_key,
            candidate_index,
            len(candidate_params),
            selection_metric,
            mean_score,
            _aggregate_refit_epochs(candidate_best_epochs, strategy="median"),
        )
        if mean_score < best_score:
            best_score = mean_score
            best_params = dict(params)
            best_epoch_values = candidate_best_epochs.copy()
            best_inner_rows = deepcopy(candidate_rows)

    if best_params is None:
        raise RuntimeError("No GP candidate could be selected from the nested inner folds.")

    refit_epochs = _aggregate_refit_epochs(best_epoch_values, strategy=epoch_strategy)
    logger.info(
        "✅ Finished: nested model selection target=%s outer_test=%s params=%s refit_epochs=%d mean_%s=%.6f",
        target_column,
        outer_test_key,
        best_params,
        refit_epochs,
        selection_metric,
        best_score,
    )

    final_model = fit_gp_with_params(
        fold_data=refit_fold,
        params=best_params,
        config=config,
        run_name=f"final:test={outer_test_key}:target={target_column}",
        epochs_override=refit_epochs,
    )

    return NestedTrainingResult(
        model=final_model,
        best_params=best_params,
        selection_metric=selection_metric,
        selection_score=float(best_score),
        refit_epochs=refit_epochs,
        all_inner_fold_rows=all_inner_rows,
        inner_fold_rows=best_inner_rows,
        final_history=deepcopy(final_model.training_history_),
    )
