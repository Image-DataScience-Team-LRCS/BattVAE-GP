from __future__ import annotations

import numpy as np
import pandas as pd

from src.gp.data.data_loader import FoldData, inverse_transform_array, transform_array
from src.gp.models.gp_model import SparseGaussianProcessRegressor
from src.gp.training.loss import interval_bounds, summarize_uncertainty
from src.common.logger.logging import get_logger


logger = get_logger(__name__)


def _predict_in_original_scale(
    model: SparseGaussianProcessRegressor,
    x_scaled: np.ndarray,
    target_scaler,
) -> tuple[np.ndarray, np.ndarray]:
    posterior = model.predict(x_scaled, include_noise=True)
    mean = inverse_transform_array(posterior.mean.reshape(-1, 1), target_scaler).reshape(-1)
    variance = posterior.variance * float(target_scaler.scale[0] ** 2)
    std = np.sqrt(np.maximum(variance, 1e-12))
    return mean, std


def predict_dataframe(
    model: SparseGaussianProcessRegressor,
    fold_data: FoldData,
    split: str = "test",
) -> pd.DataFrame:
    if split == "train":
        source_df = fold_data.train_df
        split_label = fold_data.train_key or "train"
    elif split == "val":
        if fold_data.val_df is None:
            raise ValueError("Validation split is not available for this fold.")
        source_df = fold_data.val_df
        split_label = fold_data.validation_key or "validation"
    elif split == "test":
        if fold_data.test_df is None:
            raise ValueError("Test split is not available for this fold.")
        source_df = fold_data.test_df
        split_label = fold_data.test_key or "test"
    else:
        raise ValueError(f"Unsupported split: {split}")

    x_scaled = transform_array(
        source_df[fold_data.feature_columns].to_numpy(),
        fold_data.feature_scaler,
    )
    mean, std = _predict_in_original_scale(model, x_scaled, fold_data.target_scaler)
    logger.info(
        "Generated %s predictions for split=%s target=%s: rows=%d mean_std=%.6f",
        split,
        split_label,
        fold_data.target_column,
        len(source_df),
        float(np.mean(std)),
    )

    result = source_df.copy().reset_index(drop=True)
    result["prediction_mean"] = mean
    result["prediction_std"] = std
    result["abs_error"] = np.abs(result[fold_data.target_column] - result["prediction_mean"])
    result["sq_error"] = (result[fold_data.target_column] - result["prediction_mean"]) ** 2

    for coverage in (0.50, 0.90, 0.95):
        lower, upper = interval_bounds(mean, std, coverage)
        suffix = str(int(coverage * 100))
        result[f"lower_{suffix}"] = lower
        result[f"upper_{suffix}"] = upper
        result[f"width_{suffix}"] = upper - lower
        result[f"hit_{suffix}"] = (
            (result[fold_data.target_column] >= lower) & (result[fold_data.target_column] <= upper)
        ).astype(int)

    return result


def evaluate_prediction_frame(prediction_df: pd.DataFrame, target_column: str) -> dict[str, float]:
    metrics = summarize_uncertainty(
        y_true=prediction_df[target_column].to_numpy(),
        mean=prediction_df["prediction_mean"].to_numpy(),
        std=prediction_df["prediction_std"].to_numpy(),
    )
    logger.info(
        "Prediction metrics: mae=%.6f rmse=%.6f nll=%.6f coverage50=%.4f coverage90=%.4f coverage95=%.4f",
        metrics["mae"],
        metrics["rmse"],
        metrics["nll"],
        metrics["coverage_50"],
        metrics["coverage_90"],
        metrics["coverage_95"],
    )
    return metrics
