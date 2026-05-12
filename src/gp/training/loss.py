from __future__ import annotations

from statistics import NormalDist

import numpy as np


STANDARD_NORMAL = NormalDist()


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(error * error)))


def gaussian_nll(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true)
    mean = np.asarray(mean)
    variance = np.maximum(np.asarray(std) ** 2, eps)
    return float(np.mean(0.5 * (np.log(2.0 * np.pi * variance) + ((y_true - mean) ** 2) / variance)))


def interval_bounds(mean: np.ndarray, std: np.ndarray, coverage: float) -> tuple[np.ndarray, np.ndarray]:
    z_value = STANDARD_NORMAL.inv_cdf(0.5 + coverage / 2.0)
    lower = mean - z_value * std
    upper = mean + z_value * std
    return lower, upper


def empirical_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    mask = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(mask))


def average_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def summarize_uncertainty(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict[str, float]:
    summary: dict[str, float] = {
        "mae": mae(y_true, mean),
        "rmse": rmse(y_true, mean),
        "nll": gaussian_nll(y_true, mean, std),
        "mean_std": float(np.mean(std)),
    }

    for coverage in (0.50, 0.90, 0.95):
        lower, upper = interval_bounds(mean, std, coverage)
        suffix = str(int(coverage * 100))
        summary[f"coverage_{suffix}"] = empirical_coverage(y_true, lower, upper)
        summary[f"width_{suffix}"] = average_interval_width(lower, upper)

    return summary
