from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.logger.logging import get_logger
from src.gp.analysis import build_tables, plot_results
from src.gp.data.data_preparation import load_experiment_config_model, load_latent_space_frame
from src.gp.inference import interpolate_latent
from src.gp.training.run_holdout import run_nested_holdout_training


logger = get_logger(__name__)


def _as_config_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


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


def _outer_test_datasets(config: dict[str, Any]) -> list[str]:
    configured = _as_config_list(config["experiment"].get("holdout_datasets", []))
    if configured:
        return [str(key) for key in configured]
    return [str(key) for key in config["datasets"].keys()]


def _target_columns(config: dict[str, Any]) -> list[str]:
    configured = _as_config_list(config["features"].get("target_columns", []))
    if configured:
        return [str(column) for column in configured]
    return [str(config["features"]["target_column"])]


def run(
    config_path: str | Path,
    run_mode: str | None = None,
    config_override: dict[str, Any] | None = None,
) -> None:
    config_model = load_experiment_config_model(str(config_path))
    config = _merge_config(config_model.model_dump(mode="python"), config_override)

    if config_model.experiment.model_type != "gp":
        raise ValueError("This entry point supports only `model_type: gp`.")
    if run_mode not in (None, "train", "interpolation"):
        raise ValueError("run_mode must be one of None, 'train', or 'interpolation'.")

    output_dir = Path(str(config["PATHS"]["gp_output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    training_enabled = bool(config["experiment"].get("training", True))
    interpolation_enabled = bool(config.get("interpolation", {}).get("enabled", False))

    if run_mode == "interpolation" or (not training_enabled and interpolation_enabled):
        logger.info("Starting GP interpolation inference from saved deployment models")
        interpolation_csv = interpolate_latent.run(config["_config_path"], config_override=config_override)
        logger.info("Finished GP interpolation inference at %s", interpolation_csv)
        return

    if not training_enabled:
        raise ValueError("GP training is disabled and interpolation is not enabled. Nothing to run.")

    logger.info("Starting GP nested holdout pipeline with config %s", config["_config_path"])
    logger.info("GP outputs will be written to %s", output_dir)

    frame = load_latent_space_frame(config)
    outer_test_keys = _outer_test_datasets(config)

    logger.info("Loaded %d rows across %d datasets", len(frame), frame["dataset_key"].nunique())
    summary_df, deployment_df = run_nested_holdout_training(
        frame=frame,
        config=config,
        output_dir=output_dir,
        outer_test_keys=outer_test_keys,
    )
    if summary_df.empty:
        raise RuntimeError("The GP holdout pipeline produced no summary rows.")

    summary_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"
    resolved_config_path = output_dir / "resolved_config.json"
    summary_df.to_csv(summary_path, index=False)
    summary_df.to_json(summary_json_path, orient="records", indent=2)
    resolved_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    deployment_path = output_dir / "deployment_models.csv"
    deployment_df.to_csv(deployment_path, index=False)

    logger.info("Finished saving nested holdout metrics to %s", summary_path)
    logger.info("Finished saving deployment models summary to %s", deployment_path)

    if interpolation_enabled:
        logger.info("Starting GP latent interpolation from deployment models")
        try:
            interpolation_csv = interpolate_latent.run(config["_config_path"], config_override=config_override)
            logger.info("Finished GP latent interpolation at %s", interpolation_csv)
        except Exception:
            logger.exception("GP latent interpolation failed")

    analysis_cfg = dict(config.get("analysis", {}) or {})
    run_analysis = bool(analysis_cfg.get("run_after_training", False))
    cycle_bins = int(analysis_cfg.get("cycle_bins", 10))
    analysis_failed = False
    if run_analysis:
        logger.info(
            "Starting GP analysis pipeline with cycle_bins=%d for %s",
            cycle_bins,
            output_dir,
        )
        try:
            build_tables.run(output_dir=output_dir, cycle_bins=cycle_bins)
            plot_results.run(output_dir=output_dir)
            logger.info("Finished GP analysis pipeline for %s", output_dir)
        except Exception:
            analysis_failed = True
            logger.exception("GP analysis pipeline failed")

    configured_target_columns = _target_columns(config)
    if analysis_failed:
        logger.warning(
            "Pipeline finished with analysis errors. targets=%d outer_test_crates=%d output_dir=%s",
            len(configured_target_columns),
            len(outer_test_keys),
            output_dir,
        )
    else:
        logger.info(
            "Pipeline completed successfully. targets=%d outer_test_crates=%d output_dir=%s",
            len(configured_target_columns),
            len(outer_test_keys),
            output_dir,
        )
