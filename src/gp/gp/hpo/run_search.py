from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_CONFIG = REPO_ROOT / "experiments" / "gp_hpo" / "base_config.yaml"
DEFAULT_RUNS_ROOT = REPO_ROOT / "experiments" / "gp_hpo" / "runs"


@dataclass
class HPOSettings:
    search_mode: str
    max_trials: int
    random_seed: int
    objective_column: str
    run_after_training: bool
    holdout_datasets: list[str]
    search_space: dict[str, list[Any]]
    sync_methods_with_defaults: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an isolated overnight HPO sweep for the GP latent-space pipeline."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help="Path to the dedicated HPO base config.",
    )
    parser.add_argument(
        "--hpo-root",
        type=Path,
        default=None,
        help="Output root for the HPO run. Defaults to experiments/gp_hpo/runs/<timestamp>.",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to launch each GP trial.",
    )
    parser.add_argument(
        "--search-mode",
        choices=["random", "grid"],
        default=None,
        help="Override the HPO search mode from the base config.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Override the number of trials to schedule.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Override the random seed used to sample trials.",
    )
    parser.add_argument(
        "--objective-column",
        default=None,
        help="Override the summary.csv column optimized across trials.",
    )
    parser.add_argument(
        "--run-after-training",
        action="store_true",
        help="Enable the GP analysis pipeline for each HPO trial.",
    )
    parser.add_argument(
        "--holdout-datasets",
        nargs="+",
        default=None,
        help="Override experiment.holdout_datasets for all trials.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and manifests without executing any GP trials.",
    )
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _deep_copy(data: Any) -> Any:
    return json.loads(json.dumps(data))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_relative_path(base_path: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_path.parent / path).resolve()


def _normalize_search_space(raw: dict[str, Any]) -> dict[str, list[Any]]:
    normalized: dict[str, list[Any]] = {}
    for key, values in raw.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"HPO search_space entry '{key}' must be a non-empty list.")
        normalized[str(key)] = values
    return normalized


def load_hpo_settings(base_config: dict[str, Any], args: argparse.Namespace) -> HPOSettings:
    raw_hpo = dict(base_config.get("hpo", {}) or {})
    if not raw_hpo:
        raise ValueError("Base HPO config is missing the top-level 'hpo' section.")

    search_mode = str(args.search_mode or raw_hpo.get("search_mode", "random")).lower()
    if search_mode not in {"random", "grid"}:
        raise ValueError(f"Unsupported search mode: {search_mode}")

    max_trials = int(args.max_trials or raw_hpo.get("max_trials", 24))
    if max_trials <= 0:
        raise ValueError("max_trials must be greater than zero.")

    random_seed = int(args.random_seed or raw_hpo.get("random_seed", 42))
    objective_column = str(args.objective_column or raw_hpo.get("objective_column", "rmse"))
    holdout_datasets = [str(item) for item in (args.holdout_datasets or raw_hpo.get("holdout_datasets", []))]
    if not holdout_datasets:
        raise ValueError("HPO holdout_datasets must contain at least one dataset key.")

    run_after_training = bool(raw_hpo.get("run_after_training", False) or args.run_after_training)
    search_space = _normalize_search_space(dict(raw_hpo.get("search_space", {}) or {}))
    sync_methods_with_defaults = bool(raw_hpo.get("sync_methods_with_defaults", False))

    return HPOSettings(
        search_mode=search_mode,
        max_trials=max_trials,
        random_seed=random_seed,
        objective_column=objective_column,
        run_after_training=run_after_training,
        holdout_datasets=holdout_datasets,
        search_space=search_space,
        sync_methods_with_defaults=sync_methods_with_defaults,
    )


def _set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = mapping
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def _sync_normalization_defaults(config: dict[str, Any]) -> None:
    normalization_cfg = dict(config.get("normalization", {}) or {})
    feature_method = normalization_cfg.get("feature_method")
    target_method = normalization_cfg.get("target_method")
    feature_columns = [str(column) for column in config.get("features", {}).get("input_columns", [])]
    target_columns = [str(column) for column in config.get("features", {}).get("target_columns", [])]

    if feature_method is not None and feature_columns:
        normalization_cfg["feature_column_methods"] = {
            column: str(feature_method) for column in feature_columns
        }
    if target_method is not None and target_columns:
        normalization_cfg["target_column_methods"] = {
            column: str(target_method) for column in target_columns
        }
    config["normalization"] = normalization_cfg


def _materialize_trial_config(
    base_config: dict[str, Any],
    base_config_path: Path,
    trial_id: str,
    trial_output_dir: Path,
    params: dict[str, Any],
    settings: HPOSettings,
) -> dict[str, Any]:
    config = _deep_copy(base_config)
    config.pop("hpo", None)

    config["experiment"]["name"] = f"{config['experiment']['name']}_{trial_id}"
    config["experiment"]["holdout_datasets"] = list(settings.holdout_datasets)
    config["PATHS"]["gp_output_dir"] = str(trial_output_dir)
    config.setdefault("analysis", {})["run_after_training"] = settings.run_after_training
    config.setdefault("search", {})["enabled"] = False
    config.setdefault("study", {})["run_kernel_ablation"] = False
    config.setdefault("study", {})["run_feature_normalization_ablation"] = False
    config.setdefault("study", {})["run_target_normalization_ablation"] = False

    interpolation_cfg = dict(config.get("interpolation", {}) or {})
    if interpolation_cfg:
        config["interpolation"] = interpolation_cfg
        config["PATHS"]["gp_model_summary_csv"] = str(trial_output_dir / "deployment_models.csv")
        config["PATHS"]["gp_interpolation_output_csv"] = str(trial_output_dir / "interpolation")

    for dataset_key, dataset_meta in config.get("datasets", {}).items():
        latent_csv = dataset_meta.get("latent_csv")
        if latent_csv:
            dataset_meta["latent_csv"] = str(_resolve_relative_path(base_config_path, latent_csv))
        config["datasets"][dataset_key] = dataset_meta

    for dotted_key, value in params.items():
        _set_nested(config, dotted_key, value)

    if settings.sync_methods_with_defaults:
        _sync_normalization_defaults(config)

    return config


def _combination_count(search_space: dict[str, list[Any]]) -> int:
    total = 1
    for values in search_space.values():
        total *= len(values)
    return total


def _grid_trials(search_space: dict[str, list[Any]], max_trials: int) -> list[dict[str, Any]]:
    keys = list(search_space.keys())
    iterator = itertools.product(*(search_space[key] for key in keys))
    scheduled: list[dict[str, Any]] = []
    for values in itertools.islice(iterator, max_trials):
        scheduled.append(dict(zip(keys, values, strict=True)))
    return scheduled


def _random_trials(search_space: dict[str, list[Any]], max_trials: int, random_seed: int) -> list[dict[str, Any]]:
    keys = list(search_space.keys())
    total = _combination_count(search_space)
    if max_trials >= total and total <= 100000:
        return _grid_trials(search_space, total)

    rng = random.Random(random_seed)
    scheduled: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    max_attempts = max(1000, max_trials * 50)

    while len(scheduled) < min(max_trials, total) and max_attempts > 0:
        candidate_tuple = tuple(rng.choice(search_space[key]) for key in keys)
        if candidate_tuple not in seen:
            seen.add(candidate_tuple)
            scheduled.append(dict(zip(keys, candidate_tuple, strict=True)))
        max_attempts -= 1

    if len(scheduled) < min(max_trials, total):
        raise RuntimeError("Failed to sample enough unique HPO trials from the search space.")

    return scheduled


def schedule_trials(settings: HPOSettings) -> tuple[list[dict[str, Any]], int]:
    total = _combination_count(settings.search_space)
    if settings.search_mode == "grid":
        return _grid_trials(settings.search_space, min(settings.max_trials, total)), total
    return _random_trials(settings.search_space, settings.max_trials, settings.random_seed), total


def _result_fieldnames(param_keys: list[str]) -> list[str]:
    return [
        "trial_id",
        "status",
        "objective_column",
        "objective_value",
        "mean_selection_score",
        "mean_nll",
        "mean_rmse",
        "mean_mae",
        "runtime_seconds",
        "output_dir",
        "config_file",
        "trial_log",
        *param_keys,
    ]


def _write_results_header(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def _append_result_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)


def _write_manifest(
    manifest_path: Path,
    *,
    base_config_path: Path,
    hpo_root: Path,
    settings: HPOSettings,
    total_combinations: int,
    scheduled_trials: list[dict[str, Any]],
) -> None:
    payload = {
        "base_config": str(base_config_path),
        "hpo_root": str(hpo_root),
        "search_mode": settings.search_mode,
        "max_trials": settings.max_trials,
        "random_seed": settings.random_seed,
        "objective_column": settings.objective_column,
        "run_after_training": settings.run_after_training,
        "holdout_datasets": settings.holdout_datasets,
        "total_combinations": total_combinations,
        "scheduled_trials": scheduled_trials,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mean_metric(summary_df: pd.DataFrame, column: str) -> float | None:
    if column not in summary_df.columns:
        return None
    return float(summary_df[column].mean())


def _run_trial(
    *,
    python_bin: Path,
    project_root: Path,
    config_path: Path,
    trial_log_path: Path,
) -> tuple[int, float]:
    start_time = time.time()
    with trial_log_path.open("w", encoding="utf-8") as trial_log_handle:
        completed = subprocess.run(
            [str(python_bin), "-m", "src.gp.main", "--config", str(config_path)],
            cwd=str(project_root),
            stdout=trial_log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode, time.time() - start_time


def _copy_best_outputs(
    *,
    best_result: dict[str, Any],
    best_outputs_dir: Path,
    best_config_copy: Path,
    best_interpolation_config: Path,
) -> None:
    best_output_source = Path(str(best_result["output_dir"])).resolve()
    if best_outputs_dir.exists():
        shutil.rmtree(best_outputs_dir)
    shutil.copytree(best_output_source, best_outputs_dir)

    with best_config_copy.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["PATHS"]["gp_output_dir"] = str(best_outputs_dir)
    interpolation_cfg = dict(config.get("interpolation", {}) or {})
    if interpolation_cfg:
        config["interpolation"] = interpolation_cfg
        config["PATHS"]["gp_model_summary_csv"] = str(best_outputs_dir / "deployment_models.csv")
        config["PATHS"]["gp_interpolation_output_csv"] = str(best_outputs_dir / "interpolation")

    with best_interpolation_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def main() -> None:
    args = parse_args()
    base_config_path = args.base_config.expanduser().resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base HPO config not found: {base_config_path}")

    base_config = _load_yaml(base_config_path)
    settings = load_hpo_settings(base_config, args)
    hpo_root = (
        args.hpo_root.expanduser().resolve()
        if args.hpo_root is not None
        else (DEFAULT_RUNS_ROOT / _timestamp()).resolve()
    )
    # Preserve the virtualenv executable path instead of resolving the symlink
    # to the system interpreter, otherwise child trials lose the venv packages.
    python_bin = args.python_bin.expanduser()
    if not python_bin.is_absolute():
        python_bin = (Path.cwd() / python_bin).resolve()
    if not python_bin.exists():
        raise FileNotFoundError(f"Python executable not found: {python_bin}")

    configs_dir = hpo_root / "configs"
    logs_dir = hpo_root / "logs"
    trials_dir = hpo_root / "trials"
    results_dir = hpo_root / "results"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    trials_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    scheduled_trials, total_combinations = schedule_trials(settings)
    manifest_path = results_dir / "scheduled_trials.json"
    _write_manifest(
        manifest_path,
        base_config_path=base_config_path,
        hpo_root=hpo_root,
        settings=settings,
        total_combinations=total_combinations,
        scheduled_trials=scheduled_trials,
    )

    param_keys = list(settings.search_space.keys())
    results_csv_path = results_dir / "hpo_results.csv"
    best_json_path = results_dir / "best_trial.json"
    best_config_copy = results_dir / "best_config.yaml"
    best_outputs_dir = results_dir / "best_outputs"
    best_interpolation_config = results_dir / "best_interpolation_config.yaml"
    fieldnames = _result_fieldnames(param_keys)
    _write_results_header(results_csv_path, fieldnames)

    print("============================================================")
    print("GP HPO runner")
    print(f"Base config:        {base_config_path}")
    print(f"Python:             {python_bin}")
    print(f"HPO root:           {hpo_root}")
    print(f"Search mode:        {settings.search_mode}")
    print(f"Objective column:   {settings.objective_column}")
    print(f"Holdout datasets:   {settings.holdout_datasets}")
    print(f"Total combinations: {total_combinations}")
    print(f"Scheduled trials:   {len(scheduled_trials)}")
    print(f"Dry run:            {args.dry_run}")
    print("============================================================")

    best_result: dict[str, Any] | None = None

    for index, params in enumerate(scheduled_trials, start=1):
        trial_id = f"trial_{index:04d}"
        trial_root = trials_dir / trial_id
        trial_output_dir = trial_root / "outputs"
        trial_output_dir.mkdir(parents=True, exist_ok=True)
        trial_log_path = logs_dir / f"{trial_id}.log"
        config_path = configs_dir / f"{trial_id}.yaml"

        trial_config = _materialize_trial_config(
            base_config=base_config,
            base_config_path=base_config_path,
            trial_id=trial_id,
            trial_output_dir=trial_output_dir,
            params=params,
            settings=settings,
        )
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(trial_config, handle, sort_keys=False)

        row = {
            "trial_id": trial_id,
            "status": "scheduled" if args.dry_run else "pending",
            "objective_column": settings.objective_column,
            "objective_value": None,
            "mean_selection_score": None,
            "mean_nll": None,
            "mean_rmse": None,
            "mean_mae": None,
            "runtime_seconds": None,
            "output_dir": str(trial_output_dir),
            "config_file": str(config_path),
            "trial_log": str(trial_log_path),
            **params,
        }

        print("------------------------------------------------------------")
        print(f"{trial_id}: {json.dumps(params, sort_keys=True)}")
        print(f"Config file: {config_path}")
        print(f"Output dir:  {trial_output_dir}")

        if args.dry_run:
            _append_result_row(results_csv_path, fieldnames, row)
            continue

        return_code, runtime_seconds = _run_trial(
            python_bin=python_bin,
            project_root=REPO_ROOT,
            config_path=config_path,
            trial_log_path=trial_log_path,
        )
        row["runtime_seconds"] = round(runtime_seconds, 2)
        row["status"] = "ok" if return_code == 0 else f"failed({return_code})"

        summary_path = trial_output_dir / "summary.csv"
        if return_code == 0 and summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            if settings.objective_column not in summary_df.columns:
                raise ValueError(
                    f"Objective column '{settings.objective_column}' not found in {summary_path}. "
                    f"Available columns: {list(summary_df.columns)}"
                )
            row["objective_value"] = float(summary_df[settings.objective_column].mean())
            row["mean_selection_score"] = _mean_metric(summary_df, "selection_score")
            row["mean_nll"] = _mean_metric(summary_df, "nll")
            row["mean_rmse"] = _mean_metric(summary_df, "rmse")
            row["mean_mae"] = _mean_metric(summary_df, "mae")

            if best_result is None or float(row["objective_value"]) < float(best_result["objective_value"]):
                best_result = dict(row)
                best_json_path.write_text(json.dumps(best_result, indent=2), encoding="utf-8")
                shutil.copy2(config_path, best_config_copy)
        else:
            print(f"Trial failed: {trial_id} (see {trial_log_path})")

        _append_result_row(results_csv_path, fieldnames, row)
        if row["objective_value"] is not None:
            print(
                f"Finished {trial_id}: objective[{settings.objective_column}]={float(row['objective_value']):.6f} "
                f"runtime={runtime_seconds:.2f}s"
            )
        else:
            print(f"Finished {trial_id}: status={row['status']} runtime={runtime_seconds:.2f}s")

    if args.dry_run:
        print("Dry run complete.")
        print(f"Scheduled configs: {configs_dir}")
        print(f"Manifest:          {manifest_path}")
        print(f"Results CSV:       {results_csv_path}")
        return

    if best_result is None:
        raise RuntimeError(
            "HPO finished, but no successful trial produced a valid summary. "
            f"Inspect {results_csv_path} and {logs_dir}."
        )

    _copy_best_outputs(
        best_result=best_result,
        best_outputs_dir=best_outputs_dir,
        best_config_copy=best_config_copy,
        best_interpolation_config=best_interpolation_config,
    )

    print("============================================================")
    print("GP HPO completed successfully.")
    print(f"Best trial:                {best_result['trial_id']}")
    print(f"Best {settings.objective_column}:          {float(best_result['objective_value']):.6f}")
    print(f"Best config:               {best_config_copy}")
    print(f"Best outputs copy:         {best_outputs_dir}")
    print(f"Best interpolation config: {best_interpolation_config}")
    print(f"Best trial summary:        {best_json_path}")
    print(f"All results:               {results_csv_path}")
    print("============================================================")


if __name__ == "__main__":
    main()
