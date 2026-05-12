from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.common.utils.config_schema import FullConfig
from src.common.utils.utils import resolve_config
from src.common.logger.logging import get_logger, setup_logging
from src.gp.data.data_preparation import load_experiment_config_model
from src.gp.cli import run as run_gp
from src.vae.cli import run as run_vae


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_VAE_CONFIG = REPO_ROOT / "configs" / "vae.yaml"
DEFAULT_GP_CONFIG = REPO_ROOT / "configs" / "gp.yaml"
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("vae", "gp"),
        required=True,
    )
    parser.add_argument("--config")
    parser.add_argument(
        "--dataset",
        help="Dataset key used by the VAE runner. Use `none` to rely on the configured train/val split.",
    )
    parser.add_argument(
        "--run",
        choices=("train", "inference", "interpolation"),
        help=(
            "Optional run-mode override. VAE supports train, inference, and interpolation. "
            "GP supports train and interpolation. If omitted, the config file controls execution."
        ),
    )
    args = parser.parse_args()
    if args.config is None:
        args.config = str(DEFAULT_VAE_CONFIG if args.model == "vae" else DEFAULT_GP_CONFIG)
    if args.model == "gp" and args.run == "inference":
        parser.error("--run inference is only supported with --model vae; use --run interpolation for GP")
    return args


def _build_vae_runtime_override(run_mode: str | None) -> dict | None:
    if run_mode is None:
        return None
    return {
        "GENERAL": {
            "training": run_mode == "train",
            "inference": run_mode == "inference",
            "inference_interpolation": run_mode == "interpolation",
        }
    }


def _build_gp_runtime_override(run_mode: str | None) -> dict | None:
    if run_mode is None:
        return None
    return {
        "experiment": {
            "training": run_mode == "train",
        },
        "interpolation": {
            "enabled": run_mode == "interpolation",
        }
    }


def _dispatch(
    model: str,
    config_path: str,
    dataset: str | None = None,
    vae_config_override: dict | None = None,
    gp_config_override: dict | None = None,
    run_mode: str | None = None,
) -> None:
    if model == "vae":
        dataset_key = None if dataset is not None and dataset.lower() == "none" else dataset
        run_vae(
            config_path=config_path,
            dataset_key=dataset_key,
            config_override=vae_config_override,
        )
        return

    run_gp(config_path=config_path, run_mode=run_mode, config_override=gp_config_override)


def main() -> None:
    args = parse_args()
    vae_config_override = None
    gp_config_override = None

    if args.model == "vae":
        vae_config_override = _build_vae_runtime_override(args.run)
        vae_config = FullConfig(**resolve_config(args.config, vae_config_override))
        log_level = logging.DEBUG if vae_config.GENERAL.debug_mode else logging.INFO
        setup_logging(
            filename_prefix=vae_config.GENERAL.experiment_name,
            console_output=True,
            level=log_level,
        )
    else:
        gp_config_override = _build_gp_runtime_override(args.run)
        gp_config = load_experiment_config_model(args.config, config_override=gp_config_override)
        setup_logging(
            filename_prefix=f"{gp_config.experiment.name}_",
            console_output=True,
            level=logging.INFO,
        )

    logger.info(
        "Starting root pipeline entrypoint for model %s with config %s",
        args.model,
        args.config,
    )
    _dispatch(args.model, args.config, args.dataset, vae_config_override, gp_config_override, args.run)


if __name__ == "__main__":
    main()
