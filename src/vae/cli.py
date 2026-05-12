from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.common.logger.logging import get_logger
from src.common.utils.config_schema import FullConfig
from src.common.utils.utils import (
    resolve_config,
    load_datasets,
    profile_step,
    setup_environment,
)
from src.vae.inference.checkpoints import load_checkpoint_payload
from src.vae.inference.interpolation_inference import run_interpolation_inference
from src.vae.inference.run_inference import run_inference, visualize_results
from src.vae.models.gp_prior import build_gp_prior_model
from src.vae.models.vae import build_model
from src.vae.preprocessing.processor import preprocess_main
from src.vae.training.train import Trainer


logger = get_logger(__name__)


def load_training_split(
    config: FullConfig,
    split: str,
    frozen_norm_stats: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader, FullConfig, dict]:
    dataframe = profile_step(
        "Loading datasets",
        load_datasets,
        config=config,
        split=split,
    )
    train_loader, val_loader, extras = profile_step(
        "Preprocessing data",
        preprocess_main,
        dataframe,
        config,
        frozen_norm_stats=frozen_norm_stats,
    )
    return train_loader, val_loader, config, extras.get("norm_stats", {})


def load_single_dataset(
    config: FullConfig,
    dataset_key: str,
    frozen_norm_stats: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader, FullConfig, dict]:
    dataframe = profile_step(
        "Loading datasets",
        load_datasets,
        config=config,
        data=dataset_key,
        split="train",
    )
    train_loader, val_loader, extras = profile_step(
        "Preprocessing data",
        preprocess_main,
        dataframe,
        config,
        frozen_norm_stats=frozen_norm_stats,
    )
    return train_loader, val_loader, config, extras.get("norm_stats", {})


def initialize_model(
    config: FullConfig,
    device: torch.device,
) -> Any:
    model = profile_step("Building model", build_model, config)
    return model


def initialize_gp_components(
    config: FullConfig,
    device: torch.device,
) -> Tuple[Any, Any, Any]:
    gp_model, likelihood, mll = profile_step(
        "Building GPPR model",
        build_gp_prior_model,
        config,
        device,
    )
    return gp_model, likelihood, mll


def train_model(
    model: Any,
    config: FullConfig,
    device: torch.device,
    train_norm_stats: dict,
    gp_model: Any | None,
    likelihood: Any | None,
    mll: Any | None,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Any:
    trainer = Trainer(
        model,
        config,
        device,
        train_norm_stats=train_norm_stats,
        gp_model=gp_model,
        likelihood=likelihood,
        mll=mll,
    )
    vae, filename = profile_step(
        "Training model",
        trainer.train,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    return vae, filename


def run_full_inference(
    model: Any,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: FullConfig,
    device: torch.device,
    filename: Optional[str],
) -> Any:
    return profile_step(
        "Running inference",
        run_inference,
        model,
        train_loader,
        val_loader,
        config,
        device,
        filename,
    )


def generate_visuals(config: FullConfig, soh_csv_path: Path | None = None) -> None:
    profile_step("Generating visualizations", visualize_results, config, soh_csv_path)


def run(
    config_path: str | Path,
    dataset_key: Optional[str] = None,
    config_override: Optional[dict[str, Any]] = None,
) -> None:
    config = FullConfig(**resolve_config(str(config_path), config_override))
    device = setup_environment(seed=config.GENERAL.seed, use_cuda=True)
    model = initialize_model(config, device)

    filename = None
    soh_csv_path = None
    train_loader: DataLoader | None = None
    val_loader: DataLoader | None = None
    train_norm_stats: dict[str, Any] = {}
    gp_model = likelihood = mll = None

    if config.GENERAL.training or config.GENERAL.resume_training:
        train_loader, val_loader, _, train_norm_stats = load_training_split(config, split="train")
        if dataset_key is None:
            val_loader, _, _, _ = load_training_split(
                config,
                split="val",
                frozen_norm_stats=train_norm_stats,
            )
        gp_model, likelihood, mll = initialize_gp_components(config, device)
    elif config.GENERAL.inference or config.GENERAL.inference_interpolation:
        checkpoint = load_checkpoint_payload(config=config, device=device, filename=filename)
        train_norm_stats = dict(checkpoint.get("train_norm_stats", {}) or {})
        if not train_norm_stats:
            raise ValueError(
                "Checkpoint does not contain train_norm_stats. Re-train with a newer checkpoint before running inference or interpolation."
            )
        if config.GENERAL.inference:
            if dataset_key is None:
                raise ValueError("VAE inference now requires --dataset so only the requested dataset is loaded.")
            train_loader, _, _, _ = load_single_dataset(
                config,
                dataset_key,
                frozen_norm_stats=train_norm_stats,
            )

    if config.GENERAL.training or config.GENERAL.resume_training:
        _, filename = train_model(
            model,
            config,
            device,
            train_norm_stats,
            gp_model,
            likelihood,
            mll,
            train_loader,
            val_loader,
        )

    if config.GENERAL.inference:
        soh_csv_path = run_full_inference(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            filename=filename,
        )
        generate_visuals(config, soh_csv_path)

    if config.GENERAL.inference_interpolation:
        profile_step(
            "Running interpolation inference",
            run_interpolation_inference,
            model,
            config,
            device,
            train_norm_stats,
            filename,
        )

    torch.cuda.empty_cache()
    logger.info("Pipeline completed successfully.")
