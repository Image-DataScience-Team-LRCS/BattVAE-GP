from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from src.common.logger.logging import get_logger
from src.common.utils.config_schema import FullConfig

logger = get_logger(__name__)


def resolve_checkpoint_path(
    config: FullConfig,
    filename: Optional[str] = None,
) -> Path:
    if filename:
        checkpoint_path = Path(config.PATHS.model_save) / filename
        logger.info("Loading checkpoint %s", checkpoint_path.name)
    else:
        checkpoint_path = Path(config.PATHS.model_save) / "best_model.pth"
        logger.info("Loading default checkpoint %s", checkpoint_path.name)
    return checkpoint_path


def load_checkpoint_payload(
    config: FullConfig,
    device: torch.device,
    filename: Optional[str] = None,
) -> dict:
    checkpoint_path = resolve_checkpoint_path(config, filename=filename)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def load_best_model_checkpoint(
    model: torch.nn.Module,
    config: FullConfig,
    device: torch.device,
    filename: Optional[str] = None,
) -> dict:
    checkpoint_path = resolve_checkpoint_path(config, filename=filename)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["vae_state_dict"])
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    logger.info("Loaded VAE weights from %s", checkpoint_path)
    return checkpoint
