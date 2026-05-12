from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.common.logger.logging import setup_logging
from src.common.utils.config_schema import FullConfig
from src.common.utils.utils import load_config, load_datasets, setup_environment
from src.vae.inference.interpolation_inference import run_interpolation_inference
from src.vae.models.vae import build_model
from src.vae.preprocessing.processor import preprocess_main


def _align_model_config_with_checkpoint(config: FullConfig, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["vae_state_dict"]

    encoder_layers = {
        int(key.split(".")[3])
        for key in state_dict
        if key.startswith("encoder.encoder.layers.")
    }
    baseline_blocks = {
        int(key.split(".")[3])
        for key in state_dict
        if key.startswith("decoder.baseline.blocks.")
    }
    residual_blocks = {
        int(key.split(".")[3])
        for key in state_dict
        if key.startswith("decoder.residual.blocks.")
    }

    config.HYPER_PARAMETERS.num_transformer_encoder_layers = max(encoder_layers) + 1
    config.HYPER_PARAMETERS.num_transformer_decoder_layers = max(
        max(baseline_blocks, default=-1),
        max(residual_blocks, default=-1),
    ) + 1
    config.HYPER_PARAMETERS.n_fourier_encoder = int(
        state_dict["encoder.ff.freqs"].numel()
    )
    config.HYPER_PARAMETERS.n_fourier_baseline_decoder = int(
        (state_dict["decoder.baseline.in_proj.weight"].shape[1] - 1) // 2
    )
    config.HYPER_PARAMETERS.n_fourier_residual_decoder = int(
        (state_dict["decoder.residual.q_proj.weight"].shape[1] - 1) // 2
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode GP-interpolated latent trajectories with the trained VAE."
    )
    parser.add_argument(
        "--config",
        default="configs/vae.yaml",
        help="Path to the VAE config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FullConfig(**load_config(args.config))
    checkpoint_path = Path(config.PATHS.model_save) / "best_model.pth"
    _align_model_config_with_checkpoint(config, checkpoint_path)

    log_level = logging.DEBUG if config.GENERAL.debug_mode else logging.INFO
    setup_logging(
        filename_prefix=f"{config.GENERAL.experiment_name}interpolation",
        console_output=True,
        level=log_level,
    )

    device = setup_environment(seed=config.GENERAL.seed, use_cuda=True)

    training_frame = load_datasets(config=config, data=None, split="train")
    _, _, extras = preprocess_main(training_frame, config)
    train_norm_stats = extras.get("norm_stats", {})

    model = build_model(config)
    run_interpolation_inference(
        model=model,
        config=config,
        device=device,
        train_norm_stats=train_norm_stats,
        filename=None,
    )


if __name__ == "__main__":
    main()
