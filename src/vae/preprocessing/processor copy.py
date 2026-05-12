"""
Preprocessing module for battery cycling data.
Handles the preprocessing pipeline including data loading, noise addition,
normalization and data loader creation.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from src.common.logger.logging import get_logger
from src.vae.preprocessing.data_preprocessing import DataProcessor
from src.vae.preprocessing.data_loaders import CreateDataLoader
from src.vae.analysis.plot_curves import plot_feature_curves, compare_curves
from src.vae.analysis.plot_capacity_multichannel import plot_capacity_multichannel, plot_capacity_multichannel_paper
from src.common.utils.config_schema import FullConfig
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm

from typing import List, Tuple, Dict, Optional
import sys

from functools import wraps
import time
from contextlib import contextmanager

logger = get_logger(__name__)



def validate_padding(max_cycle_length: int, config: FullConfig) -> int:
    extra_padding = 8 - (max_cycle_length % 8)

    logger.warning("⚠️ Please be carefull here extra padding is controlled. ⚠️")
    if config.HYPER_PARAMETERS.input_seq_len is None:
        config.HYPER_PARAMETERS.input_seq_len = extra_padding + max_cycle_length
        logger.info(
            f"Extra Padding: {extra_padding} is added to make length {max_cycle_length} divisible by 8."
        )
        logger.info(
        f"⚙️  Input sequence length is set to {config.HYPER_PARAMETERS.input_seq_len}."
        )
        return extra_padding

    extra_padding = config.HYPER_PARAMETERS.input_seq_len - max_cycle_length
    logger.info(
        f"⚙️  Input sequence length is set to {config.HYPER_PARAMETERS.input_seq_len}."
    )
    return extra_padding


def preprocess_main(
    data: pd.DataFrame,
    config: FullConfig,
    frozen_norm_stats: Optional[Dict[str, dict]] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, np.ndarray]]:
    """
    Main function to preprocess the battery cycling data for diffusion model.

    This function:
    1. Loads and validates configuration
    2. Loads raw battery cycling data
    3. Processes data through preprocessing pipeline
    4. Creates training and validation data loaders
    5. Generates visualization for validation

    Returns:
        tuple: Contains:
            - train_loader (DataLoader): Training data loader
            - val_loader (DataLoader): Validation data loader
            - tensor_dataset (TensorDataset): Complete dataset

    Raises:
        FileNotFoundError: If configuration or data files are not found
        RuntimeError: If preprocessing pipeline fails
    """
    try:
        logger.info("Initializing data processor...")
        processor = DataProcessor(config)

        logger.info("Processing data through pipeline...")
        max_cycle_length = processor.prepare_cycle_data(data)
        validate_padding(max_cycle_length, config)

        norm_cfg = config.NORMALIZATION.model_dump()
        features, masks, labels, token_masks, feature_names, label_names, extras = (
            processor.create_capacity_features_and_masks(
                normalize=True,
                norm_cfg=norm_cfg,
                frozen_norm_stats=frozen_norm_stats,
            )
        )

        plot_capacity_multichannel(
            features, masks, labels, feature_names,
            source_datasets=["dataset_0"], 
            cycles=[10, 1000, 4500],
            save_dir=Path("artifacts/visualizations"), 
            dpi=300,
            combine_datasets=True,
            color_mode="cycle",  # "categorical" or "cycle"
            cmap_name="viridis", # used when color_mode="cycle"
        )

        # plot_capacity_multichannel_paper(
        #     features, masks, labels, feature_names,
        #     source_datasets=["dataset_7"],
        #     # cycles=[10, 1000, 4500],
        #     save_dir=Path("artifacts/visualizations"),
        #     dpi=600,
        #     combine_datasets=True,
        #     charge_cmap_name="viridis",
        #     discharge_cmap_name="rainbow",
        # )


        logger.info(
            "Simplified preprocessing ready with channels: %s",
            ", ".join(feature_names),
        )

        data_loader = CreateDataLoader(config)
        logger.info("Creating data loaders...")
        (
            train_loader,
            val_loader,
        ) = data_loader.create_data_loaders(features, masks, token_masks, labels)
        extras["full_loader"] = data_loader.create_full_loader()

        logger.info("Preprocessing completed successfully")
        return train_loader, val_loader, extras

    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    raise SystemExit("Run preprocessing via main.py so train normalization stats can be reused for validation.")
