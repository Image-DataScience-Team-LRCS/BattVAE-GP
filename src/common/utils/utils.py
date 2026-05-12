import os
import sys
import yaml
import csv
import shutil
import torch
import random
import pandas as pd
import numpy as np
import math
import time
from pathlib import Path
from src.common.logger.logging import setup_logging, get_logger
from pprint import pformat
from typing import Optional, Any, Callable, Literal
from src.common.utils.config_schema import FullConfig
from torch.utils.data import ConcatDataset, DataLoader

logger = get_logger(__name__)


def _merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_dict(path: Path) -> dict:
    with open(path, "r") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return data


def _normalize_vae_datasets(raw_datasets: dict) -> dict:
    normalized: dict = {}
    for key, value in raw_datasets.items():
        if not isinstance(value, dict):
            normalized[key] = value
            continue

        # Grouped entries such as `gp_interpolation: {data8: {...}, data9: {...}}`
        # are flattened into `gp_interpolation.data8` so the runtime config keeps
        # a simple dataset registry while the YAML stays visually organized.
        if "path" not in value and all(isinstance(child, dict) for child in value.values()):
            for child_key, child_value in value.items():
                normalized[f"{key}.{child_key}"] = child_value
            continue

        normalized[key] = value
    return normalized


def profile_step(name: str, func: Callable[..., Any], *args, **kwargs) -> Any:
    logger.info(f"⏳ Starting: {name}")
    start = time.time()
    result = func(*args, **kwargs)
    logger.info(f"✅ Finished: {name} in {time.time() - start:.2f} seconds")
    return result


def load_datasets(config: FullConfig, data: Optional[str] = None, split: Literal["train", "val", "test"] = "train",) -> pd.DataFrame:
    if config is None:
        logger.error(
            "Configuration dictionary 'config' must be provided.", exc_info=True
        )
        sys.exit(1)

    datasets_config = config.DATASETS

    # Inference mode: load only the specified dataset
    if data is not None:
        if data not in datasets_config:
            logger.error(f"Dataset '{data}' not found in configuration.", exc_info=True)
            sys.exit(1)

        dataset_info = datasets_config[data]
        path = dataset_info.path

        if not path or not os.path.exists(path):
            logger.error(
                f"File for dataset '{data}' does not exist at: {path}", exc_info=True
            )
            sys.exit(1)

        try:
            df = pd.read_csv(path)
            info_dict = dataset_info.dict(exclude={"path"})

            # Add metadata columns
            for key, value in info_dict.items():
                if key not in df.columns:
                    df[key] = value

            df["source_dataset"] = data
            df["global_index"] = df.Cycle.astype(str)
            df["Current"] = -1 * df["Current"]

            logger.info(f"✅ Loaded inference data from {path}")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to load dataset '{data}': {e}", exc_info=True)
            sys.exit(1)

    # Training mode: load all datasets
    all_dfs = []

    # Training or validation mode
    if split == "train":
        selected_data_names = config.GENERAL.training_datasets
    elif split == "val":
        selected_data_names = config.GENERAL.validation_datasets
    else:
        logger.error(f"Invalid split name: {split}")
        sys.exit(1)

    # Intersection: keep only datasets that are in both config.DATASETS and training_datasets
    try:
        selected_datasets = {k: v for k, v in datasets_config.items() if k in selected_data_names}
        logger.info(f"Selected datasets: {selected_datasets.keys()}")
    except Exception as e:
        logger.error(f"�� Failed to find intersection between datasets and training datasets: {e}", exc_info=True)
        logger.error(f"Available datasets: {list(datasets_config.keys())}")
        logger.error(f"Training datasets: {selected_data_names}")
        sys.exit(1)

    for name, info in selected_datasets.items():
        path = info.path
        if not path or not os.path.exists(path):
            logger.info(f"⚠️ Skipping {name}: Invalid or missing path.")
            continue

        try:
            df = pd.read_csv(path)

            # Add metadata as columns
            info_dict = info.dict(exclude={"path"})

            # Add metadata columns
            # Add metadata columns
            for key, value in info_dict.items():
                if key not in df.columns:
                    df[key] = value

            df["source_dataset"] = name
            df["global_index"] = f"{info.id}_{name}" + df.Cycle.astype(str)
            df["Current"] = -1 * df["Current"]

            all_dfs.append(df)
            logger.info(f"✅ Loaded {name} from {path}")
        except Exception as e:
            logger.error(f"❌ Failed to load {name}: {e}", exc_info=True)

    if not all_dfs:
        logger.error("No valid datasets were loaded.")
        sys.exit(1)

    full_df = pd.concat(all_dfs, ignore_index=True)
    return full_df


# Load hyperparameters from a YAML configuration file
def load_config(config_path: str, *, log_config: bool = True) -> dict:
    path = Path(config_path).resolve()
    config = _load_yaml_dict(path)

    general_config = config.get("GENERAL", {}) or {}
    paths_file = general_config.pop("paths_file", None) or config.pop("PATHS_FILE", None)
    if paths_file:
        paths_config = _load_yaml_dict((path.parent / paths_file).resolve())
        config["PATHS"] = _merge_dicts(paths_config.get("PATHS", {}), config.get("PATHS", {}))

    datasets_file = general_config.pop("datasets_file", None) or config.pop("DATASETS_FILE", None)
    if datasets_file:
        datasets_config = _load_yaml_dict((path.parent / datasets_file).resolve())
        shared_vae_datasets = _normalize_vae_datasets(datasets_config.get("VAE_DATASETS", {}))
        shared_gp_interpolation_datasets = _normalize_vae_datasets(datasets_config.get("GP_INTERPOLATION_DATASETS", {}))
        config["DATASETS"] = _merge_dicts(shared_vae_datasets, config.get("DATASETS", {}))
        config["GP_INTERPOLATION_DATASETS"] = _merge_dicts(
            shared_gp_interpolation_datasets,
            config.get("GP_INTERPOLATION_DATASETS", {}),
        )

    if log_config:
        logger.info("Loaded config:\n%s", pformat(config))
        # logger.info("Loaded config:\n%s", yaml.dump(config))
    return config


def resolve_config(config_path: str, config_override: Optional[dict] = None) -> dict:
    config = load_config(config_path, log_config=False)
    if config_override:
        config = _merge_dicts(config, config_override)
    logger.info("Resolved config:\n%s", pformat(config))
    return config


def save_metrics(metrics: np.ndarray, metrics_path: Path) -> None:
    metrics_path.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "epoch",
        "train_loss",
        "reconstruction_loss",
        "kl_loss",
        "gp_loss",
        "val_loss",
        "val_gp_loss",
        "val_kl_loss",
        "val_reconstruction_loss",
        "val_soh_loss",
        "soh_loss",
        "kl_per_dimension",
    ]

    # Allow newly added metrics (for example flatness_loss / val_flatness_loss)
    # without breaking CSV export.
    all_keys = set(base_fields)
    for row in metrics:
        if isinstance(row, dict):
            all_keys.update(row.keys())
    extra_fields = sorted(k for k in all_keys if k not in base_fields)
    fieldnames = base_fields + extra_fields

    with open(metrics_path / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(metrics)


def save_original_data(array: np.ndarray, save_path: Path) -> None:
    dir_path = Path(save_path) / "reconstructions"
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)

    header = "Cycle,Voltage, Current, VoltageDerivative, CurrentDerivative\n"
    with open(save_path, mode="w", newline="") as file:
        file.write(header)
        for cycle_number, cycle_data in enumerate(array, start=1):

            voltage_data = cycle_data[0]
            current_data = cycle_data[1]
            voltage_derivative = cycle_data[2]
            current_derivative = cycle_data[3]

            # voltage_str = ' '.join(map(str, cycle_data))
            voltage_str = ",".join(map(lambda x: f"{x:.6f}", voltage_data))
            current_str = ",".join(map(lambda x: f"{x:.6f}", current_data))
            voltage_derivative_str = ",".join(
                map(lambda x: f"{x:.6f}", voltage_derivative)
            )
            current_derivative_str = ",".join(
                map(lambda x: f"{x:.6f}", current_derivative)
            )

            file.write(
                f"{int(cycle_number)}, [{voltage_str}], [{current_str}], [{voltage_derivative_str}], [{current_derivative_str}]\n"
            )


def save_reconstructed_data(
    reconstructed_data: torch.Tensor,
    cycle_numbers: torch.Tensor,
    batch_idx: int,
    save_dir: Path,
    epoch: int = False,
) -> None:
    # Convert tensors to NumPy arrays for easier processing
    reconstructed_data = reconstructed_data.cpu().detach().numpy()
    cycle_numbers = cycle_numbers.cpu().detach().numpy()

    # Prepare the output file
    if epoch:
        save_dir = Path(save_dir) / "reconstructions"
        epoch_filename = save_dir / f"reconstructed_epoch_{epoch + 1}.csv"
    else:
        epoch_filename = Path(save_dir) / f"reconstructed_final.csv"

    mode = "w" if batch_idx == 0 else "a"

    # Write header if it's the first batch
    header = (
        "Cycle,Voltage, Current, VoltageDerivative, CurrentDerivative\n"
        if batch_idx == 0
        else ""
    )

    # Save data to the CSV file
    with open(epoch_filename, mode) as file:
        if header:
            file.write(header)
        for cycle_number, cycle_data in zip(cycle_numbers, reconstructed_data):
            voltage_data = cycle_data[0]
            current_data = cycle_data[1]
            voltage_derivative = cycle_data[2]
            current_derivative = cycle_data[3]

            # voltage_str = ' '.join(map(str, cycle_data))
            voltage_str = ",".join(map(lambda x: f"{x:.6f}", voltage_data))
            current_str = ",".join(map(lambda x: f"{x:.6f}", current_data))
            voltage_derivative_str = ",".join(
                map(lambda x: f"{x:.6f}", voltage_derivative)
            )
            current_derivative_str = ",".join(
                map(lambda x: f"{x:.6f}", current_derivative)
            )

            file.write(
                f"{int(cycle_number)}, [{voltage_str}], [{current_str}], [{voltage_derivative_str}], [{current_derivative_str}]\n"
            )


def normalize_targets(
    targets: torch.Tensor, max_val: int, min_val: int
) -> torch.Tensor:
    """Normalize targets to the range [0, 1]"""
    normalized_targets = (targets - min_val) / (max_val - min_val)
    return normalized_targets


def make_directories(config: dict) -> None:
    config_paths = config["PATHS"]
    for key, path in config_paths.items():
        dir_path = os.path.dirname(path) if os.path.isfile(path) else path
        os.makedirs(dir_path, exist_ok=True)


def merge_dataloaders(
    dataloader1: DataLoader, dataloader2: DataLoader, batch_size: int
) -> DataLoader:
    merged_dataset: torch.utils.data.Dataset = ConcatDataset(
        [dataloader1.dataset, dataloader2.dataset]
    )
    merged_loader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=True)
    return merged_loader


def setup_environment(
    seed: int = 42, use_cuda: bool = True, memory_fraction: Optional[float] = None
) -> torch.device:
    set_seed(seed)
    cuda_available = torch.cuda.is_available()
    if use_cuda and not cuda_available:
        logger.error(
            "CUDA requested but not available. Check GPU installation.", exc_info=True
        )
        raise RuntimeError("CUDA requested but not available. Check GPU installation.")

    device = torch.device("cuda" if (use_cuda and cuda_available) else "cpu")

    # Configure CUDA settings if using GPU
    if device.type == "cuda":
        current_device = torch.cuda.current_device()

        if memory_fraction:
            torch.cuda.set_per_process_memory_fraction(memory_fraction, current_device)
            logger.info(
                f"🔧 Setting CUDA memory fraction to {memory_fraction * 100:.1f}%"
            )

        # Print GPU info
        gpu_properties = torch.cuda.get_device_properties(current_device)
        logger.info(f"  🖥️ GPU Information:")
        logger.info(f"   • Name: {gpu_properties.name}")
        logger.info(f"   • Memory: {gpu_properties.total_memory / 1024**3:.1f} GB")
        logger.info(
            f"   • Compute Capability: {gpu_properties.major}.{gpu_properties.minor}"
        )

        # Clear GPU cache
        torch.cuda.empty_cache()
    else:
        logger.warning("\n💻 Using CPU for computation")

    logger.info(f"  🔧 Environment setup complete:")
    logger.info(f"   • Device: {device}")
    logger.info(f"   • Random Seed: {seed}")
    logger.info(f"   • PyTorch Version: {torch.__version__}")

    return device


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
