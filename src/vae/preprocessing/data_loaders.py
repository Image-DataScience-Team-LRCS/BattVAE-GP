import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
from src.common.utils.config_schema import FullConfig
from src.common.logger.logging import get_logger
from torch.utils.data import DataLoader, TensorDataset, random_split

logger = get_logger(__name__)


class CreateDataLoader:
    def __init__(self, config: FullConfig) -> None:
        self.config = config
        self.hyper_parameters = config.HYPER_PARAMETERS

        self.train_loader = None
        self.val_loader = None
        self.tensor_dataset = None
    
    
    def create_data_loaders(
            self, features: np.ndarray, masks: np.ndarray, token_masks: np.ndarray, labels: np.ndarray
    ) -> tuple:
        """
        Create training and validation DataLoaders.

        Args:
            voltage_values (np.ndarray): Array of shape (n_cycles, sequence_length)
            cycle_labels (np.ndarray): Array of shape (n_cycles,)

        Returns:
            tuple: Contains (train_loader, val_loader, tensor_dataset)

        Raises:
            ValueError: If input shapes mismatch or invalid split parameters
            RuntimeError: If error occurs during dataset creation
        """
        try:
            self._validate_inputs(features, masks, token_masks, labels)
            self._create_tensor_dataset(features, masks, token_masks, labels)
            self._create_train_val_split()
            self._create_data_loaders()

            return self.train_loader, self.val_loader

        except Exception as e:
            logger.error(f"Error creating data loaders: {str(e)}")
            sys.exit(1)

    def _validate_inputs(
            self, features: np.ndarray, masks: np.ndarray, token_masks: np.ndarray, labels: np.ndarray
    ) -> None:
        """Validate input arrays for data loader creation."""
        if len(features) != len(masks) or len(features) != len(labels) or len(features) != len(token_masks):
            logger.error(f"Mismatch in input lengths")
            sys.exit(1)


    def _create_tensor_dataset(
            self, features: np.ndarray, masks: np.ndarray, token_masks: np.ndarray, labels: np.ndarray
    ) -> None:
        """Create TensorDataset from input arrays."""

        # feature_tensor = (torch.tensor(features, dtype=torch.float32))
        # mask_tensor = (torch.tensor(masks, dtype=torch.float32))
        # label_tensor = (torch.tensor(labels, dtype=torch.float32))

        feature_tensor = torch.from_numpy(features).contiguous()
        mask_tensor = torch.from_numpy(masks).contiguous()
        token_masks_tensor = torch.from_numpy(token_masks).contiguous()
        label_tensor = torch.from_numpy(labels).contiguous()

        self.tensor_dataset = TensorDataset(feature_tensor, mask_tensor, token_masks_tensor,  label_tensor)

    def _create_train_val_split(self) -> None:
        """Create training and validation dataset splits."""
        train_split = self.hyper_parameters.train_split
        # if not 0 < train_split < 1:
        #     raise ValueError(
        #         f"Invalid train_split value: {train_split}. Must be between 0 and 1"
        #     )

        dataset_size = len(self.tensor_dataset)
        train_size = int(train_split * dataset_size)
        val_size = dataset_size - train_size

        # if train_size == 0 or val_size == 0:
        #     raise ValueError(
        #         f"Invalid split sizes: train_size={train_size}, val_size={val_size}. "
        #         f"Adjust train_split parameter ({train_split})"
        #     )

        self.train_dataset, self.val_dataset = random_split(
            self.tensor_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        self.train_indices = self.train_dataset.indices
        self.val_indices = self.val_dataset.indices

    def _create_data_loaders(self) -> None:
        """Create DataLoader instances for training and validation."""
        batch_size = self.hyper_parameters.batch_size

        try:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=min(mp.cpu_count(), 4),
                pin_memory=True,
                persistent_workers=True
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=min(mp.cpu_count(), 4),
                pin_memory=True,
                persistent_workers=True
            )

            logger.info(f"✅ DataLoaders created:")
            logger.info(f"✅ Train samples: {len(self.train_dataset)}")
            logger.info(f"✅ Val samples: {len(self.val_dataset)}")
            logger.info(f"✅ Batch size: {batch_size}")
        except Exception as e:
            logger.error(f"Error creating optimized data loaders: {str(e)}")
            sys.exit(1)

    def create_full_loader(self) -> DataLoader:
        """Create an ordered loader over the full tensor dataset for inference/evaluation."""
        if self.tensor_dataset is None:
            raise RuntimeError("Tensor dataset has not been created yet.")

        batch_size = self.hyper_parameters.batch_size
        return DataLoader(
            self.tensor_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=min(mp.cpu_count(), 4),
            pin_memory=True,
            persistent_workers=True,
        )
