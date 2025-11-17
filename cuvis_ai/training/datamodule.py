"""GraphDataModule base class for consistent data loading in cuvis.ai training."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class CuvisDataModule(pl.LightningDataModule, ABC):
    """Abstract base class for data modules used with cuvis.ai Graph training.

    This class enforces a consistent interface for data loading across different
    datasets. All cuvis.ai datamodules should inherit from this base to ensure
    compatibility with the Graph.train() method.

    **IMPORTANT**: All dataloaders must yield dictionaries with the following structure:
    - Required keys: "cube" (or "x") - input tensor
    - Optional keys: "mask" (or "labels") - ground truth labels
    - Additional keys: any metadata needed for processing

    The datamodule must provide at minimum:
    - train_dataloader() for training data
    - val_dataloader() for validation data
    - optionally test_dataloader() for test data

    Examples
    --------
    >>> class MyDataModule(GraphDataModule):
    ...     def __init__(self, data_dir: str, batch_size: int = 32):
    ...         super().__init__()
    ...         self.data_dir = data_dir
    ...         self.batch_size = batch_size
    ...
    ...     def prepare_data(self):
    ...         # Download or preprocess data (runs on single GPU)
    ...         pass
    ...
    ...     def setup(self, stage: str = None):
    ...         # Load data for train/val/test (runs on all GPUs)
    ...         pass
    ...
    ...     def train_dataloader(self):
    ...         # IMPORTANT: Must yield dictionaries with "cube" or "x" key
    ...         return DataLoader(
    ...             self.train_dataset,
    ...             batch_size=self.batch_size,
    ...             collate_fn=self._collate_to_dict
    ...         )
    ...
    ...     def val_dataloader(self):
    ...         return DataLoader(
    ...             self.val_dataset,
    ...             batch_size=self.batch_size,
    ...             collate_fn=self._collate_to_dict
    ...         )
    ...
    ...     def _collate_to_dict(self, batch):
    ...         # Example collate function that returns dictionary
    ...         return {
    ...             "cube": torch.stack([item[0] for item in batch]),
    ...             "mask": torch.stack([item[1] for item in batch])
    ...         }
    """

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader.

        Returns
        -------
        DataLoader
            PyTorch DataLoader for training data
        """
        pass

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader.

        Returns
        -------
        DataLoader
            PyTorch DataLoader for validation data
        """
        pass

    def test_dataloader(self) -> DataLoader | None:
        """Return the test dataloader (optional).

        Returns
        -------
        DataLoader | None
            PyTorch DataLoader for test data, or None if not implemented
        """
        return None


__all__ = ["CuvisDataModule"]
