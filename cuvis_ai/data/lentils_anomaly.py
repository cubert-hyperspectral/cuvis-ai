from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from cuvis_ai.data.datasets import SingleCu3sDataset


def _first_available(path: Path, pattern: str) -> Path:
    """Return the first file matching pattern within path."""
    matches = sorted(path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find any files matching '{pattern}' in {path}")
    return matches[0]


def _resolve_assets(root: Path, dataset_name: str) -> tuple[Path, Path]:
    """Locate cube and label files for given dataset name.

    Args:
        root: Root directory containing dataset files
        dataset_name: Name of the dataset (e.g., "Lentils")

    Returns:
        Tuple of (cu3s_path, label_path)
    """
    default_cube = root / f"{dataset_name}.cu3s"
    default_label = root / f"{dataset_name}.json"

    cu3s = (
        default_cube if default_cube.exists() else _first_available(root, f"{dataset_name}*.cu3s")
    )
    label = (
        default_label if default_label.exists() else _first_available(root, f"{dataset_name}*.json")
    )
    return cu3s, label


class SingleCu3sDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cu3s_path: str | None = None,
        label_path: str | None = None,
        data_dir: str | None = None,
        dataset_name: str | None = None,
        train_ids: list[int] | None = None,
        val_ids: list[int] | None = None,
        test_ids: list[int] | None = None,
        batch_size: int = 2,
        processing_mode: str = "Reflectance",
    ) -> None:
        """Initialize SingleCu3sDataModule.

        Two modes of operation:
        1. Explicit paths: Provide both cu3s_path AND label_path
        2. Auto-resolve: Provide both data_dir AND dataset_name

        Args:
            cu3s_path: Direct path to .cu3s file (takes precedence)
            label_path: Direct path to .json annotation file (takes precedence)
            data_dir: Directory containing dataset files
            dataset_name: Name of dataset (e.g., "Lentils")
            train_ids: List of measurement indices for training
            val_ids: List of measurement indices for validation
            test_ids: List of measurement indices for testing
            batch_size: Batch size for dataloaders
            processing_mode: Cuvis processing mode string ("Raw", "Reflectance")

        Raises:
            ValueError: If neither (cu3s_path, label_path) nor (data_dir, dataset_name) provided
        """
        super().__init__()

        # Priority 1: Explicit paths
        if cu3s_path and label_path:
            self.cu3s_path = Path(cu3s_path)
            self.label_path = Path(label_path)
        # Priority 2: Auto-resolve from data_dir + dataset_name
        elif data_dir and dataset_name:
            self.cu3s_path, self.label_path = _resolve_assets(Path(data_dir), dataset_name)
        else:
            raise ValueError(
                "Must provide either (cu3s_path AND label_path) OR (data_dir AND dataset_name)"
            )

        self.batch_size = batch_size
        self.train_ids = train_ids or []
        self.val_ids = val_ids or []
        self.test_ids = test_ids or []
        self.processing_mode = processing_mode

    def prepare_data(self) -> None:
        # Only download if using auto-resolve mode with Lentils dataset
        # Skip for explicit paths (gRPC clients provide their own data)
        pass

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_ds = SingleCu3sDataset(
                cu3s_path=str(self.cu3s_path),
                label_path=str(self.label_path),
                processing_mode=self.processing_mode,
                measurement_indices=self.train_ids,
            )
            self.val_ds = SingleCu3sDataset(
                cu3s_path=str(self.cu3s_path),
                label_path=str(self.label_path),
                processing_mode=self.processing_mode,
                measurement_indices=self.val_ids,
            )

        if stage == "test" or stage is None:
            self.test_ds = SingleCu3sDataset(
                cu3s_path=str(self.cu3s_path),
                label_path=str(self.label_path),
                processing_mode=self.processing_mode,
                measurement_indices=self.test_ids,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, shuffle=False, batch_size=self.batch_size)
