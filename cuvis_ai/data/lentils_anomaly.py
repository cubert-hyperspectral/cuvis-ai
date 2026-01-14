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
        Tuple of (cu3s_file_path, annotation_json_path)
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
        cu3s_file_path: str | None = None,
        annotation_json_path: str | None = None,
        data_dir: str | None = None,
        dataset_name: str | None = None,
        train_ids: list[int] | None = None,
        val_ids: list[int] | None = None,
        test_ids: list[int] | None = None,
        batch_size: int = 2,
        processing_mode: str = "Reflectance",
        normalize_to_unit: bool = False,
    ) -> None:
        """Initialize SingleCu3sDataModule.

        Two modes of operation:
        1. Explicit paths: Provide both cu3s_file_path AND annotation_json_path
        2. Auto-resolve: Provide both data_dir AND dataset_name

        Args:
            cu3s_file_path: Direct path to .cu3s file (takes precedence)
            annotation_json_path: Direct path to .json annotation file (takes precedence)
            data_dir: Directory containing dataset files
            dataset_name: Name of dataset (e.g., "Lentils")
            train_ids: List of measurement indices for training
            val_ids: List of measurement indices for validation
            test_ids: List of measurement indices for testing
            batch_size: Batch size for dataloaders
            processing_mode: Cuvis processing mode string ("Raw", "Reflectance")
            normalize_to_unit: If True, normalize cube per-channel to [0, 1].
                For band selection workflows, keep False to preserve spectral ratios.

        Raises:
            ValueError: If neither (cu3s_file_path, annotation_json_path) nor (data_dir, dataset_name) provided
        """
        super().__init__()

        # Priority 1: Explicit paths
        if cu3s_file_path and annotation_json_path:
            self.cu3s_file_path = Path(cu3s_file_path)
            self.annotation_json_path = Path(annotation_json_path)
        # Priority 2: Auto-resolve from data_dir + dataset_name
        elif data_dir and dataset_name:
            self.cu3s_file_path, self.annotation_json_path = _resolve_assets(
                Path(data_dir), dataset_name
            )
        else:
            raise ValueError(
                "Must provide either (cu3s_file_path AND annotation_json_path) OR (data_dir AND dataset_name)"
            )

        self.batch_size = batch_size
        self.train_ids = train_ids or []
        self.val_ids = val_ids or []
        self.test_ids = test_ids or []
        self.processing_mode = processing_mode
        self.normalize_to_unit = normalize_to_unit
        self.train_ds: SingleCu3sDataset | None = None
        self.val_ds: SingleCu3sDataset | None = None
        self.test_ds: SingleCu3sDataset | None = None

    def prepare_data(self) -> None:
        # Only download if using auto-resolve mode with Lentils dataset
        # Skip for explicit paths (gRPC clients provide their own data)
        pass

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            if self.train_ids:
                self.train_ds = SingleCu3sDataset(
                    cu3s_file_path=str(self.cu3s_file_path),
                    annotation_json_path=str(self.annotation_json_path),
                    processing_mode=self.processing_mode,
                    measurement_indices=self.train_ids,
                    normalize_to_unit=self.normalize_to_unit,
                )
            else:
                self.train_ds = None

            if self.val_ids:
                self.val_ds = SingleCu3sDataset(
                    cu3s_file_path=str(self.cu3s_file_path),
                    annotation_json_path=str(self.annotation_json_path),
                    processing_mode=self.processing_mode,
                    measurement_indices=self.val_ids,
                    normalize_to_unit=self.normalize_to_unit,
                )
            else:
                self.val_ds = None

        if stage == "test" or stage is None:
            if not self.test_ids:
                raise ValueError("test_ids must be provided to build the test dataset.")
            self.test_ds = SingleCu3sDataset(
                cu3s_file_path=str(self.cu3s_file_path),
                annotation_json_path=str(self.annotation_json_path),
                processing_mode=self.processing_mode,
                measurement_indices=self.test_ids,
                normalize_to_unit=self.normalize_to_unit,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("Train dataset is not initialized. Call setup('fit') first.")
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("Validation dataset is not initialized.")
        return DataLoader(self.val_ds, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("Test dataset is not initialized. Call setup('test') first.")
        return DataLoader(self.test_ds, shuffle=False, batch_size=self.batch_size)
