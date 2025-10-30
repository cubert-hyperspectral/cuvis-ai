from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cuvis
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from cuvis_ai.data.datasets import SingleCu3sDataset
from cuvis_ai.data.public_datasets import PublicDatasets


def _first_available(path: Path, pattern: str) -> Path:
    """Return the first file matching pattern within path."""
    matches = sorted(path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find any files matching '{pattern}' in {path}")
    return matches[0]


def _resolve_lentils_assets(root: Path) -> tuple[Path, Path]:
    """Locate the Lentils cube and label files regardless of suffix."""
    default_cube = root / "Lentils.cu3s"
    default_label = root / "Lentils.json"

    cu3s = default_cube if default_cube.exists() else _first_available(root, "Lentils*.cu3s")
    label = default_label if default_label.exists() else _first_available(root, "Lentils*.json")
    return cu3s, label


def _bucket_measurements(total: int) -> tuple[list[int], list[int], list[int]]:
    """
    Create train/val/test index splits that always fall within `[0, total)`.

    With very small datasets we fall back to reusing the available indices.
    """
    indices = list(range(total))
    if not indices:
        raise ValueError("No measurements found in the Lentils dataset.")

    train = indices[: min(1, total)]
    remaining = indices[len(train) :]

    def _take_or_fallback(pool: Iterable[int], default: list[int]) -> list[int]:
        seq = list(pool)
        return seq[:1] if seq else default.copy()

    val = _take_or_fallback(remaining[:1], train)
    test = _take_or_fallback(remaining[1:2], val)
    return train, val, test


class LentilsAnomoly(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 2):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size

        self.train_ds = None
        self.vald_ds = None
        self.test_ds = None

    def prepare_data(self):
        PublicDatasets().download_dataset("Lentils", download_path=str(self.data_dir))

    def setup(self, stage=None):
        cu3s_path, label_path = _resolve_lentils_assets(self.data_dir)
        session = cuvis.SessionFile(str(cu3s_path))
        total_measurements = len(session)

        train_idx, val_idx, test_idx = _bucket_measurements(total_measurements)

        if stage == "fit" or stage is None:
            self.train_ds = SingleCu3sDataset(
                cu3s_path=str(cu3s_path),
                label_path=str(label_path),
                processing_mode="Reflectance",
                measurement_indices=train_idx,
            )
            self.vald_ds = SingleCu3sDataset(
                cu3s_path=str(cu3s_path),
                label_path=str(label_path),
                processing_mode="Reflectance",
                measurement_indices=val_idx,
            )

        if stage == "test" or stage is None:
            self.test_ds = SingleCu3sDataset(
                cu3s_path=str(cu3s_path),
                label_path=str(label_path),
                processing_mode="Reflectance",
                measurement_indices=test_idx,
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.vald_ds, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, batch_size=self.batch_size)
