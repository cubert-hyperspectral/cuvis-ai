from __future__ import annotations

from pathlib import Path

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


class LentilsAnomaly(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 2,
        train_ids: list[int] = None,  # [0, 1, 2],
        val_ids: list[int] = None,  # [3, 4, 5],
        test_ids: list[int] = None,  # [9, 10, 11, 12, 13],  # Have no labels
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size

        self.train_ds = None
        self.vald_ds = None
        self.test_ds = None
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids

    def prepare_data(self) -> None:
        PublicDatasets().download_dataset("Lentils", download_path=str(self.data_dir))

    def setup(self, stage: str | None = None) -> None:
        cu3s_path, label_path = _resolve_lentils_assets(self.data_dir)
        # session = cuvis.SessionFile(str(cu3s_path))

        if stage == "fit" or stage is None:
            self.train_ds = SingleCu3sDataset(
                cu3s_path=str(cu3s_path),
                label_path=str(label_path),
                processing_mode="Reflectance",
                measurement_indices=self.train_ids,
            )
            self.val_ds = SingleCu3sDataset(
                cu3s_path=str(cu3s_path),
                label_path=str(label_path),
                processing_mode="Reflectance",
                measurement_indices=self.val_ids,
            )

        if stage == "test" or stage is None:
            self.test_ds = SingleCu3sDataset(
                cu3s_path=str(cu3s_path),
                label_path=str(label_path),
                processing_mode="Reflectance",
                measurement_indices=self.test_ids,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, shuffle=False, batch_size=self.batch_size)
