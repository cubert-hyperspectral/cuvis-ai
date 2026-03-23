"""Multi-file .cu3s dataset and data module for group-aware training.

Handles data layouts where each frame is a separate .cu3s file in a directory,
with a COCO-format annotation JSON per day. Supports group-constrained
train/val/test splitting (e.g., keeping groups of 4 consecutive frames together).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cuvis
import numpy as np
import pytorch_lightning as pl
from loguru import logger
from skimage.draw import polygon2mask
from torch.utils.data import DataLoader, Dataset


def _extract_frame_number(stem: str) -> int:
    """Extract the trailing integer from a filename stem like 'Auto_000_000123'."""
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else 0


def _parse_coco_json(json_path: Path) -> dict[str, Any]:
    """Parse a COCO annotation JSON and return structured data."""
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image[int(ann["image_id"])].append(ann)

    cat_map = {int(c["id"]): str(c["name"]) for c in data.get("categories", [])}
    return {"anns_by_image": dict(anns_by_image), "cat_map": cat_map}


def _build_category_mask(anns: list[dict[str, Any]], h: int, w: int) -> np.ndarray:
    """Build an [H, W] int32 category mask from COCO polygon annotations."""
    mask = np.zeros((h, w), dtype=np.int32)
    for ann in anns:
        cat_id = int(ann.get("category_id", 0))
        segs = ann.get("segmentation", [])
        if not isinstance(segs, list):
            continue
        for seg in segs:
            if not isinstance(seg, list) or len(seg) < 6:
                continue
            xy = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
            try:
                poly = polygon2mask((h, w), xy[:, [1, 0]])
                mask[poly] = cat_id
            except Exception:
                continue
    return mask


class MultiFileCu3sDataset(Dataset):
    """Dataset loading individual .cu3s frame files with COCO annotations.

    Each item is one hyperspectral frame loaded from a separate .cu3s file.
    Annotations are parsed from a per-day COCO JSON.

    Parameters
    ----------
    frame_records : list[dict]
        Each dict has keys: ``cu3s_path`` (str/Path), ``annotation_json`` (str/Path),
        ``image_id`` (int).
    processing_mode : str
        Cuvis processing mode: ``"Reflectance"`` or ``"Raw"``.
    """

    def __init__(
        self,
        frame_records: list[dict[str, Any]],
        processing_mode: str = "Reflectance",
    ) -> None:
        self.records = frame_records
        self.processing_mode = processing_mode

        # Pre-parse all unique annotation JSONs (typically one per day)
        self._ann_cache: dict[str, dict[str, Any]] = {}
        for rec in self.records:
            jp = str(rec["annotation_json"])
            if jp and jp not in self._ann_cache:
                self._ann_cache[jp] = _parse_coco_json(Path(jp))

        # Read wavelengths from the first file to expose as a property
        if self.records:
            first_sess = cuvis.SessionFile(str(self.records[0]["cu3s_path"]))
            mesu0 = first_sess.get_measurement(0)
            self.wavelengths_nm = np.array(mesu0.cube.wavelength, dtype=np.int32).ravel()
            self.num_channels = mesu0.cube.channels
        else:
            self.wavelengths_nm = np.array([], dtype=np.int32)
            self.num_channels = 0

    def __len__(self) -> int:
        """Return the number of frames in the dataset."""
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | int]:
        """Load one frame (cube + COCO-derived mask) and its metadata."""
        rec = self.records[idx]
        cu3s_path = str(rec["cu3s_path"])
        image_id = int(rec["image_id"])

        session = cuvis.SessionFile(cu3s_path)
        pc = cuvis.ProcessingContext(session)

        pm = getattr(cuvis.ProcessingMode, self.processing_mode, cuvis.ProcessingMode.Raw)
        try:
            pc.processing_mode = pm
        except Exception:
            pc.processing_mode = cuvis.ProcessingMode.Raw

        mesu = session.get_measurement(0)
        if "cube" not in mesu.data:
            mesu = pc.apply(mesu)

        cube: np.ndarray = mesu.cube.array
        wavelengths = np.array(mesu.cube.wavelength, dtype=np.int32).ravel()

        # Build category mask from annotations
        jp = str(rec["annotation_json"])
        if jp and jp in self._ann_cache:
            anns = self._ann_cache[jp]["anns_by_image"].get(image_id, [])
        else:
            anns = []
        mask = _build_category_mask(anns, cube.shape[0], cube.shape[1])

        return {
            "cube": cube,
            "mask": mask,
            "wavelengths": wavelengths,
            "mesu_index": image_id,
        }


class MultiFileCu3sDataModule(pl.LightningDataModule):
    """DataModule for multi-file .cu3s datasets with group-aware train/val/test splits.

    Parameters
    ----------
    splits_csv : str | Path
        Path to the CSV produced by ``lentils_splits.py``. Must have columns:
        ``split, cu3s_path, annotation_json, image_id``.
    batch_size : int
        Batch size for all dataloaders.
    processing_mode : str
        ``"Reflectance"`` or ``"Raw"``.
    """

    def __init__(
        self,
        splits_csv: str | Path,
        batch_size: int = 4,
        processing_mode: str = "Reflectance",
    ) -> None:
        super().__init__()
        self.splits_csv = Path(splits_csv)
        self.batch_size = batch_size
        self.processing_mode = processing_mode
        self.train_ds: MultiFileCu3sDataset | None = None
        self.val_ds: MultiFileCu3sDataset | None = None
        self.test_ds: MultiFileCu3sDataset | None = None

    def _load_records(self) -> dict[str, list[dict[str, Any]]]:
        """Parse the splits CSV into per-split frame record lists."""
        import csv

        records: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        with self.splits_csv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = row["split"]
                if split not in records:
                    continue
                records[split].append(
                    {
                        "cu3s_path": row["cu3s_path"],
                        "annotation_json": row["annotation_json"],
                        "image_id": int(row["image_id"]),
                    }
                )
        return records

    def setup(self, stage: str | None = None) -> None:
        """Create train/val/test datasets for the requested Lightning stage."""
        records = self._load_records()
        if stage == "fit" or stage is None:
            if records["train"]:
                self.train_ds = MultiFileCu3sDataset(
                    records["train"],
                    processing_mode=self.processing_mode,
                )
                logger.info(f"Train dataset: {len(self.train_ds)} frames")
            if records["val"]:
                self.val_ds = MultiFileCu3sDataset(
                    records["val"],
                    processing_mode=self.processing_mode,
                )
                logger.info(f"Val dataset: {len(self.val_ds)} frames")

        if stage == "test" or stage is None:
            if records["test"]:
                self.test_ds = MultiFileCu3sDataset(
                    records["test"],
                    processing_mode=self.processing_mode,
                )
                logger.info(f"Test dataset: {len(self.test_ds)} frames")

    def train_dataloader(self) -> DataLoader:
        """Return a DataLoader for the training split."""
        if self.train_ds is None:
            raise RuntimeError("Train dataset not initialized. Call setup('fit').")
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader:
        """Return a DataLoader for the validation split."""
        if self.val_ds is None:
            raise RuntimeError("Val dataset not initialized. Call setup('fit').")
        return DataLoader(
            self.val_ds,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=0,
        )

    def test_dataloader(self) -> DataLoader:
        """Return a DataLoader for the test split."""
        if self.test_ds is None:
            raise RuntimeError("Test dataset not initialized. Call setup('test').")
        return DataLoader(
            self.test_ds,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=0,
        )
