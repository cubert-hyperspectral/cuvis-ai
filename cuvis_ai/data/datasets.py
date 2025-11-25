import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import cuvis
import numpy as np
from loguru import logger
from skimage.draw import polygon2mask
from torch.utils.data import Dataset

from cuvis_ai.data.coco_labels import Annotation, COCOData, RLE2mask
from cuvis_ai.utils.general import _resolve_measurement_indices, normalize_per_channel_vectorized


class SingleCu3sDataset(Dataset):
    """Load cube frames from .cu3s sessions with optional COCO-derived masks."""

    def __init__(
        self,
        cu3s_path: str,
        label_path: str | None = None,
        processing_mode: cuvis.ProcessingMode | str | None = "Raw",
        measurement_indices: Sequence[int] | Iterable[int] | None = None,
    ) -> None:
        self.cu3s_path = cu3s_path
        assert os.path.exists(cu3s_path), f"Dataset path does not exist: {cu3s_path}"
        assert Path(cu3s_path).suffix == ".cu3s", (
            f"Dataset path must point to a .cu3s file: {cu3s_path}"
        )

        self.session = cuvis.SessionFile(cu3s_path)
        self.pc = cuvis.ProcessingContext(self.session)

        has_white_ref = self.session.get_reference(0, cuvis.ReferenceType.White) is not None
        has_dark_ref = self.session.get_reference(0, cuvis.ReferenceType.Dark) is not None
        if processing_mode is not None:
            if isinstance(processing_mode, str):
                processing_mode = getattr(cuvis.ProcessingMode, processing_mode, "Raw")

            if processing_mode == cuvis.ProcessingMode.Reflectance:
                assert has_white_ref and has_dark_ref, (
                    "Reflectance processing mode requires both White and Dark references "
                    "in the cu3s file."
                )
            self.pc.processing_mode = processing_mode

        mesu0 = self.session.get_measurement(0)
        self.num_channels = mesu0.cube.channels
        self.wavelengths = np.array(mesu0.cube.wavelength).ravel()
        self._total_measurements = len(self.session)
        self.measurement_indices = _resolve_measurement_indices(
            measurement_indices, max_index=self._total_measurements
        )
        # Backwards compatibility for legacy callers.
        self.mes_ids = self.measurement_indices

        logger.info(
            f"Loaded cu3s dataset from {cu3s_path} with {len(self.measurement_indices)} "
            f"measurements: {self.measurement_indices}"
        )
        self.has_labels = (
            label_path is not None
            and Path(label_path).exists()
            or Path(cu3s_path).with_suffix(".json").exists()
        )
        if self.has_labels and label_path is None:
            # Sane fallback: label file is named the same as the Session File
            label_path = Path(cu3s_path).with_suffix(".json")
        self._coco: COCOData | None = None
        self.class_labels: dict[int, str] | None = None
        if self.has_labels:
            try:
                self._coco = COCOData.from_path(label_path)
            except Exception as e:
                logger.warning(f"Could not load annotation for {label_path}:", e)
                self.has_labels = False
            logger.info(f"Category map: {self._coco.category_id_to_name}")
            self.class_labels = self._coco.category_id_to_name

    def __len__(self) -> int:
        return len(self.measurement_indices)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | int]:
        mesu_index = self.measurement_indices[idx]
        mesu = self.session.get_measurement(mesu_index)  # starts the cound from 0
        if "cube" not in mesu.data:
            mesu = self.pc.apply(mesu)
        cube_array = mesu.cube.array.astype(np.float32)

        cube_array_norm = normalize_per_channel_vectorized(cube_array, 0.0, 1.0)

        out: dict[str, np.ndarray | int] = {"cube": cube_array_norm, "mesu_index": mesu_index}

        if self.has_labels and self._coco is not None:
            # Check if we have a valid COCO image_id for this frame index
            if mesu_index in self._coco.image_ids:
                image_id = self._coco.image_ids[self._coco.image_ids.index(mesu_index)]
                anns = self._coco.annotations.where(image_id=image_id)
                category_mask = create_mask(
                    annotations=anns,
                    image_height=cube_array.shape[0],
                    image_width=cube_array.shape[1],
                )
            else:
                # Frame index not in available annotations
                category_mask = np.zeros(cube_array.shape[:2], dtype=np.int32)
            out["mask"] = category_mask

        return out


def create_mask(
    annotations: Iterable[Annotation],
    image_height: int,
    image_width: int,
    overlap_strategy: str = "overwrite",
) -> np.ndarray:
    category_mask = np.zeros((image_height, image_width), dtype=np.int32)
    for ann in annotations:
        segs = ann.segmentation
        mask = ann.mask
        cat_id = int(ann.category_id)
        if not segs and not mask:
            continue

        if isinstance(segs, list) and len(segs) > 0 and isinstance(segs[0], (list, tuple)):
            for seg in segs:
                if len(seg) < 6:
                    continue
                xy = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
                # polygon2mask expects coords in (row, col) format and returns a filled boolean mask
                poly_mask = polygon2mask(
                    (image_height, image_width), xy[:, [1, 0]]
                )  # Swap x,y to row,col
                if overlap_strategy == "overwrite":
                    category_mask[poly_mask] = cat_id
                else:
                    write_idx = poly_mask & (category_mask == 0)
                    category_mask[write_idx] = cat_id
        if isinstance(mask, dict) and len(mask.get("counts", lambda: [])) > 0:
            mask_width, mask_height = mask.get("size")
            # decode RLE mask
            decoded = RLE2mask(mask.get("counts"), mask_width=mask_width, mask_height=mask_height)

            if overlap_strategy == "overwrite":
                write_mask = decoded
            else:
                write_mask = decoded & (category_mask == 0)
            category_mask[write_mask] = cat_id

    return category_mask
