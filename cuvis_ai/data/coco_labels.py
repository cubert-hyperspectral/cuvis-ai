import json
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dataclass_wizard import JSONWizard
from pycocotools.coco import COCO
from skimage.draw import polygon2mask
from torchvision.tv_tensors import BoundingBoxes, Mask


def RLE2mask(rle: list, mask_width: int, mask_height: int) -> np.ndarray:
    mask = np.zeros(mask_width * mask_height, np.uint8)
    ids = 0
    value = 0
    for c in rle:
        mask[ids : ids + c] = value
        value = not value
        ids += c
    mask = mask.reshape((mask_height, mask_width), order="F")
    return mask.astype(bool, copy=False)


class SafeWizard(JSONWizard):
    """
    JSONWizard subclass that safely converts dataclasses to dicts,
    keeping non-serializable objects (e.g., torch Tensors, Masks)
    as-is instead of falling back to string representations.
    """

    def to_dict_safe(self):
        """
        Like `to_dict()`, but leaves unsupported types untouched.
        """
        base_dict = super().to_dict()
        final_dict = {}

        for key, value in vars(self).items():
            if not self._is_json_serializable(value):
                # keep original object (Mask, Tensor, etc.)
                final_dict[key] = value
                continue
            val = base_dict.get(key, value)
            final_dict[key] = val
        return final_dict

    @staticmethod
    def _is_json_serializable(obj):
        try:
            json.dumps(obj)
            return True
        except Exception:
            return False


@dataclass
class Info(JSONWizard):
    description: str | None = None
    url: str | None = None
    version: int | None = None
    contributor: str | None = None
    date_created: str | None = None


@dataclass
class License(JSONWizard):
    id: int
    name: str
    url: str | None = None


@dataclass
class Category(JSONWizard):
    id: int
    name: str
    supercategory: str | None = None


@dataclass
class Image(JSONWizard):
    id: int
    file_name: str
    height: int
    width: int
    license: int | None = None
    flickr_url: str | None = None
    coco_url: str | None = None
    date_captured: str | None = None
    wavelength: list[float] | None = field(default_factory=list)


@dataclass
class Annotation(SafeWizard):
    id: int
    image_id: int
    category_id: int
    segmentation: list | None = None
    area: float | None = None
    bbox: list[float] | None = None
    mask: dict | None = None
    iscrowd: int | None = 0
    auxiliary: dict[str, Any] | None = field(default_factory=dict)

    def to_dict_safe(self):
        """
        Like `to_dict()`, but leaves unsupported types untouched.
        """
        base_dict = super().to_dict()
        final_dict = {}

        for key, value in vars(self).items():
            if not self._is_json_serializable(value):
                # keep original object (Mask, Tensor, etc.)
                final_dict[key] = value
                continue
            val = base_dict.get(key, value)
            final_dict[key] = val
        return final_dict

    @staticmethod
    def _is_json_serializable(obj):
        try:
            json.dumps(obj)
            return True
        except Exception:
            return False

    def to_torchvision(self, size):
        """Convert COCO-style bbox/segmentation/mask into torchvision tensors."""
        out = copy(self)

        if self.bbox is not None:
            out.bbox = BoundingBoxes(
                torch.tensor([self.bbox], dtype=torch.float32), format="XYWH", canvas_size=size
            )

        if (
            self.segmentation is not None
            and isinstance(self.segmentation, list)
            and self.segmentation != []
        ):
            coords = np.array(self.segmentation[0]).reshape(-1, 2)
            mask_np = polygon2mask(size, coords).astype(np.uint8)
            out.segmentation = Mask(torch.from_numpy(mask_np))

        if self.mask is not None:
            mask_np = RLE2mask(self.mask["counts"], self.mask["size"])
            out.mask = Mask(torch.from_numpy(mask_np))

        return out.to_dict_safe()
        # return out


class QueryableList:
    def __init__(self, items: list[Any]):
        self._items = items

    def where(self, **conditions):
        """
        Filter items based on conditions.
        :param conditions: Keyword arguments representing field=value filters.
        :return: A new QueryableList with filtered items.
        """
        filtered_items = self._items
        for key, value in conditions.items():
            filtered_items = [item for item in filtered_items if getattr(item, key) == value]
        return list(filtered_items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


class COCOData:
    def __init__(self, coco: COCO):
        self._coco = coco
        self._image_ids = None
        self._categories = None
        self._category_id_to_name = None
        self._annotations = None
        self._images = None

    @classmethod
    def from_path(cls, path: Path | str):
        return cls(COCO(path))

    @property
    def image_ids(self) -> list[int]:
        if self._image_ids is None:
            self._image_ids = sorted(self._coco.imgs.keys())
        return self._image_ids

    @property
    def info(self) -> Info:
        return Info.from_dict(self._coco.dataset["info"])

    @property
    def license(self) -> License:
        return Info.from_dict(self._coco.dataset["licenses"])

    @property
    def annotations(self) -> QueryableList:
        if self._annotations is None:
            self._annotations = QueryableList(
                [Annotation.from_dict(v) for v in self._coco.anns.values()]
            )
        return self._annotations

    @property
    def categories(self) -> list[Category]:
        if self._categories is None:
            self._categories = [Category.from_dict(v) for v in self._coco.cats.values()]
        return self._categories

    @property
    def category_id_to_name(self) -> dict[int, str]:
        if self._category_id_to_name is None:
            self._category_id_to_name = {cat.id: cat.name for cat in self.categories}
        return self._category_id_to_name

    @property
    def images(self) -> list[Image]:
        if self._images is None:
            self._images = [Image.from_dict(v) for v in self._coco.imgs.values()]
        return self._images

    def save(self, path: str | Path):
        """
        Save the current COCOData object (images, annotations, categories, etc.)
        back into a COCO-style JSON file.

        Automatically converts dataclasses to plain dicts and ensures
        compliance with standard COCO structure.
        """
        path = str(path)
        dataset = {
            "info": self.info.to_dict() if hasattr(self, "info") else {},
            "licenses": [lic.to_dict() for lic in self._coco.dataset.get("licenses", [])]
            if "licenses" in self._coco.dataset
            else [],
            "images": [img.to_dict() for img in self.images],
            "annotations": [],
            "categories": [cat.to_dict() for cat in self.categories],
        }

        for ann in self.annotations:
            if isinstance(ann, Annotation):
                dataset["annotations"].append(ann.to_dict_safe())
            elif isinstance(ann, dict):
                dataset["annotations"].append(ann)
            else:
                raise TypeError(f"Unsupported annotation type: {type(ann)}")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

        print(f"COCOData saved successfully to: {path}")


if __name__ == "__main__":
    import os
    from pathlib import Path

    import cuvis

    session_file_path = (
        "C:\\Users\\nima.ghorbani\\code-repos\\cuvis.ai.examples\\data\\Lentils\\Lentils_000.cu3s"
    )
    session = cuvis.SessionFile(session_file_path)
    measurement = session.get_measurement(0)
    labelpath = Path(session_file_path).with_suffix(".json")
    assert os.path.exists(labelpath), f"Label file not found: {labelpath}"

    canvas_size = measurement.cube.width, measurement.cube.height
    coco = COCOData.from_path(str(labelpath))
    print("Categories:", coco.category_id_to_name)

    anns = coco.annotations.where(image_id=coco.image_ids[0])[0]
    labels = anns.to_torchvision(canvas_size)

    print(labels["segmentation"])
