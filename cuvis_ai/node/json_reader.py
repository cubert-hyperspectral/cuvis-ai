"""JSON source nodes for detections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec


class DetectionJsonReader(Node):
    """Read COCO detection JSON and emit tensors per frame.

    Outputs per call:
      - frame_id: int64 [1]
      - bboxes: float32 [1, N, 4] (xyxy)
      - category_ids: int64 [1, N]
      - confidences: float32 [1, N]
      - orig_hw: int64 [1, 2]
    """

    OUTPUT_SPECS = {
        "frame_id": PortSpec(dtype=torch.int64, shape=(1,), description="Frame index."),
        "bboxes": PortSpec(dtype=torch.float32, shape=(1, -1, 4), description="Boxes [1,N,4] xyxy"),
        "category_ids": PortSpec(
            dtype=torch.int64, shape=(1, -1), description="Category IDs [1,N]"
        ),
        "confidences": PortSpec(dtype=torch.float32, shape=(1, -1), description="Scores [1,N]"),
        "orig_hw": PortSpec(dtype=torch.int64, shape=(1, 2), description="Original [H,W]"),
    }

    def __init__(self, json_path: str, **kwargs: Any) -> None:
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON not found: {self.json_path}")

        with self.json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self._images = {int(img["id"]): img for img in data.get("images", [])}
        self._annotations_by_img: dict[int, list[dict[str, Any]]] = {}
        for ann in data.get("annotations", []):
            self._annotations_by_img.setdefault(int(ann["image_id"]), []).append(ann)

        self._frame_ids = sorted(self._images.keys())
        self._cursor = 0

        super().__init__(json_path=str(self.json_path), **kwargs)

    def reset(self) -> None:  # noqa: D401
        """Rewind to the first frame."""
        self._cursor = 0

    def forward(self, context: Context | None = None, **_: Any) -> dict[str, Any]:  # noqa: ARG002
        if self._cursor >= len(self._frame_ids):
            raise StopIteration("No more frames in detection JSON")

        frame_id = self._frame_ids[self._cursor]
        self._cursor += 1

        img = self._images[frame_id]
        anns = self._annotations_by_img.get(frame_id, [])

        bboxes = []
        cats = []
        scores = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            bboxes.append([x, y, x + w, y + h])
            cats.append(int(ann.get("category_id", 0)))
            scores.append(float(ann.get("score", 0.0)))

        bboxes_t = (
            torch.tensor([bboxes], dtype=torch.float32)
            if bboxes
            else torch.empty((1, 0, 4), dtype=torch.float32)
        )
        cats_t = (
            torch.tensor([cats], dtype=torch.int64)
            if cats
            else torch.empty((1, 0), dtype=torch.int64)
        )
        scores_t = (
            torch.tensor([scores], dtype=torch.float32)
            if scores
            else torch.empty((1, 0), dtype=torch.float32)
        )

        h = int(img.get("height", 0))
        w = int(img.get("width", 0))
        orig_hw = torch.tensor([[h, w]], dtype=torch.int64)

        return {
            "frame_id": torch.tensor([frame_id], dtype=torch.int64),
            "bboxes": bboxes_t,
            "category_ids": cats_t,
            "confidences": scores_t,
            "orig_hw": orig_hw,
        }


__all__ = ["DetectionJsonReader"]
