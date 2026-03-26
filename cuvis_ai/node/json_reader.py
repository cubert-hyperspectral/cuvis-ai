"""JSON source nodes for detections and tracking results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cuvis_ai_core.data.rle import coco_rle_decode
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger


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
        """Emit detections for the next frame in the detection JSON stream."""
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
            category_id = ann.get("category_id", 0)
            score = ann.get("score", 0.0)
            cats.append(int(category_id) if category_id is not None else 0)
            scores.append(float(score) if score is not None else 0.0)

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


class TrackingResultsReader(Node):
    """Read tracking results JSON (bbox or mask format) and emit per-frame tensors.

    Supports two JSON formats:

    1. **COCO bbox tracking** — ``images`` + ``annotations`` with ``bbox`` and
       ``track_id`` fields.  Emits ``bboxes``, ``category_ids``, ``confidences``,
       ``track_ids``.
    2. **Video COCO** — ``videos`` + ``annotations`` with ``segmentations``
       list of RLE dicts.  Emits ``mask`` label map and ``object_ids``.

    Optional outputs are ``None`` when the format doesn't provide them.

    **Frame synchronization**: When the optional ``frame_id`` input is connected
    (e.g. from ``CU3SDataNode.mesu_index``), the reader looks up detections for
    that specific frame instead of cursor-advancing.  This guarantees that the
    emitted bboxes/masks correspond to the same frame as the cube data.  When
    ``frame_id`` is not connected, the reader uses the internal cursor (legacy
    behavior).
    """

    INPUT_SPECS = {
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(1,),
            description=(
                "Frame index to look up. When connected, disables cursor "
                "and emits detections for this specific frame."
            ),
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "frame_id": PortSpec(dtype=torch.int64, shape=(1,), description="Frame index."),
        "orig_hw": PortSpec(dtype=torch.int64, shape=(1, 2), description="Frame [H, W]."),
        "bboxes": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, 4),
            description="Boxes [1,N,4] xyxy. Present in bbox-tracking format.",
            optional=True,
        ),
        "category_ids": PortSpec(
            dtype=torch.int64,
            shape=(1, -1),
            description="Category IDs [1,N]. Present in bbox-tracking format.",
            optional=True,
        ),
        "confidences": PortSpec(
            dtype=torch.float32,
            shape=(1, -1),
            description="Scores [1,N]. Present in bbox-tracking format.",
            optional=True,
        ),
        "track_ids": PortSpec(
            dtype=torch.int64,
            shape=(1, -1),
            description="Track IDs [1,N]. Present in bbox-tracking format.",
            optional=True,
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(1, -1, -1),
            description="Label map [1,H,W]; pixel = object ID. Present in mask formats.",
            optional=True,
        ),
        "object_ids": PortSpec(
            dtype=torch.int64,
            shape=(1, -1),
            description="Active object IDs [1,N]. Present in mask formats.",
            optional=True,
        ),
    }

    def __init__(
        self,
        json_path: str,
        required_format: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON not found: {self.json_path}")
        if required_format is not None and required_format not in {"coco_bbox", "video_coco"}:
            raise ValueError(
                "required_format must be one of {'coco_bbox', 'video_coco'} when provided."
            )

        with self.json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Detect format and build per-frame lookup
        if "videos" in data and "annotations" in data:
            self._format = "video_coco"
            self._init_video_coco(data)
        elif "images" in data and "annotations" in data:
            self._format = "coco_bbox"
            self._init_coco_bbox(data)
        else:
            raise ValueError(
                f"Unsupported tracking JSON format in {self.json_path}. "
                "Expected COCO bbox (images+annotations) "
                "or video COCO (videos+annotations)."
            )

        self._required_format = required_format
        self._format_mismatch_msg: str | None = None
        if self._required_format is not None and self._format != self._required_format:
            self._format_mismatch_msg = (
                f"Tracking JSON format is '{self._format}', "
                f"but required_format is '{self._required_format}'."
            )

        self._cursor = 0
        logger.info(
            "[TrackingResultsReader] format={}, required_format={}, frames={}, path={}",
            self._format,
            self._required_format,
            len(self._frame_ids),
            self.json_path,
        )

        super().__init__(json_path=str(self.json_path), required_format=required_format, **kwargs)

    # -- Format-specific init --------------------------------------------------

    def _init_coco_bbox(self, data: dict) -> None:
        """Index COCO bbox annotations by image ID for per-frame lookup."""
        self._images = {int(img["id"]): img for img in data.get("images", [])}
        self._annotations_by_img: dict[int, list[dict]] = {}
        for ann in data.get("annotations", []):
            self._annotations_by_img.setdefault(int(ann["image_id"]), []).append(ann)
        self._frame_ids = sorted(self._images.keys())

    def _init_video_coco(self, data: dict) -> None:
        """Index SAM-style video COCO segmentations by frame and track ID."""
        videos = data.get("videos", [])
        annotations = data.get("annotations", [])
        if not videos:
            raise ValueError("SAM3 video COCO JSON missing 'videos' entries")

        video = videos[0]
        frame_indices = video.get("frame_indices")
        if frame_indices is None:
            start_frame = int(video.get("start_frame", 0))
            length = int(video.get("length", 0))
            if length > 0:
                frame_indices = list(range(start_frame, start_frame + length))
            else:
                max_len = max(
                    (len(ann.get("segmentations", [])) for ann in annotations),
                    default=0,
                )
                frame_indices = list(range(start_frame, start_frame + max_len))

        self._mask_data: dict[int, dict[int, dict]] = {}
        self._mask_hw: tuple[int, int] = (0, 0)
        for ann in annotations:
            track_id = int(ann.get("track_id", ann.get("id", 0)))
            for idx, seg in enumerate(ann.get("segmentations", [])):
                if seg is None or idx >= len(frame_indices):
                    continue
                fi = int(frame_indices[idx])
                self._mask_data.setdefault(fi, {})[track_id] = seg
                if self._mask_hw == (0, 0):
                    size = seg.get("size", [0, 0])
                    self._mask_hw = (int(size[0]), int(size[1]))

        self._frame_ids = sorted({int(fi) for fi in frame_indices} | set(self._mask_data.keys()))

    # -- Common methods --------------------------------------------------------

    @property
    def num_frames(self) -> int:
        """Return the number of frames addressable by this reader."""
        return len(self._frame_ids)

    @property
    def format(self) -> str:
        """Return the detected tracking JSON format identifier."""
        return self._format

    def reset(self) -> None:
        """Rewind sequential reads to the first available frame."""
        self._cursor = 0

    def forward(
        self,
        frame_id: torch.Tensor | None = None,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, Any]:
        """Emit tracking tensors for an explicit frame or the next cursor frame."""
        if self._format_mismatch_msg is not None:
            raise ValueError(self._format_mismatch_msg)

        if frame_id is not None:
            # Lookup mode: emit detections for the requested frame
            fid = int(frame_id.item())
        else:
            # Cursor mode (legacy): advance cursor sequentially
            if self._cursor >= len(self._frame_ids):
                raise StopIteration("No more frames in tracking JSON")
            fid = self._frame_ids[self._cursor]
            self._cursor += 1

        if self._format == "coco_bbox":
            return self._emit_coco_bbox(fid)
        else:
            return self._emit_video_coco(fid)

    def _emit_coco_bbox(self, frame_id: int) -> dict[str, Any]:
        """Build bbox-style tracking outputs for one frame."""
        img = self._images.get(frame_id)
        empty_mask = torch.empty((1, 0, 0), dtype=torch.int32)
        empty_oids = torch.empty((1, 0), dtype=torch.int64)
        if img is None:
            # Frame not in JSON — return empty detections
            return {
                "frame_id": torch.tensor([frame_id], dtype=torch.int64),
                "orig_hw": torch.tensor([[0, 0]], dtype=torch.int64),
                "bboxes": torch.empty((1, 0, 4), dtype=torch.float32),
                "category_ids": torch.empty((1, 0), dtype=torch.int64),
                "confidences": torch.empty((1, 0), dtype=torch.float32),
                "track_ids": torch.empty((1, 0), dtype=torch.int64),
                "mask": empty_mask,
                "object_ids": empty_oids,
            }
        anns = self._annotations_by_img.get(frame_id, [])

        bboxes, cats, scores, tids = [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            bboxes.append([x, y, x + w, y + h])
            category_id = ann.get("category_id", 0)
            score = ann.get("score", 0.0)
            track_id = ann.get("track_id", -1)
            cats.append(int(category_id) if category_id is not None else 0)
            scores.append(float(score) if score is not None else 0.0)
            tids.append(int(track_id) if track_id is not None else -1)

        n = len(bboxes)
        bboxes_t = (
            torch.tensor([bboxes], dtype=torch.float32)
            if n
            else torch.empty((1, 0, 4), dtype=torch.float32)
        )
        cats_t = (
            torch.tensor([cats], dtype=torch.int64) if n else torch.empty((1, 0), dtype=torch.int64)
        )
        scores_t = (
            torch.tensor([scores], dtype=torch.float32)
            if n
            else torch.empty((1, 0), dtype=torch.float32)
        )
        tids_t = (
            torch.tensor([tids], dtype=torch.int64) if n else torch.empty((1, 0), dtype=torch.int64)
        )

        h_px = int(img.get("height", 0))
        w_px = int(img.get("width", 0))

        return {
            "frame_id": torch.tensor([frame_id], dtype=torch.int64),
            "orig_hw": torch.tensor([[h_px, w_px]], dtype=torch.int64),
            "bboxes": bboxes_t,
            "category_ids": cats_t,
            "confidences": scores_t,
            "track_ids": tids_t,
            "mask": empty_mask,
            "object_ids": empty_oids,
        }

    def _rles_to_label_map(
        self,
        obj_rles: dict[int, dict],
        h: int,
        w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert {obj_id: rle} to label map [1,H,W] and object_ids [1,N]."""
        label_map = np.zeros((h, w), dtype=np.int32)
        obj_ids = sorted(obj_rles.keys())
        for oid in obj_ids:
            binary = coco_rle_decode(obj_rles[oid])
            label_map[binary > 0] = oid

        mask_t = torch.from_numpy(label_map).unsqueeze(0)  # [1, H, W]
        oids_t = (
            torch.tensor([obj_ids], dtype=torch.int64)
            if obj_ids
            else torch.empty((1, 0), dtype=torch.int64)
        )
        return mask_t, oids_t

    def _emit_video_coco(self, frame_id: int) -> dict[str, Any]:
        """Build mask-style tracking outputs for one frame."""
        obj_rles = self._mask_data.get(frame_id, {})
        h, w = self._mask_hw

        if (h == 0 or w == 0) and obj_rles:
            first_rle = next(iter(obj_rles.values()))
            size = first_rle.get("size", [0, 0])
            h, w = int(size[0]), int(size[1])

        if h == 0 or w == 0:
            mask_t = torch.zeros((1, 1, 1), dtype=torch.int32)
            oids_t = torch.empty((1, 0), dtype=torch.int64)
        else:
            mask_t, oids_t = self._rles_to_label_map(obj_rles, h, w)

        return {
            "frame_id": torch.tensor([frame_id], dtype=torch.int64),
            "orig_hw": torch.tensor([[h, w]], dtype=torch.int64),
            "bboxes": torch.empty((1, 0, 4), dtype=torch.float32),
            "category_ids": torch.empty((1, 0), dtype=torch.int64),
            "confidences": torch.empty((1, 0), dtype=torch.float32),
            "track_ids": torch.empty((1, 0), dtype=torch.int64),
            "mask": mask_t,
            "object_ids": oids_t,
        }


__all__ = ["DetectionJsonReader", "TrackingResultsReader"]
