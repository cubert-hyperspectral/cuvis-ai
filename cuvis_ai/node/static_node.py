"""Static nodes and helpers for frame-indexed mask prompt schedules."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from cuvis_ai_core.data.rle import coco_rle_decode
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec

_PROMPT_SPEC_RE = re.compile(r"\s*(\d+):(\d+)@(\d+)\s*")


@dataclass(frozen=True)
class MaskPromptSpec:
    """One prompt-schedule entry."""

    object_id: int
    detection_id: int
    frame_id: int
    order: int


def parse_mask_prompt_spec(spec: str, order: int = 0) -> MaskPromptSpec:
    """Parse ``<object_id>:<detection_id>@<frame_id>`` into a typed spec."""
    match = _PROMPT_SPEC_RE.fullmatch(spec)
    if match is None:
        raise ValueError(
            f"Invalid prompt spec '{spec}'. Expected format <object_id>:<detection_id>@<frame_id>."
        )
    return MaskPromptSpec(
        object_id=int(match.group(1)),
        detection_id=int(match.group(2)),
        frame_id=int(match.group(3)),
        order=int(order),
    )


def _image_hw(image_entry: dict[str, Any], frame_id: int) -> tuple[int, int]:
    height = int(image_entry.get("height", 0))
    width = int(image_entry.get("width", 0))
    if height <= 0 or width <= 0:
        raise ValueError(f"Image metadata for frame {frame_id} must include positive height/width.")
    return height, width


def _segmentation_hw(segmentation: Any) -> tuple[int, int] | None:
    if isinstance(segmentation, dict):
        size = segmentation.get("size")
        if isinstance(size, list | tuple) and len(size) == 2:
            height = int(size[0])
            width = int(size[1])
            if height > 0 and width > 0:
                return height, width
    return None


def _select_annotation_for_prompt(
    annotations_by_frame: dict[int, list[dict[str, Any]]],
    detection_id: int,
    frame_id: int,
) -> dict[str, Any]:
    frame_annotations = list(annotations_by_frame.get(int(frame_id), []))
    if not frame_annotations:
        raise ValueError(f"Frame {frame_id} has no annotations in the detection JSON.")

    has_track_ids = any("track_id" in ann for ann in frame_annotations)
    if has_track_ids:
        by_track = {int(ann["track_id"]): ann for ann in frame_annotations if "track_id" in ann}
        if detection_id not in by_track:
            raise ValueError(
                f"Track ID {detection_id} not found on frame {frame_id}. "
                f"Available track_ids: {sorted(by_track)}"
            )
        return by_track[detection_id]

    sorted_annotations = sorted(
        frame_annotations,
        key=lambda ann: float(ann.get("score", 0.0)),
        reverse=True,
    )
    rank = int(detection_id) - 1
    if rank < 0 or rank >= len(sorted_annotations):
        raise ValueError(
            f"Detection rank {detection_id} is out of range on frame {frame_id} "
            f"(have {len(sorted_annotations)} detections)."
        )
    return sorted_annotations[rank]


def _decode_segmentation(
    segmentation: Any,
    image_hw: tuple[int, int],
    *,
    frame_id: int,
) -> np.ndarray:
    height, width = image_hw
    if isinstance(segmentation, dict):
        mask = coco_rle_decode(segmentation)
    elif isinstance(segmentation, list):
        polygons = segmentation
        if polygons and isinstance(polygons[0], (int, float)):
            polygons = [polygons]
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons:
            coords = np.asarray(polygon, dtype=np.float32)
            if coords.size == 0:
                continue
            if coords.size % 2 != 0:
                raise ValueError(
                    f"Polygon segmentation on frame {frame_id} has an odd coord count."
                )
            pts = np.rint(coords.reshape(-1, 2)).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], color=1)
    else:
        raise ValueError(
            f"Unsupported segmentation type on frame {frame_id}: {type(segmentation).__name__}."
        )

    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape != (height, width):
        raise ValueError(
            f"Decoded segmentation shape {mask.shape} does not match image shape {(height, width)} "
            f"on frame {frame_id}."
        )
    return (mask > 0).astype(np.uint8)


def load_mask_prompt_schedule(
    json_path: str | Path,
    prompt_specs: Sequence[str] | None,
) -> tuple[dict[int, np.ndarray], dict[int, tuple[int, int]], tuple[int, int] | None]:
    """Load detection JSON and build per-frame label-map prompts."""
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"Detection JSON not found: {json_file}")

    with json_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    images = {int(img["id"]): img for img in data.get("images", [])}
    if not images:
        raise ValueError(f"Detection JSON {json_file} must contain COCO 'images' entries.")

    annotations_by_frame: dict[int, list[dict[str, Any]]] = {}
    annotation_hw_by_frame: dict[int, set[tuple[int, int]]] = {}
    for annotation in data.get("annotations", []):
        frame_id = int(annotation["image_id"])
        annotations_by_frame.setdefault(frame_id, []).append(annotation)
        segmentation_hw = _segmentation_hw(annotation.get("segmentation"))
        if segmentation_hw is not None:
            annotation_hw_by_frame.setdefault(frame_id, set()).add(segmentation_hw)

    frame_hw_by_id = {frame_id: _image_hw(img, frame_id) for frame_id, img in images.items()}
    for frame_id, seg_hws in annotation_hw_by_frame.items():
        if len(seg_hws) > 1:
            raise ValueError(
                f"Frame {frame_id} has inconsistent segmentation sizes: {sorted(seg_hws)}."
            )
        segmentation_hw = next(iter(seg_hws))
        current_hw = frame_hw_by_id.get(frame_id)
        if current_hw is None or current_hw[0] <= 1 or current_hw[1] <= 1:
            frame_hw_by_id[frame_id] = segmentation_hw

    usable_hws = sorted({hw for hw in frame_hw_by_id.values() if hw[0] > 1 and hw[1] > 1})
    unique_hw = usable_hws or sorted(set(frame_hw_by_id.values()))
    default_hw = unique_hw[0] if len(unique_hw) == 1 else None

    masks_by_frame: dict[int, np.ndarray] = {}
    for order, raw_spec in enumerate(prompt_specs or []):
        spec = parse_mask_prompt_spec(raw_spec, order=order)
        if spec.frame_id not in frame_hw_by_id:
            raise ValueError(
                f"Frame {spec.frame_id} referenced by '{raw_spec}' is missing in images."
            )
        annotation = _select_annotation_for_prompt(
            annotations_by_frame=annotations_by_frame,
            detection_id=spec.detection_id,
            frame_id=spec.frame_id,
        )
        if "segmentation" not in annotation:
            raise ValueError(
                f"Annotation selected by '{raw_spec}' does not contain a 'segmentation' field."
            )

        frame_hw = frame_hw_by_id[spec.frame_id]
        binary_mask = _decode_segmentation(
            annotation["segmentation"],
            frame_hw,
            frame_id=spec.frame_id,
        )
        if int(np.count_nonzero(binary_mask)) == 0:
            raise ValueError(f"Annotation selected by '{raw_spec}' has an empty segmentation mask.")
        frame_mask = masks_by_frame.setdefault(
            spec.frame_id,
            np.zeros(frame_hw, dtype=np.int32),
        )
        frame_mask[binary_mask.astype(bool)] = int(spec.object_id)

    return masks_by_frame, frame_hw_by_id, default_hw


class MaskPrompt(Node):
    """Emit a scheduled label-map prompt mask for the requested frame."""

    INPUT_SPECS = {
        "frame_id": PortSpec(dtype=torch.int64, shape=(1,), description="Source frame index [1]."),
    }
    OUTPUT_SPECS = {
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(1, -1, -1),
            description="Prompt label map [1,H,W]. 0=background, positive values are object IDs.",
        ),
    }

    def __init__(
        self,
        json_path: str,
        prompt_specs: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.json_path = Path(json_path)
        self._prompt_specs = [str(spec) for spec in (prompt_specs or [])]
        self._masks_by_frame, self._frame_hw_by_id, self._default_hw = load_mask_prompt_schedule(
            self.json_path,
            self._prompt_specs,
        )
        super().__init__(json_path=str(self.json_path), prompt_specs=self._prompt_specs, **kwargs)

    def _resolve_frame_hw(self, frame_id: int) -> tuple[int, int]:
        if frame_id in self._frame_hw_by_id:
            frame_hw = self._frame_hw_by_id[frame_id]
            if (
                (frame_hw[0] <= 1 or frame_hw[1] <= 1)
                and self._default_hw is not None
                and self._default_hw[0] > 1
                and self._default_hw[1] > 1
            ):
                return self._default_hw
            return frame_hw
        if self._default_hw is not None:
            return self._default_hw
        raise ValueError(
            f"Frame {frame_id} is missing from {self.json_path} and no default frame size is available."
        )

    def forward(
        self,
        frame_id: torch.Tensor,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Emit the scheduled prompt label map for ``frame_id`` or an empty mask."""
        if frame_id is None or frame_id.numel() == 0:
            raise ValueError("MaskPrompt requires a non-empty frame_id input.")

        current_frame_id = int(frame_id.reshape(-1)[0].item())
        frame_hw = self._resolve_frame_hw(current_frame_id)
        label_map = self._masks_by_frame.get(current_frame_id)

        if label_map is None:
            mask_t = torch.zeros((1, frame_hw[0], frame_hw[1]), dtype=torch.int32)
        else:
            mask_t = (
                torch.from_numpy(np.array(label_map, copy=True)).unsqueeze(0).to(dtype=torch.int32)
            )
        return {"mask": mask_t}


__all__ = [
    "MaskPrompt",
    "MaskPromptSpec",
    "load_mask_prompt_schedule",
    "parse_mask_prompt_spec",
]
