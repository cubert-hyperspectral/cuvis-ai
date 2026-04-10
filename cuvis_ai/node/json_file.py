"""JSON source and sink nodes for detection and tracking."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cuvis_ai_core.data.rle import (
    coco_rle_area,
    coco_rle_decode,
    coco_rle_encode,
    coco_rle_to_bbox,
)
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger

# ---------------------------------------------------------------------------
# Writer base classes
# ---------------------------------------------------------------------------


class _BaseJsonWriterNode(Node):
    """Shared JSON write lifecycle for sink nodes."""

    OUTPUT_SPECS = {}

    def __init__(
        self,
        output_json_path: str,
        atomic_write: bool = True,
        flush_interval: int = 0,
        **kwargs: Any,
    ) -> None:
        if not output_json_path:
            raise ValueError("output_json_path must be a non-empty path.")
        if flush_interval < 0:
            raise ValueError("flush_interval must be >= 0.")

        self.output_json_path = Path(output_json_path)
        self.atomic_write = bool(atomic_write)
        self.flush_interval = int(flush_interval)
        self._dirty = False
        self._frames_since_flush = 0

        self.output_json_path.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(
            output_json_path=output_json_path,
            atomic_write=atomic_write,
            flush_interval=flush_interval,
            **kwargs,
        )

    def _mark_dirty_and_maybe_flush(self) -> None:
        """Record a pending write and flush immediately when the interval is reached."""
        self._dirty = True
        self._frames_since_flush += 1
        if self.flush_interval > 0 and self._frames_since_flush >= self.flush_interval:
            self._flush_json()

    def _finish_flush(self) -> None:
        """Reset dirty-state tracking after a payload has been written."""
        self._dirty = False
        self._frames_since_flush = 0

    def _write_json_direct(self, payload: dict[str, Any]) -> None:
        """Write a JSON payload straight to the destination file."""
        with self.output_json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    def _write_json_atomic(self, payload: dict[str, Any]) -> None:
        """Write a JSON payload via a temporary file and atomic rename."""
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self.output_json_path.parent),
            prefix=f".{self.output_json_path.stem}_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.output_json_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _write_payload(self, payload: dict[str, Any]) -> None:
        """Persist a payload with either atomic or direct file writes."""
        if self.atomic_write:
            self._write_json_atomic(payload)
        else:
            self._write_json_direct(payload)

    def _flush_json(self) -> None:
        """Serialize the subclass-specific in-memory state to disk."""
        raise NotImplementedError

    def close(self) -> None:
        """Flush pending changes before shutdown."""
        if self._dirty:
            self._flush_json()

    def __del__(self) -> None:
        """Attempt a best-effort flush when the writer is garbage collected."""
        try:
            self.close()
        except Exception as exc:
            logger.debug("Failed to flush JSON writer during cleanup: {}", exc)


class _BaseCocoTrackWriter(_BaseJsonWriterNode):
    """Shared tensor parsing helpers for tracking writers."""

    @staticmethod
    def _parse_frame_id(frame_id: torch.Tensor) -> int:
        """Convert a scalar frame-id tensor to a Python integer."""
        if frame_id.numel() != 1:
            raise ValueError("frame_id must contain exactly one scalar value.")
        return int(frame_id.reshape(-1)[0].item())

    @staticmethod
    def _parse_mask(mask: torch.Tensor) -> torch.Tensor:
        """Normalize a mask tensor to a 2D `[H, W]` view."""
        if mask.ndim == 3:
            if mask.shape[0] != 1:
                raise ValueError(
                    f"mask must have shape [1, H, W] or [H, W], got {tuple(mask.shape)}."
                )
            return mask[0]
        if mask.ndim == 2:
            return mask
        raise ValueError(f"mask must have shape [1, H, W] or [H, W], got {tuple(mask.shape)}.")

    @staticmethod
    def _parse_vector(tensor: torch.Tensor, port_name: str) -> torch.Tensor:
        """Normalize a vector-like tensor to shape `[N]`."""
        if tensor.ndim == 2:
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"{port_name} must have shape [1, N] or [N], got {tuple(tensor.shape)}."
                )
            return tensor[0]
        if tensor.ndim == 1:
            return tensor
        raise ValueError(f"{port_name} must have shape [1, N] or [N], got {tuple(tensor.shape)}.")

    @staticmethod
    def _validate_alignment(
        lhs: torch.Tensor, rhs: torch.Tensor, lhs_name: str, rhs_name: str
    ) -> None:
        """Ensure two tensors describe the same number of objects."""
        if int(lhs.numel()) != int(rhs.numel()):
            raise ValueError(f"{lhs_name} and {rhs_name} must have identical lengths.")


# ---------------------------------------------------------------------------
# Writer concrete classes
# ---------------------------------------------------------------------------


class CocoTrackMaskWriter(_BaseCocoTrackWriter):
    """Write mask tracking outputs into video_coco JSON."""

    INPUT_SPECS = {
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(1,),
            description="Frame identifier tensor [1]. Required.",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(1, -1, -1),
            description="Label map [1, H, W], pixel value = object ID, 0 = background.",
        ),
        "object_ids": PortSpec(
            dtype=torch.int64,
            shape=(1, -1),
            description="Object IDs [1, N], index-aligned to score tensors.",
        ),
        "detection_scores": PortSpec(
            dtype=torch.float32,
            shape=(1, -1),
            description="Detection scores [1, N], index-aligned to object_ids.",
        ),
        "category_ids": PortSpec(
            dtype=torch.int64,
            shape=(1, -1),
            description="Optional category IDs [1, N], index-aligned to object_ids.",
            optional=True,
        ),
        "category_semantics": PortSpec(
            dtype=torch.uint8,
            shape=(-1,),
            description="Optional UTF-8 JSON bytes mapping category IDs to category names.",
            optional=True,
        ),
    }

    def __init__(
        self,
        output_json_path: str,
        default_category_name: str = "object",
        write_empty_frames: bool = True,
        atomic_write: bool = True,
        flush_interval: int = 0,
        **kwargs: Any,
    ) -> None:
        if not default_category_name:
            raise ValueError("default_category_name must be a non-empty string.")

        self.default_category_name = default_category_name
        self.write_empty_frames = bool(write_empty_frames)
        self._frame_hw_by_id: dict[int, tuple[int, int]] = {}
        self._track_segmentations: dict[int, dict[int, dict[str, Any]]] = {}
        self._track_scores: dict[int, dict[int, float]] = {}
        self._track_bboxes: dict[int, dict[int, list[float]]] = {}
        self._track_areas: dict[int, dict[int, float]] = {}
        self._track_category_ids: dict[int, int] = {}
        self._category_id_to_name: dict[int, str] = {}

        super().__init__(
            output_json_path=output_json_path,
            atomic_write=atomic_write,
            flush_interval=flush_interval,
            default_category_name=default_category_name,
            write_empty_frames=write_empty_frames,
            **kwargs,
        )

    def _drop_frame(self, frame_idx: int) -> None:
        """Remove any cached tracking data for a frame being replaced."""
        self._frame_hw_by_id.pop(frame_idx, None)
        for store in (
            self._track_segmentations,
            self._track_scores,
            self._track_bboxes,
            self._track_areas,
        ):
            for track_id in list(store.keys()):
                store[track_id].pop(frame_idx, None)
                if not store[track_id]:
                    del store[track_id]

    @staticmethod
    def _parse_category_semantics(category_semantics: torch.Tensor) -> dict[int, str]:
        """Decode category-name metadata from a UTF-8 JSON byte tensor."""
        semantics_bytes = _BaseCocoTrackWriter._parse_vector(
            category_semantics, port_name="category_semantics"
        )
        try:
            payload = bytes(int(v) for v in semantics_bytes.to(dtype=torch.uint8).cpu().tolist())
            decoded = json.loads(payload.decode("utf-8"))
        except Exception as exc:  # pragma: no cover - precise exception depends on payload
            raise ValueError("category_semantics must contain valid UTF-8 JSON bytes.") from exc

        if not isinstance(decoded, dict):
            raise ValueError("category_semantics JSON must decode to an object mapping.")

        parsed: dict[int, str] = {}
        for key, value in decoded.items():
            try:
                category_id = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid category_semantics key: {key!r}.") from exc
            if category_id <= 0:
                raise ValueError("category_semantics category IDs must be positive integers.")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"category_semantics value for category {category_id} must be a non-empty string."
                )
            parsed[category_id] = value
        return parsed

    def _update_category_semantics(self, category_semantics: torch.Tensor | None) -> None:
        """Merge validated category semantics into the writer state."""
        if category_semantics is None:
            return
        parsed = self._parse_category_semantics(category_semantics)
        for category_id, name in parsed.items():
            existing = self._category_id_to_name.get(category_id)
            if existing is not None and existing != name:
                raise ValueError(
                    f"category_semantics changed meaning for category_id={category_id}: "
                    f"{existing!r} -> {name!r}."
                )
            self._category_id_to_name[category_id] = name

    def forward(
        self,
        frame_id: torch.Tensor,
        mask: torch.Tensor,
        object_ids: torch.Tensor,
        detection_scores: torch.Tensor,
        category_ids: torch.Tensor | None = None,
        category_semantics: torch.Tensor | None = None,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, Any]:
        """Store one frame of tracked masks and metadata for later JSON export."""
        frame_idx = self._parse_frame_id(frame_id)
        mask_2d = self._parse_mask(mask)
        ids_1d = self._parse_vector(object_ids, port_name="object_ids")
        scores_1d = self._parse_vector(detection_scores, port_name="detection_scores")
        self._validate_alignment(ids_1d, scores_1d, "object_ids", "detection_scores")
        category_ids_1d: torch.Tensor | None = None
        if category_ids is not None:
            category_ids_1d = self._parse_vector(category_ids, port_name="category_ids")
            self._validate_alignment(ids_1d, category_ids_1d, "object_ids", "category_ids")
        self._update_category_semantics(category_semantics)

        frame_height = int(mask_2d.shape[0])
        frame_width = int(mask_2d.shape[1])

        # Replacing an existing frame should be idempotent.
        self._drop_frame(frame_idx)

        object_ids_list = ids_1d.to(dtype=torch.int64).cpu().tolist()
        detection_scores_list = scores_1d.to(dtype=torch.float32).cpu().tolist()
        category_ids_list = (
            category_ids_1d.to(dtype=torch.int64).cpu().tolist()
            if category_ids_1d is not None
            else [1] * len(object_ids_list)
        )
        score_by_obj_id: dict[int, float] = {
            int(obj_id): float(score)
            for obj_id, score in zip(object_ids_list, detection_scores_list, strict=False)
            if int(obj_id) > 0
        }
        category_by_obj_id: dict[int, int] = {}
        for obj_id, category_id in zip(object_ids_list, category_ids_list, strict=False):
            oid = int(obj_id)
            cid = int(category_id)
            if oid <= 0:
                continue
            if cid <= 0:
                raise ValueError("category_ids must be positive for tracked objects.")
            existing_category_id = self._track_category_ids.get(oid)
            if existing_category_id is not None and existing_category_id != cid:
                raise ValueError(
                    f"Track {oid} received conflicting category IDs: "
                    f"{existing_category_id} vs {cid}."
                )
            self._track_category_ids.setdefault(oid, cid)
            category_by_obj_id[oid] = cid
            fallback_name = self.default_category_name if cid == 1 else f"category_{cid}"
            self._category_id_to_name.setdefault(cid, fallback_name)
        present_obj_ids = {
            int(obj_id)
            for obj_id in mask_2d.to(dtype=torch.int64).unique().cpu().tolist()
            if int(obj_id) > 0
        }

        export_obj_ids: list[int] = []
        seen_obj_ids: set[int] = set()
        for obj_id in object_ids_list:
            oid = int(obj_id)
            if oid <= 0 or oid not in present_obj_ids or oid in seen_obj_ids:
                continue
            seen_obj_ids.add(oid)
            export_obj_ids.append(oid)

        if not export_obj_ids and not self.write_empty_frames:
            return {}

        self._frame_hw_by_id[frame_idx] = (frame_height, frame_width)

        for oid in export_obj_ids:
            obj_mask = mask_2d.eq(oid)
            if not bool(torch.any(obj_mask)):
                continue

            mask_np = obj_mask.to(dtype=torch.uint8).detach().cpu().numpy()
            rle_json = coco_rle_encode(mask_np)
            bbox = coco_rle_to_bbox(rle_json)
            area = coco_rle_area(rle_json)

            self._track_segmentations.setdefault(oid, {})[frame_idx] = rle_json
            self._track_scores.setdefault(oid, {})[frame_idx] = float(score_by_obj_id.get(oid, 0.0))
            self._track_bboxes.setdefault(oid, {})[frame_idx] = bbox
            self._track_areas.setdefault(oid, {})[frame_idx] = area
            if oid in category_by_obj_id:
                self._track_category_ids.setdefault(oid, category_by_obj_id[oid])

        self._mark_dirty_and_maybe_flush()
        return {}

    def _flush_json(self) -> None:
        """Emit the accumulated mask tracks in video-COCO format."""
        frame_indices = sorted(self._frame_hw_by_id.keys())
        first_frame = frame_indices[0] if frame_indices else 0
        frame_h, frame_w = self._frame_hw_by_id.get(first_frame, (0, 0))

        annotations = []
        ann_id = 1
        for track_id in sorted(self._track_segmentations.keys()):
            per_frame_seg = self._track_segmentations.get(track_id, {})
            per_frame_scores = self._track_scores.get(track_id, {})
            per_frame_bboxes = self._track_bboxes.get(track_id, {})
            per_frame_areas = self._track_areas.get(track_id, {})

            segmentations: list[dict[str, Any] | None] = []
            detection_scores: list[float | None] = []
            bboxes: list[list[float] | None] = []
            areas: list[float | None] = []
            for fid in frame_indices:
                seg = per_frame_seg.get(fid)
                if seg is None:
                    segmentations.append(None)
                    detection_scores.append(None)
                    bboxes.append(None)
                    areas.append(None)
                    continue
                segmentations.append(seg)
                detection_scores.append(float(per_frame_scores.get(fid, 0.0)))
                bboxes.append(per_frame_bboxes.get(fid))
                areas.append(float(per_frame_areas.get(fid, 0.0)))

            annotations.append(
                {
                    "id": ann_id,
                    "track_id": int(track_id),
                    "category_id": int(self._track_category_ids.get(track_id, 1)),
                    "segmentations": segmentations,
                    "detection_scores": detection_scores,
                    "bboxes": bboxes,
                    "areas": areas,
                }
            )
            ann_id += 1

        categories = [
            {"id": int(category_id), "name": name}
            for category_id, name in sorted(self._category_id_to_name.items())
        ]
        if not categories:
            categories = [{"id": 1, "name": self.default_category_name}]

        payload = {
            "info": {"description": "Mask tracking results", "version": "1.0"},
            "videos": [
                {
                    "id": 1,
                    "name": self.output_json_path.stem,
                    "frame_indices": frame_indices,
                    "start_frame": int(first_frame if frame_indices else 0),
                    "length": int(len(frame_indices)),
                    "height": int(frame_h),
                    "width": int(frame_w),
                }
            ],
            "annotations": annotations,
            "categories": categories,
        }
        self._write_payload(payload)
        self._finish_flush()


class DetectionCocoJsonNode(_BaseJsonWriterNode):
    """Write frame-wise detections into COCO detection JSON."""

    INPUT_SPECS = {
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(1,),
            description="Frame identifier tensor [1]. Required.",
        ),
        "bboxes": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, 4),
            description="Boxes [1, N, 4] xyxy in original image coordinates.",
        ),
        "category_ids": PortSpec(
            dtype=torch.int64,
            shape=(1, -1),
            description="COCO class IDs [1, N].",
        ),
        "confidences": PortSpec(
            dtype=torch.float32,
            shape=(1, -1),
            description="Detection scores [1, N].",
        ),
        "orig_hw": PortSpec(
            dtype=torch.int64,
            shape=(1, 2),
            description="Frame [H, W] for image metadata.",
        ),
    }

    def __init__(
        self,
        output_json_path: str,
        category_id_to_name: dict[int, str] | None = None,
        write_empty_frames: bool = True,
        atomic_write: bool = True,
        flush_interval: int = 0,
        **kwargs: Any,
    ) -> None:
        self.category_id_to_name: dict[int, str] = (
            dict(category_id_to_name) if category_id_to_name is not None else {0: "person"}
        )
        self.write_empty_frames = bool(write_empty_frames)
        self._frames_by_id: dict[int, dict[str, Any]] = {}

        super().__init__(
            output_json_path=output_json_path,
            atomic_write=atomic_write,
            flush_interval=flush_interval,
            category_id_to_name=self.category_id_to_name,
            write_empty_frames=write_empty_frames,
            **kwargs,
        )

    def forward(
        self,
        frame_id: torch.Tensor,
        bboxes: torch.Tensor,
        category_ids: torch.Tensor,
        confidences: torch.Tensor,
        orig_hw: torch.Tensor,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, Any]:
        """Store one frame of detections for COCO JSON serialization."""
        frame_idx = _BaseCocoTrackWriter._parse_frame_id(frame_id)
        ids_1d = _BaseCocoTrackWriter._parse_vector(category_ids, port_name="category_ids")
        scores_1d = _BaseCocoTrackWriter._parse_vector(confidences, port_name="confidences")
        _BaseCocoTrackWriter._validate_alignment(ids_1d, scores_1d, "category_ids", "confidences")

        h, w = int(orig_hw[0, 0]), int(orig_hw[0, 1])
        boxes_2d = bboxes[0] if bboxes.ndim == 3 else bboxes

        n = int(ids_1d.numel())
        detections: list[dict[str, Any]] = []
        for i in range(n):
            x1, y1, x2, y2 = boxes_2d[i].cpu().tolist()
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            detections.append(
                {
                    "category_id": int(ids_1d[i].item()),
                    "bbox": [float(x1), float(y1), bw, bh],
                    "area": bw * bh,
                    "score": float(scores_1d[i].item()),
                }
            )

        if not detections and not self.write_empty_frames:
            return {}

        self._frames_by_id[frame_idx] = {
            "frame_idx": frame_idx,
            "height": h,
            "width": w,
            "detections": detections,
        }
        self._mark_dirty_and_maybe_flush()
        return {}

    def _flush_json(self) -> None:
        """Write the cached frame detections as a COCO detection file."""
        frames = [self._frames_by_id[idx] for idx in sorted(self._frames_by_id.keys())]

        images = [
            {
                "id": int(frame["frame_idx"]),
                "file_name": f"frame_{int(frame['frame_idx']):06d}",
                "height": int(frame["height"]),
                "width": int(frame["width"]),
            }
            for frame in frames
        ]

        annotations = []
        ann_id = 1
        for frame in frames:
            frame_idx = int(frame["frame_idx"])
            for det in frame["detections"]:
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": frame_idx,
                        "category_id": det["category_id"],
                        "bbox": det["bbox"],
                        "area": det["area"],
                        "iscrowd": 0,
                        "score": det["score"],
                    }
                )
                ann_id += 1

        categories = [
            {"id": cat_id, "name": name}
            for cat_id, name in sorted(self.category_id_to_name.items())
        ]

        payload = {
            "info": {"description": "Detection results", "version": "1.0"},
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        self._write_payload(payload)
        self._finish_flush()


class CocoTrackBBoxWriter(_BaseCocoTrackWriter):
    """Write tracked bbox outputs into COCO tracking JSON."""

    INPUT_SPECS = {
        "frame_id": PortSpec(dtype=torch.int64, shape=(1,), description="Frame identifier [1]."),
        "bboxes": PortSpec(
            dtype=torch.float32, shape=(1, -1, 4), description="Boxes [1,N,4] xyxy."
        ),
        "category_ids": PortSpec(dtype=torch.int64, shape=(1, -1), description="Class IDs [1,N]."),
        "confidences": PortSpec(dtype=torch.float32, shape=(1, -1), description="Scores [1,N]."),
        "track_ids": PortSpec(dtype=torch.int64, shape=(1, -1), description="Track IDs [1,N]."),
        "orig_hw": PortSpec(dtype=torch.int64, shape=(1, 2), description="[H,W] for metadata."),
    }

    def __init__(
        self,
        output_json_path: str,
        category_id_to_name: dict[int, str] | None = None,
        write_empty_frames: bool = True,
        atomic_write: bool = True,
        flush_interval: int = 0,
        **kwargs: Any,
    ) -> None:
        self.category_id_to_name: dict[int, str] = (
            dict(category_id_to_name) if category_id_to_name is not None else {0: "object"}
        )
        self.write_empty_frames = bool(write_empty_frames)
        self._frames_by_id: dict[int, dict[str, Any]] = {}

        super().__init__(
            output_json_path=output_json_path,
            atomic_write=atomic_write,
            flush_interval=flush_interval,
            category_id_to_name=self.category_id_to_name,
            write_empty_frames=write_empty_frames,
            **kwargs,
        )

    def forward(
        self,
        frame_id: torch.Tensor,
        bboxes: torch.Tensor,
        category_ids: torch.Tensor,
        confidences: torch.Tensor,
        track_ids: torch.Tensor,
        orig_hw: torch.Tensor,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, Any]:
        """Store one frame of tracked bounding boxes for later export."""
        frame_idx = self._parse_frame_id(frame_id)
        ids_1d = self._parse_vector(category_ids, port_name="category_ids")
        scores_1d = self._parse_vector(confidences, port_name="confidences")
        track_ids_1d = self._parse_vector(track_ids, port_name="track_ids")
        self._validate_alignment(ids_1d, scores_1d, "category_ids", "confidences")
        self._validate_alignment(ids_1d, track_ids_1d, "category_ids", "track_ids")

        h, w = int(orig_hw[0, 0]), int(orig_hw[0, 1])
        boxes_2d = bboxes[0] if bboxes.ndim == 3 else bboxes

        n = int(ids_1d.numel())
        detections: list[dict[str, Any]] = []
        for i in range(n):
            x1, y1, x2, y2 = boxes_2d[i].cpu().tolist()
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            detections.append(
                {
                    "category_id": int(ids_1d[i].item()),
                    "bbox": [float(x1), float(y1), bw, bh],
                    "area": bw * bh,
                    "score": float(scores_1d[i].item()),
                    "track_id": int(track_ids_1d[i].item()),
                }
            )

        if not detections and not self.write_empty_frames:
            return {}

        self._frames_by_id[frame_idx] = {
            "frame_idx": frame_idx,
            "height": h,
            "width": w,
            "detections": detections,
        }
        self._mark_dirty_and_maybe_flush()
        return {}

    def _flush_json(self) -> None:
        """Write the cached tracked boxes as COCO tracking annotations."""
        frames = [self._frames_by_id[idx] for idx in sorted(self._frames_by_id.keys())]

        images = [
            {
                "id": int(frame["frame_idx"]),
                "file_name": f"frame_{int(frame['frame_idx']):06d}",
                "height": int(frame["height"]),
                "width": int(frame["width"]),
            }
            for frame in frames
        ]

        annotations = []
        ann_id = 1
        for frame in frames:
            frame_idx = int(frame["frame_idx"])
            for det in frame["detections"]:
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": frame_idx,
                        "category_id": det["category_id"],
                        "bbox": det["bbox"],
                        "area": det["area"],
                        "iscrowd": 0,
                        "score": det["score"],
                        "track_id": det["track_id"],
                    }
                )
                ann_id += 1

        categories = [
            {"id": cat_id, "name": name}
            for cat_id, name in sorted(self.category_id_to_name.items())
        ]

        payload = {
            "info": {"description": "BBox tracking results", "version": "1.0"},
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        self._write_payload(payload)
        self._finish_flush()


# ---------------------------------------------------------------------------
# Reader classes
# ---------------------------------------------------------------------------


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
        vid_h = int(video.get("height", 0))
        vid_w = int(video.get("width", 0))

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
        self._mask_hw: tuple[int, int] = (vid_h, vid_w)
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


__all__ = [
    "CocoTrackBBoxWriter",
    "CocoTrackMaskWriter",
    "DetectionCocoJsonNode",
    "DetectionJsonReader",
    "TrackingResultsReader",
]
