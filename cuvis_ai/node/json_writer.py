"""JSON sink nodes for tracking outputs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pycocotools.mask as mask_util
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec


def _encode_mask_to_rle(mask_2d: torch.Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
    """Encode a single binary mask [H, W] to COCO RLE for JSON and metrics."""
    mask_np = mask_2d.to(dtype=torch.uint8).detach().cpu().numpy()
    encoded = mask_util.encode(np.asfortranarray(mask_np))

    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts_str = counts.decode("utf-8")
    elif isinstance(counts, str):
        counts_str = counts
    else:
        counts_str = str(counts)

    size = [int(v) for v in encoded["size"]]
    rle_json = {"size": size, "counts": counts_str}
    rle_metrics = {"size": size, "counts": counts_str.encode("utf-8")}
    return rle_json, rle_metrics


class TrackingCocoJsonNode(Node):
    """Write frame-wise tracking outputs into COCO instance-segmentation JSON."""

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
    }

    OUTPUT_SPECS = {}

    def __init__(
        self,
        output_json_path: str,
        category_name: str = "person",
        write_empty_frames: bool = True,
        atomic_write: bool = True,
        **kwargs: Any,
    ) -> None:
        if not output_json_path:
            raise ValueError("output_json_path must be a non-empty path.")
        if not category_name:
            raise ValueError("category_name must be a non-empty string.")

        self.output_json_path = Path(output_json_path)
        self.category_name = category_name
        self.write_empty_frames = bool(write_empty_frames)
        self.atomic_write = bool(atomic_write)
        self._frames_by_id: dict[int, dict[str, Any]] = {}

        self.output_json_path.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(
            output_json_path=output_json_path,
            category_name=category_name,
            write_empty_frames=write_empty_frames,
            atomic_write=atomic_write,
            **kwargs,
        )

    def forward(
        self,
        frame_id: torch.Tensor,
        mask: torch.Tensor,
        object_ids: torch.Tensor,
        detection_scores: torch.Tensor,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, Any]:
        """Update one frame entry and rewrite the COCO JSON payload."""
        frame_idx = self._parse_frame_id(frame_id)
        mask_2d = self._parse_mask(mask)
        ids_1d = self._parse_vector(object_ids, port_name="object_ids")
        det_scores_1d = self._parse_vector(detection_scores, port_name="detection_scores")
        self._validate_alignment(ids_1d, det_scores_1d)

        frame_height = int(mask_2d.shape[0])
        frame_width = int(mask_2d.shape[1])

        object_ids_list = ids_1d.to(dtype=torch.int64).cpu().tolist()
        detection_scores_list = det_scores_1d.to(dtype=torch.float32).cpu().tolist()

        objects: list[dict[str, Any]] = []
        for idx, obj_id in enumerate(object_ids_list):
            if int(obj_id) <= 0:
                # Background/non-object IDs must not be exported as track annotations.
                continue
            obj_mask = mask_2d.eq(int(obj_id))
            if not bool(obj_mask.any()):
                continue

            rle_json, rle_metrics = _encode_mask_to_rle(obj_mask)
            bbox = mask_util.toBbox(rle_metrics).tolist()
            area = int(mask_util.area(rle_metrics))

            objects.append(
                {
                    "object_id": int(obj_id),
                    "detection_score": float(detection_scores_list[idx]),
                    "segmentation": rle_json,
                    "bbox": bbox,
                    "area": area,
                }
            )

        if not objects and not self.write_empty_frames:
            return {}

        self._frames_by_id[frame_idx] = {
            "frame_idx": frame_idx,
            "height": frame_height,
            "width": frame_width,
            "objects": objects,
        }
        self._flush_json()
        return {}

    @staticmethod
    def _parse_frame_id(frame_id: torch.Tensor) -> int:
        """Extract a scalar integer frame index from a single-element tensor."""
        if frame_id.numel() != 1:
            raise ValueError("frame_id must contain exactly one scalar value.")
        return int(frame_id.reshape(-1)[0].item())

    @staticmethod
    def _parse_mask(mask: torch.Tensor) -> torch.Tensor:
        """Squeeze a [1, H, W] or [H, W] mask tensor to 2-D."""
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
        """Squeeze a [1, N] or [N] vector tensor to 1-D, raising on invalid shape."""
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
    def _validate_alignment(object_ids: torch.Tensor, detection_scores: torch.Tensor) -> None:
        """Verify that object_ids and detection_scores have matching lengths."""
        if int(object_ids.numel()) != int(detection_scores.numel()):
            raise ValueError("object_ids and detection_scores must have identical lengths.")

    def _flush_json(self) -> None:
        """Assemble all accumulated frames into a COCO-format dict and write to disk."""
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
            for obj in frame["objects"]:
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": frame_idx,
                        "category_id": 1,
                        "segmentation": obj["segmentation"],
                        "bbox": obj["bbox"],
                        "area": obj["area"],
                        "score": obj["detection_score"],
                        "iscrowd": 0,
                        "track_id": obj["object_id"],
                    }
                )
                ann_id += 1

        payload = {
            "info": {"description": "SAM3 HSI tracking results", "version": "1.0"},
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": self.category_name}],
        }
        if self.atomic_write:
            self._write_json_atomic(payload)
        else:
            self._write_json_direct(payload)

    def _write_json_direct(self, payload: dict[str, Any]) -> None:
        """Write the JSON payload directly to the output file."""
        with self.output_json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    def _write_json_atomic(self, payload: dict[str, Any]) -> None:
        """Write the JSON payload atomically via a temporary file and rename."""
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


__all__ = ["TrackingCocoJsonNode"]
