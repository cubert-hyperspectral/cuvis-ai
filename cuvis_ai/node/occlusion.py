"""Synthetic occlusion nodes for tracking evaluation (pure PyTorch)."""

from __future__ import annotations

import abc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from cuvis_ai_core.data.rle import coco_rle_decode
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger

from cuvis_ai.utils.poisson_inpaint import poisson_inpaint


class OcclusionNodeBase(Node, abc.ABC):
    """Base class for synthetic occlusion from tracking masks."""

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, -1, 3),
            description="Single RGB frame [1, H, W, 3] in [0, 1].",
        ),
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(1,),
            description="Frame index [1].",
        ),
    }

    OUTPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, -1, 3),
            description="Possibly occluded RGB frame [1, H, W, 3] in [0, 1].",
        ),
    }

    def __init__(
        self,
        tracking_json_path: str,
        track_ids: list[int],
        occlusion_start_frame: int,
        occlusion_end_frame: int,
        **kwargs,
    ) -> None:
        path = Path(tracking_json_path)
        if not path.is_file():
            raise FileNotFoundError(f"Tracking JSON not found: {tracking_json_path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        track_id_set = set(track_ids)

        self._masks_by_frame: dict[int, list[dict]] = {}
        for ann in data.get("annotations", []):
            tid = ann.get("track_id")
            if tid not in track_id_set:
                continue
            fid = int(ann["image_id"])
            if fid < occlusion_start_frame or fid > occlusion_end_frame:
                continue
            seg = ann.get("segmentation")
            if seg is None or not isinstance(seg, dict):
                continue
            entry = {
                "track_id": int(tid),
                "bbox": ann["bbox"],
                "segmentation": seg,
            }
            self._masks_by_frame.setdefault(fid, []).append(entry)

        self.occlusion_start_frame = int(occlusion_start_frame)
        self.occlusion_end_frame = int(occlusion_end_frame)

        n_frames = len(self._masks_by_frame)
        n_annots = sum(len(v) for v in self._masks_by_frame.values())
        logger.info(
            "OcclusionNode: loaded {} annotations across {} frames for tracks {} (range [{}, {}])",
            n_annots,
            n_frames,
            track_ids,
            occlusion_start_frame,
            occlusion_end_frame,
        )

        super().__init__(
            tracking_json_path=tracking_json_path,
            track_ids=track_ids,
            occlusion_start_frame=occlusion_start_frame,
            occlusion_end_frame=occlusion_end_frame,
            **kwargs,
        )

    @staticmethod
    def _resize_mask_nearest(mask: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
        """Resize binary mask with nearest-neighbour using torch ops."""
        mh, mw = mask.shape[:2]
        if mh == frame_h and mw == frame_w:
            return mask.astype(np.uint8)
        mask_t = torch.from_numpy(mask.astype(np.float32)).view(1, 1, mh, mw)
        resized = F.interpolate(mask_t, size=(frame_h, frame_w), mode="nearest")
        return (resized[0, 0] > 0.5).to(torch.uint8).cpu().numpy()

    def _get_masks_for_frame(
        self,
        frame_idx: int,
        frame_h: int,
        frame_w: int,
    ) -> list[tuple[np.ndarray, list]]:
        """Decode RLE masks for a frame and resize if needed."""
        entries = self._masks_by_frame.get(frame_idx, [])
        results: list[tuple[np.ndarray, list]] = []
        for entry in entries:
            mask = coco_rle_decode(entry["segmentation"])  # uint8 [H, W]
            resized_mask = self._resize_mask_nearest(mask, frame_h, frame_w)
            results.append((resized_mask, entry["bbox"]))
        return results

    @abc.abstractmethod
    def _apply_occlusion(
        self,
        frame: torch.Tensor,
        masks: list[tuple[np.ndarray, list]],
    ) -> torch.Tensor:
        """Apply occlusion to a single frame [H, W, C]."""

    def _forward_tensor(
        self,
        *,
        data: torch.Tensor,
        output_key: str,
        frame_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        frame_idx = int(frame_id[0].item())

        if frame_idx < self.occlusion_start_frame or frame_idx > self.occlusion_end_frame:
            return {output_key: data}

        h, w = data.shape[1], data.shape[2]
        masks = self._get_masks_for_frame(frame_idx, h, w)
        if not masks:
            return {output_key: data}

        occluded = self._apply_occlusion(data[0], masks)
        return {output_key: occluded.unsqueeze(0)}

    @torch.no_grad()
    def forward(
        self,
        rgb_image: torch.Tensor,
        frame_id: torch.Tensor,
        **_,
    ) -> dict[str, torch.Tensor]:
        return self._forward_tensor(data=rgb_image, output_key="rgb_image", frame_id=frame_id)


class PoissonOcclusionNode(OcclusionNodeBase):
    """Pure-PyTorch occlusion node for either RGB frames or hyperspectral cubes."""

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, -1, 3),
            description="Single RGB frame [1, H, W, 3] in [0, 1].",
            optional=True,
        ),
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, -1, -1),
            description="Single cube frame [1, H, W, C].",
            optional=True,
        ),
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(1,),
            description="Frame index [1].",
        ),
    }

    OUTPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, -1, 3),
            description="Possibly occluded RGB frame [1, H, W, 3].",
            optional=True,
        ),
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, -1, -1),
            description="Possibly occluded cube frame [1, H, W, C].",
            optional=True,
        ),
    }

    _VALID_SHAPES = ("bbox", "mask")
    _VALID_BBOX_MODES = ("static", "dynamic")

    def __init__(
        self,
        tracking_json_path: str,
        track_ids: list[int],
        occlusion_start_frame: int,
        occlusion_end_frame: int,
        fill_color: tuple[float, float, float] | str = "poisson",
        *,
        input_key: str | None = None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        occlusion_shape: str = "bbox",
        bbox_mode: str = "static",
        static_bbox_scale: float = 1.2,
        static_bbox_padding_px: int = 0,
        static_full_width_x: bool = False,
        **kwargs,
    ) -> None:
        if occlusion_shape not in self._VALID_SHAPES:
            raise ValueError(
                f"occlusion_shape must be one of {self._VALID_SHAPES}, got '{occlusion_shape}'"
            )
        if bbox_mode not in self._VALID_BBOX_MODES:
            raise ValueError(
                f"bbox_mode must be one of {self._VALID_BBOX_MODES}, got '{bbox_mode}'"
            )
        if static_bbox_scale <= 0:
            raise ValueError("static_bbox_scale must be > 0")
        if static_bbox_padding_px < 0:
            raise ValueError("static_bbox_padding_px must be >= 0")
        if int(max_iter) <= 0:
            raise ValueError("max_iter must be > 0")
        if float(tol) <= 0:
            raise ValueError("tol must be > 0")
        if input_key is not None and input_key not in {"rgb_image", "cube"}:
            raise ValueError("input_key must be 'rgb_image', 'cube', or None")

        self._use_poisson_fill = False
        if isinstance(fill_color, str):
            if fill_color != "poisson":
                raise ValueError("fill_color string must be exactly 'poisson'")
            self.fill_color: tuple[float, float, float] | str = fill_color
            self._use_poisson_fill = True
        else:
            parsed_fill = tuple(float(c) for c in fill_color)
            if len(parsed_fill) != 3:
                raise ValueError("fill_color tuple must have exactly 3 values")
            if any(c < 0.0 or c > 1.0 for c in parsed_fill):
                raise ValueError("fill_color tuple values must be in [0, 1]")
            self.fill_color = parsed_fill

        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.input_key = input_key
        self.occlusion_shape = occlusion_shape
        self.bbox_mode = bbox_mode
        self.static_bbox_scale = float(static_bbox_scale)
        self.static_bbox_padding_px = int(static_bbox_padding_px)
        self.static_full_width_x = bool(static_full_width_x)

        super().__init__(
            tracking_json_path=tracking_json_path,
            track_ids=track_ids,
            occlusion_start_frame=occlusion_start_frame,
            occlusion_end_frame=occlusion_end_frame,
            fill_color=self.fill_color,
            input_key=self.input_key,
            max_iter=self.max_iter,
            tol=self.tol,
            occlusion_shape=occlusion_shape,
            bbox_mode=bbox_mode,
            static_bbox_scale=static_bbox_scale,
            static_bbox_padding_px=static_bbox_padding_px,
            static_full_width_x=static_full_width_x,
            **kwargs,
        )

        self._static_bboxes_by_track: dict[int, list[float]] = {}
        if self.occlusion_shape == "bbox" and self.bbox_mode == "static":
            self._static_bboxes_by_track = self._build_static_bboxes_by_track()
            logger.info(
                "PoissonOcclusionNode static bboxes: {} tracks (scale={}, padding_px={})",
                len(self._static_bboxes_by_track),
                self.static_bbox_scale,
                self.static_bbox_padding_px,
            )

    def _build_static_bboxes_by_track(self) -> dict[int, list[float]]:
        """Build one fixed bbox per track as union of all in-range bboxes."""
        bounds_by_track: dict[int, list[float]] = {}
        for entries in self._masks_by_frame.values():
            for entry in entries:
                tid = int(entry["track_id"])
                x, y, w, h = entry["bbox"]
                x0 = float(x)
                y0 = float(y)
                x1 = float(x + w)
                y1 = float(y + h)
                if tid not in bounds_by_track:
                    bounds_by_track[tid] = [x0, y0, x1, y1]
                    continue
                cur = bounds_by_track[tid]
                cur[0] = min(cur[0], x0)
                cur[1] = min(cur[1], y0)
                cur[2] = max(cur[2], x1)
                cur[3] = max(cur[3], y1)

        static_by_track: dict[int, list[float]] = {}
        for tid, (x0, y0, x1, y1) in bounds_by_track.items():
            union_w = max(1.0, x1 - x0)
            union_h = max(1.0, y1 - y0)
            cx = (x0 + x1) * 0.5
            cy = (y0 + y1) * 0.5

            scaled_w = union_w * self.static_bbox_scale + 2.0 * self.static_bbox_padding_px
            scaled_h = union_h * self.static_bbox_scale + 2.0 * self.static_bbox_padding_px
            scaled_w = max(1.0, scaled_w)
            scaled_h = max(1.0, scaled_h)

            sx0 = cx - 0.5 * scaled_w
            sy0 = cy - 0.5 * scaled_h
            static_by_track[tid] = [sx0, sy0, scaled_w, scaled_h]

        return static_by_track

    def _get_masks_for_frame(
        self,
        frame_idx: int,
        frame_h: int,
        frame_w: int,
    ) -> list[tuple[np.ndarray, list]]:
        if self.occlusion_shape == "bbox" and self.bbox_mode == "static":
            if not self._static_bboxes_by_track:
                return []
            empty_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            return [(empty_mask, bbox_xywh) for bbox_xywh in self._static_bboxes_by_track.values()]
        return super()._get_masks_for_frame(frame_idx, frame_h, frame_w)

    def _bbox_to_clipped_coords(
        self,
        bbox_xywh: list[float],
        frame_h: int,
        frame_w: int,
    ) -> tuple[int, int, int, int]:
        x, y, w, h = bbox_xywh
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(frame_w, int(round(x + w)))
        y1 = min(frame_h, int(round(y + h)))
        if self.bbox_mode == "static" and self.static_full_width_x:
            x0 = 0
            x1 = frame_w
        return x0, y0, x1, y1

    def _build_combined_mask(
        self,
        masks: list[tuple[np.ndarray, list]],
        frame_h: int,
        frame_w: int,
    ) -> np.ndarray:
        combined_mask = np.zeros((frame_h, frame_w), dtype=bool)
        for binary_mask, bbox_xywh in masks:
            if self.occlusion_shape == "bbox":
                x0, y0, x1, y1 = self._bbox_to_clipped_coords(bbox_xywh, frame_h, frame_w)
                if x0 >= x1 or y0 >= y1:
                    continue
                combined_mask[y0:y1, x0:x1] = True
            else:
                combined_mask[binary_mask.astype(bool)] = True
        return combined_mask

    def _apply_occlusion(
        self,
        frame: torch.Tensor,
        masks: list[tuple[np.ndarray, list]],
    ) -> torch.Tensor:
        frame_h, frame_w = frame.shape[:2]
        combined_mask = self._build_combined_mask(masks, frame_h, frame_w)
        if not np.any(combined_mask):
            return frame

        if self._use_poisson_fill:
            mask_t = torch.from_numpy(combined_mask).to(frame.device)
            return poisson_inpaint(frame, mask_t, max_iter=self.max_iter, tol=self.tol)

        result = frame.clone()
        fill = torch.tensor(self.fill_color, dtype=frame.dtype, device=frame.device)
        mask_t = torch.from_numpy(combined_mask).to(frame.device)
        result[mask_t, :] = fill
        return result

    @torch.no_grad()
    def forward(
        self,
        frame_id: torch.Tensor,
        rgb_image: torch.Tensor | None = None,
        cube: torch.Tensor | None = None,
        **_,
    ) -> dict[str, torch.Tensor]:
        if self.input_key == "rgb_image":
            if rgb_image is None:
                raise ValueError(
                    "PoissonOcclusionNode configured for rgb_image but none was provided"
                )
            return self._forward_tensor(data=rgb_image, output_key="rgb_image", frame_id=frame_id)

        if self.input_key == "cube":
            if cube is None:
                raise ValueError("PoissonOcclusionNode configured for cube but none was provided")
            return self._forward_tensor(data=cube, output_key="cube", frame_id=frame_id)

        if (rgb_image is None) and (cube is None):
            raise ValueError("PoissonOcclusionNode requires exactly one input: rgb_image or cube")
        if (rgb_image is not None) and (cube is not None):
            raise ValueError("PoissonOcclusionNode accepts either rgb_image or cube, not both")

        if rgb_image is not None:
            return self._forward_tensor(data=rgb_image, output_key="rgb_image", frame_id=frame_id)

        assert cube is not None
        return self._forward_tensor(data=cube, output_key="cube", frame_id=frame_id)


# Backward-compatible aliases.
class SolidOcclusionNode(PoissonOcclusionNode):
    """Deprecated alias of PoissonOcclusionNode."""


class PoissonCubeOcclusionNode(PoissonOcclusionNode):
    """Deprecated alias of PoissonOcclusionNode with cube-only ports."""

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, -1, -1),
            description="Single cube frame [1, H, W, C].",
        ),
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(1,),
            description="Frame index [1].",
        ),
    }

    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, -1, -1),
            description="Possibly occluded cube frame [1, H, W, C].",
        ),
    }

    @torch.no_grad()
    def forward(
        self,
        cube: torch.Tensor,
        frame_id: torch.Tensor,
        **_,
    ) -> dict[str, torch.Tensor]:
        return super().forward(frame_id=frame_id, cube=cube)
