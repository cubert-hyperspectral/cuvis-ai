"""Video utilities: frame iteration, datasets, Lightning DataModule, and export nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from cuvis_ai_core.data.video import (  # noqa: F401
    VideoFrameDataModule,
    VideoFrameDataset,
    VideoIterator,
)
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.utils.torch_draw import draw_text

# ---------------------------------------------------------------------------
# ToVideoNode — write RGB frame batches to MP4 via OpenCV
# ---------------------------------------------------------------------------
class ToVideoNode(Node):
    """Write incoming RGB frames directly to a video file.

    This node opens a single OpenCV ``VideoWriter`` and appends frames on each
    ``forward`` call. It is intended for streaming pipelines where frames arrive
    incrementally.

    Parameters
    ----------
    output_video_path : str
        Output path for the generated video file (for example ``.mp4``).
    frame_rate : float, optional
        Video frame rate in frames per second. Must be positive. Default is ``10.0``.
    frame_rotation : int | None, optional
        Optional frame rotation in degrees. Supported values are ``-90``, ``90``, ``180``
        (and aliases ``270``, ``-270``, ``-180``). Positive values rotate
        anticlockwise (counterclockwise), negative values rotate clockwise.
        Default is ``None`` (no rotation).
    codec : str, optional
        FourCC codec string (length 4). Default is ``"mp4v"``.
    overlay_title : str | None, optional
        Optional static title rendered at the top center with its own slim
        darkened background block. Default is ``None``.
    """

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB frames [B, H, W, 3] in [0, 1] or [0, 255]",
        ),
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(-1,),
            description="Frame / measurement index [B] to render as text overlay.",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "video_path": PortSpec(
            dtype=str,
            shape=(),
            description="Path to the output video file",
        )
    }

    def __init__(
        self,
        output_video_path: str,
        frame_rate: float = 10.0,
        frame_rotation: int | None = None,
        codec: str = "mp4v",
        overlay_title: str | None = None,
        **kwargs: Any,
    ) -> None:
        if frame_rate <= 0:
            raise ValueError("frame_rate must be > 0")
        if len(codec) != 4:
            raise ValueError("codec must be a 4-character FourCC string")
        valid_rotations = {None, 0, 90, -90, 180, -180, 270, -270}
        if frame_rotation not in valid_rotations:
            raise ValueError(
                "frame_rotation must be one of: None, 0, 90, -90, 180, -180, 270, -270"
            )

        self.output_video_path = Path(output_video_path)
        self.frame_rate = float(frame_rate)
        self.frame_rotation = self._normalize_rotation(frame_rotation)
        self.codec = codec
        self.overlay_title = (
            None
            if overlay_title is None or not str(overlay_title).strip()
            else str(overlay_title).strip()
        )
        self._writer: cv2.VideoWriter | None = None
        self._frame_size: tuple[int, int] | None = None

        self.output_video_path.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(
            output_video_path=output_video_path,
            frame_rate=frame_rate,
            frame_rotation=frame_rotation,
            codec=codec,
            overlay_title=self.overlay_title,
            **kwargs,
        )

    @staticmethod
    def _normalize_rotation(frame_rotation: int | None) -> int | None:
        """Normalize equivalent rotation aliases to {-90, 90, 180} or None."""
        if frame_rotation in (None, 0):
            return None
        if frame_rotation in (180, -180):
            return 180
        if frame_rotation in (90, -270):
            return 90
        if frame_rotation in (-90, 270):
            return -90
        return frame_rotation

    def _rotate_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Rotate one frame according to configured frame_rotation."""
        if self.frame_rotation is None:
            return frame
        if self.frame_rotation == 90:
            return torch.rot90(frame, k=1, dims=(0, 1))
        if self.frame_rotation == -90:
            return torch.rot90(frame, k=-1, dims=(0, 1))
        if self.frame_rotation == 180:
            return torch.rot90(frame, k=2, dims=(0, 1))
        return frame

    def _init_writer(self, height: int, width: int) -> None:
        """Initialize the OpenCV writer lazily on first frame."""
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(
            str(self.output_video_path),
            fourcc,
            self.frame_rate,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer at {self.output_video_path}")
        self._writer = writer
        self._frame_size = (height, width)

    @staticmethod
    def _to_uint8_batch(rgb_image: torch.Tensor) -> torch.Tensor:
        """Convert input RGB frames to uint8 tensor on CPU."""
        if rgb_image.ndim != 4 or rgb_image.shape[-1] != 3:
            raise ValueError(
                f"Expected rgb_image with shape [B, H, W, 3], got {tuple(rgb_image.shape)}"
            )

        rgb_cpu = rgb_image.detach().cpu()
        if torch.is_floating_point(rgb_cpu):
            return (rgb_cpu.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        if rgb_cpu.dtype != torch.uint8:
            rgb_cpu = rgb_cpu.clamp(0, 255).to(torch.uint8)
        return rgb_cpu

    @staticmethod
    def _darken_region(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
        """Darken an RGB uint8 region in-place for text readability."""
        if x1 <= x0 or y1 <= y0:
            return
        region = frame[y0:y1, x0:x1]
        if region.size == 0:
            return
        region[:] = np.rint(region.astype(np.float32) * 0.25).astype(np.uint8)

    def _draw_title_overlay(self, frame: torch.Tensor) -> None:
        """Render an optional centered title overlay in-place on a uint8 HWC frame."""
        if not self.overlay_title:
            return

        frame_np = np.ascontiguousarray(frame.numpy())
        frame_h, frame_w = int(frame_np.shape[0]), int(frame_np.shape[1])
        if frame_h <= 0 or frame_w <= 0:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        line_type = cv2.LINE_AA
        margin_y = 8
        reserved_side_margin = 96
        fallback_side_margin = 8
        max_box_width = frame_w - 2 * reserved_side_margin
        if max_box_width <= 0:
            max_box_width = frame_w - 2 * fallback_side_margin

        chosen_scale = 0.35
        chosen_thickness = 1
        text_width = 0
        text_height = 0
        baseline = 0
        for font_scale in (0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35):
            thickness = 2 if font_scale >= 0.55 else 1
            pad_x = 8 if font_scale >= 0.55 else 6
            pad_y = 6 if font_scale >= 0.55 else 4
            (candidate_width, candidate_height), candidate_baseline = cv2.getTextSize(
                self.overlay_title, font, font_scale, thickness
            )
            candidate_box_width = int(candidate_width) + 2 * pad_x
            chosen_scale = font_scale
            chosen_thickness = thickness
            text_width = int(candidate_width)
            text_height = int(candidate_height)
            baseline = int(candidate_baseline)
            if candidate_box_width <= max_box_width:
                break

        pad_x = 8 if chosen_scale >= 0.55 else 6
        pad_y = 6 if chosen_scale >= 0.55 else 4
        box_width = int(text_width) + 2 * pad_x
        box_height = int(text_height) + int(baseline) + 2 * pad_y

        x0 = max(0, (frame_w - box_width) // 2)
        y0 = max(0, margin_y)
        x1 = min(frame_w, x0 + box_width)
        y1 = min(frame_h, y0 + box_height)
        self._darken_region(frame_np, x0=x0, y0=y0, x1=x1, y1=y1)

        text_origin = (
            min(frame_w - 1, x0 + pad_x),
            min(frame_h - 1, y0 + pad_y + int(text_height)),
        )
        cv2.putText(
            frame_np,
            self.overlay_title,
            text_origin,
            font,
            chosen_scale,
            (255, 255, 255),
            chosen_thickness,
            line_type,
        )

        frame.copy_(torch.from_numpy(frame_np))

    def forward(
        self,
        rgb_image: torch.Tensor,
        frame_id: torch.Tensor | None = None,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, str]:
        """Append incoming RGB frames to the configured video file."""
        rgb_u8 = self._to_uint8_batch(rgb_image)

        for b, frame in enumerate(rgb_u8):
            self._draw_title_overlay(frame)
            if frame_id is not None and b < len(frame_id):
                fid = int(frame_id[b].item())
                draw_text(frame, 8, 8, f"frame {fid}", (255, 255, 255), scale=2, bg=True)
            frame = self._rotate_frame(frame)
            height, width = int(frame.shape[0]), int(frame.shape[1])
            if self._writer is None:
                self._init_writer(height=height, width=width)
            elif self._frame_size != (height, width):
                raise ValueError(
                    f"All frames must share one size. Expected {self._frame_size}, got {(height, width)}"
                )

            # RGB -> BGR for OpenCV writer
            bgr_frame = frame[..., [2, 1, 0]].numpy()
            self._writer.write(bgr_frame)

        return {"video_path": str(self.output_video_path)}

    def close(self) -> None:
        """Release the underlying video writer if it exists."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __del__(self) -> None:
        """Best-effort cleanup for writer handle."""
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# VideoFrameNode — passthrough source node for RGB frames
# ---------------------------------------------------------------------------
class VideoFrameNode(Node):
    """Passthrough source node that receives RGB frames from the batch."""

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB frame [B, H, W, 3] in [0, 1].",
        ),
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(-1,),
            description="Frame index [B].",
            optional=True,
        ),
    }
    OUTPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB frame [B, H, W, 3] in [0, 1].",
        ),
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(-1,),
            description="Frame index [B].",
        ),
    }

    def forward(
        self,
        rgb_image: torch.Tensor,
        frame_id: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Pass through RGB frames and optional frame IDs from the batch."""
        result: dict[str, torch.Tensor] = {"rgb_image": rgb_image}
        if frame_id is not None:
            result["frame_id"] = frame_id
        return result


__all__ = [
    "ToVideoNode",
    "VideoFrameDataModule",
    "VideoFrameDataset",
    "VideoFrameNode",
    "VideoIterator",
]
