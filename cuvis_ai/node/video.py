"""Video export nodes for writing RGB frame streams to a video file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec


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
    """

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB frames [B, H, W, 3] in [0, 1] or [0, 255]",
        )
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
        self._writer: cv2.VideoWriter | None = None
        self._frame_size: tuple[int, int] | None = None

        self.output_video_path.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(
            output_video_path=output_video_path,
            frame_rate=frame_rate,
            frame_rotation=frame_rotation,
            codec=codec,
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

    def forward(
        self,
        rgb_image: torch.Tensor,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, str]:
        """Append incoming RGB frames to the configured video file."""
        rgb_u8 = self._to_uint8_batch(rgb_image)

        for frame in rgb_u8:
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


__all__ = ["ToVideoNode"]
