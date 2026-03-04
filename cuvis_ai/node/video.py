"""Video utilities: frame iteration, datasets, Lightning DataModule, and export nodes."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger
from torch.utils.data import DataLoader, Dataset


def _import_torchcodec() -> type:
    """Lazy import for torchcodec (requires FFmpeg native libraries at runtime)."""
    return importlib.import_module("torchcodec.decoders").SimpleVideoDecoder


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


# ---------------------------------------------------------------------------
# VideoIterator — frame-level access to MP4/AVI files via torchcodec
# ---------------------------------------------------------------------------
class VideoIterator:
    """Iterate over frames of an MP4/AVI video via torchcodec."""

    def __init__(self, source_path: str) -> None:
        self.source_path = source_path
        assert Path(source_path).exists(), f"Video file {source_path} does not exist"

        _SimpleVideoDecoder = _import_torchcodec()
        self.video_decoder = _SimpleVideoDecoder(source_path)
        self.num_frames: int = len(self.video_decoder)

        self.enable_random_access = True
        if self.num_frames < 0:
            logger.error(
                "Cannot determine number of frames. Random access is disabled: {}",
                source_path,
            )
            self.enable_random_access = False
            self.num_frames = 0

        self.frame_rate: float = self.video_decoder.metadata.average_fps
        self.basename: str = Path(source_path).stem

        first_frame = self.video_decoder[0]
        self.image_width: int = first_frame.shape[2]
        self.image_height: int = first_frame.shape[1]

    def __len__(self) -> int:
        return self.num_frames

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for frame_id in range(len(self)):
            yield self.get_frame(frame_id)

    def get_frame(self, frame_id: int) -> dict[str, Any]:
        try:
            frame = self.video_decoder[frame_id].permute(1, 2, 0).numpy()
        except Exception as e:
            logger.error("Error reading frame {}: {}", frame_id, e)
            return {"frame_id": frame_id, "image": np.zeros((1, 1, 3), dtype=np.uint8)}

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return {"frame_id": frame_id, "image": frame_bgr, "basename": self.basename}


# ---------------------------------------------------------------------------
# VideoFrameDataset — torch Dataset wrapping VideoIterator
# ---------------------------------------------------------------------------
class VideoFrameDataset(Dataset):
    """Torch map-style dataset that yields RGB float32 tensors from a video."""

    def __init__(self, video_iter: VideoIterator, end_frame: int = -1) -> None:
        self.video_iter = video_iter
        total = len(video_iter)
        self.n_frames = min(end_frame, total) if end_frame > 0 else total
        self.fps: float = video_iter.frame_rate

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        frame_data = self.video_iter.get_frame(idx)
        bgr = frame_data["image"]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_f32 = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        return {"rgb_image": rgb_f32}


# ---------------------------------------------------------------------------
# VideoFrameDataModule — LightningDataModule for the Predictor
# ---------------------------------------------------------------------------
class VideoFrameDataModule(pl.LightningDataModule):
    """DataModule that reads MP4 frames for use with cuvis-ai Predictor."""

    def __init__(self, video_path: str, end_frame: int = -1, batch_size: int = 1) -> None:
        super().__init__()
        self.video_path = video_path
        self.end_frame = end_frame
        self.batch_size = batch_size
        self.predict_ds: VideoFrameDataset | None = None
        self.fps: float = 10.0

    def setup(self, stage: str | None = None) -> None:
        if stage == "predict" or stage is None:
            video_iter = VideoIterator(self.video_path)
            self.predict_ds = VideoFrameDataset(video_iter, end_frame=self.end_frame)
            self.fps = video_iter.frame_rate
            if self.fps <= 0:
                self.fps = 10.0

    def predict_dataloader(self) -> DataLoader:
        if self.predict_ds is None:
            raise RuntimeError("Predict dataset not initialized. Call setup('predict') first.")
        return DataLoader(self.predict_ds, shuffle=False, batch_size=self.batch_size, num_workers=0)


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
    }
    OUTPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB frame [B, H, W, 3] in [0, 1].",
        ),
    }

    def forward(self, rgb_image: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        return {"rgb_image": rgb_image}


__all__ = [
    "ToVideoNode",
    "VideoFrameDataModule",
    "VideoFrameDataset",
    "VideoFrameNode",
    "VideoIterator",
]
