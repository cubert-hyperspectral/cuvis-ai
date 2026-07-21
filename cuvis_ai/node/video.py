"""Video utilities: frame iteration, datasets, Lightning DataModule, and export nodes."""

from __future__ import annotations

import os
import subprocess  # nosec B404
import warnings
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger

from cuvis_ai.utils.torch_draw import draw_text
from cuvis_ai_core.data.video import (  # noqa: F401
    VideoFrameDataModule,
    VideoFrameDataset,
    VideoIterator,
)
from cuvis_ai_core.node import Node

# Resolve the bundled ffmpeg binary once at import. imageio_ffmpeg validates the
# binary by spawning subprocess.Popen, so resolving lazily inside the class
# would re-trigger validation under test monkeypatches of subprocess.Popen.
try:
    _BUNDLED_FFMPEG_BIN: str | None = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # noqa: BLE001 — bundled binary missing/corrupt; surfaces at use
    _BUNDLED_FFMPEG_BIN = None


# ---------------------------------------------------------------------------
# _FrameRenderMixin — RGB-frame rendering shared by the video/image sink nodes
# ---------------------------------------------------------------------------
class _FrameRenderMixin:
    """Frame-rendering helpers shared by the RGB sink nodes (video and image).

    Provides rotation normalization/application, batch uint8 conversion, and the
    optional centered-title overlay, so ``ToVideoNode`` and ``ToImage`` prepare
    frames identically before their differing write backends take over.
    """

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


# ---------------------------------------------------------------------------
# ToVideoNode — write RGB frame batches to a video file via ffmpeg subprocess
# ---------------------------------------------------------------------------
class ToVideoNode(_FrameRenderMixin, Node):
    """Write incoming RGB frames directly to a video file via ffmpeg.

    This node lazily starts a single ``ffmpeg`` subprocess on the first frame and
    pipes raw ``rgb24`` bytes to its stdin; ffmpeg handles encoding, bitrate
    control, and muxing. ``close()`` sends EOF and waits for ffmpeg to flush the
    trailer — callers must invoke it explicitly (e.g. in a ``finally`` block of
    the enclosing pipeline driver) to surface encoder errors.

    The ffmpeg binary is resolved via ``imageio_ffmpeg`` by default (bundled with
    the wheel — no system install needed). Override with the
    ``CUVIS_AI_FFMPEG_BIN`` environment variable to point at a custom build
    (e.g. one with ``h264_nvenc`` / ``vaapi`` / ``amf`` hardware encoders).

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
    video_codec : str, optional
        ffmpeg ``-c:v`` codec name (e.g. ``"libx264"``, ``"libx265"``).
        Default is ``"libx264"``.
    bitrate : str, optional
        ffmpeg ``-b:v`` target bitrate (e.g. ``"12M"``, ``"8000k"``).
        Default is ``"12M"``.
    overlay_title : str | None, optional
        Optional static title rendered at the top center with its own slim
        darkened background block. Default is ``None``.
    write_mode : str, optional
        How the mp4 is finalized. ``"full"`` (default) writes a standard file
        with ``-movflags +faststart`` (the ``moov`` atom is moved to the front on
        a clean ``close()``); best for a finished run whose driver calls
        ``close()``, but unreadable until then. ``"partial"`` writes a fragmented
        mp4 (``-movflags +frag_keyframe+empty_moov+default_base_moof``) that stays
        playable during recording and after an unclean stop; use it for a
        streaming / gRPC session with no guaranteed driver ``close()``.
    """

    _category = NodeCategory.SINK
    _tags = frozenset({NodeTag.VIDEO})

    _WRITE_MODE_MOVFLAGS = {
        "full": "+faststart",
        "partial": "+frag_keyframe+empty_moov+default_base_moof",
    }

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

    OUTPUT_SPECS: dict[str, PortSpec] = {}  # sink node

    def __init__(
        self,
        output_video_path: str,
        frame_rate: float = 10.0,
        frame_rotation: int | None = None,
        video_codec: str = "libx264",
        bitrate: str = "12M",
        overlay_title: str | None = None,
        write_mode: str = "full",
        **kwargs: Any,
    ) -> None:
        if frame_rate <= 0:
            raise ValueError("frame_rate must be > 0")
        if not isinstance(video_codec, str) or not video_codec.strip():
            raise ValueError("video_codec must be a non-empty string")
        if not isinstance(bitrate, str) or not bitrate.strip():
            raise ValueError("bitrate must be a non-empty string (e.g. '12M', '8000k')")
        if write_mode not in self._WRITE_MODE_MOVFLAGS:
            raise ValueError(
                f"write_mode must be one of {sorted(self._WRITE_MODE_MOVFLAGS)}, got {write_mode!r}"
            )
        valid_rotations = {None, 0, 90, -90, 180, -180, 270, -270}
        if frame_rotation not in valid_rotations:
            raise ValueError(
                "frame_rotation must be one of: None, 0, 90, -90, 180, -180, 270, -270"
            )

        self.output_video_path = Path(output_video_path)
        self.frame_rate = float(frame_rate)
        self.frame_rotation = self._normalize_rotation(frame_rotation)
        self.video_codec = video_codec.strip()
        self.bitrate = bitrate.strip()
        self.write_mode = write_mode
        self.movflags = self._WRITE_MODE_MOVFLAGS[write_mode]
        self.overlay_title = (
            None
            if overlay_title is None or not str(overlay_title).strip()
            else str(overlay_title).strip()
        )
        if self.overlay_title:
            warnings.warn(
                "ToVideoNode renders overlay_title with cv2; it will move to the shared torch "
                "text renderer (cuvis_ai.utils.torch_draw.draw_text) in v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._proc: subprocess.Popen[bytes] | None = None
        self._frame_size: tuple[int, int] | None = None
        # Records a teardown-time finalize failure (see cleanup()); None until one happens.
        self._finalize_error: str | None = None

        self.output_video_path.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(
            output_video_path=output_video_path,
            frame_rate=frame_rate,
            frame_rotation=frame_rotation,
            video_codec=self.video_codec,
            bitrate=self.bitrate,
            overlay_title=self.overlay_title,
            write_mode=write_mode,
            **kwargs,
        )

    @staticmethod
    def _resolve_ffmpeg_binary() -> str:
        """Return the ffmpeg binary path, honoring CUVIS_AI_FFMPEG_BIN override."""
        override = os.environ.get("CUVIS_AI_FFMPEG_BIN", "").strip()
        if override:
            return override
        if _BUNDLED_FFMPEG_BIN is None:
            raise RuntimeError(
                "imageio_ffmpeg bundled ffmpeg binary not found and "
                "CUVIS_AI_FFMPEG_BIN is unset; reinstall imageio-ffmpeg or "
                "point CUVIS_AI_FFMPEG_BIN at a working ffmpeg"
            )
        return _BUNDLED_FFMPEG_BIN

    def _build_ffmpeg_argv(self, height: int, width: int) -> list[str]:
        """Build the ffmpeg argv for a raw rgb24 stdin pipe -> encoded file."""
        return [
            self._resolve_ffmpeg_binary(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.frame_rate),
            "-i",
            "pipe:",
            "-c:v",
            self.video_codec,
            "-b:v",
            self.bitrate,
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-movflags",
            self.movflags,
            str(self.output_video_path),
        ]

    def _init_ffmpeg(self, height: int, width: int) -> None:
        """Spawn the ffmpeg subprocess lazily on first frame."""
        argv = self._build_ffmpeg_argv(height=height, width=width)
        try:
            proc = subprocess.Popen(  # nosec B603
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"ffmpeg binary not found at {argv[0]!r} — "
                "the bundled imageio_ffmpeg binary is missing or the "
                "CUVIS_AI_FFMPEG_BIN override points at a non-existent path"
            ) from exc
        self._proc = proc
        self._frame_size = (height, width)

    def _collect_stderr_after_exit(self) -> str:
        """Wait for ffmpeg to exit (short timeout) and return its full stderr text.

        Only safe to call when ffmpeg has errored or is expected to exit imminently:
        the ``stderr.read()`` is blocking and only returns once the writer end is
        closed by the child. Waiting for the process first guarantees that.
        """
        if self._proc is None:
            return ""
        if self._proc.poll() is None:
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    return "<ffmpeg did not terminate after kill()>"
        if self._proc.stderr is None:
            return ""
        try:
            data = self._proc.stderr.read() or b""
        except (ValueError, OSError):
            return ""
        return data.decode("utf-8", errors="replace")

    def forward(
        self,
        rgb_image: torch.Tensor,
        frame_id: torch.Tensor | None = None,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, Any]:
        """Append incoming RGB frames to the configured video file.

        Returns
        -------
        dict
            Empty dict (sink node).
        """
        rgb_u8 = self._to_uint8_batch(rgb_image)

        for b, frame in enumerate(rgb_u8):
            self._draw_title_overlay(frame)
            if frame_id is not None and b < len(frame_id):
                fid = int(frame_id[b].item())
                draw_text(frame, 8, 8, f"frame {fid}", (255, 255, 255), scale=2, bg=True)
            frame = self._rotate_frame(frame)
            height, width = int(frame.shape[0]), int(frame.shape[1])
            if self._proc is None:
                self._init_ffmpeg(height=height, width=width)
            elif self._frame_size != (height, width):
                raise ValueError(
                    f"All frames must share one size. Expected {self._frame_size}, got {(height, width)}"
                )

            assert self._proc is not None and self._proc.stdin is not None
            frame_bytes = np.ascontiguousarray(frame.numpy()).tobytes()
            try:
                self._proc.stdin.write(frame_bytes)
            except (BrokenPipeError, OSError) as exc:
                stderr_text = self._collect_stderr_after_exit()
                returncode = self._proc.poll() if self._proc is not None else None
                raise RuntimeError(
                    f"ffmpeg exited during frame write (returncode={returncode}): {stderr_text}"
                ) from exc

        return {}

    def close(self) -> None:
        """Flush EOF to ffmpeg, wait for mux, and surface any encoder errors.

        Idempotent — repeated calls are no-ops. Must be called explicitly by the
        pipeline driver; do not rely on ``__del__`` for normal teardown.
        """
        proc = self._proc
        if proc is None:
            return
        self._proc = None

        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError) as exc:
                logger.debug("ffmpeg stdin close raised during teardown: {}", exc)

        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"ffmpeg did not terminate for {self.output_video_path} (kill after 30s wait)"
                ) from None
            stderr_text = b""
            if proc.stderr is not None:
                try:
                    stderr_text = proc.stderr.read() or b""
                except (ValueError, OSError):
                    stderr_text = b""
            raise RuntimeError(
                "ffmpeg timed out during mux of "
                f"{self.output_video_path}: {stderr_text.decode('utf-8', errors='replace')}"
            ) from None

        stderr_text = b""
        if proc.stderr is not None:
            try:
                stderr_text = proc.stderr.read() or b""
            except (ValueError, OSError):
                stderr_text = b""

        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with non-zero return code {proc.returncode} "
                f"for {self.output_video_path}: "
                f"{stderr_text.decode('utf-8', errors='replace')}"
            )

    def cleanup(self) -> None:
        """Finalize the video file when the hosting pipeline is torn down.

        A gRPC/session pipeline has no explicit driver ``close()`` call, so the
        session-teardown ``cleanup()`` (invoked by ``CuvisPipeline.cleanup`` on
        session close, pipeline replacement, or run stop) is where the ffmpeg
        trailer gets flushed. ``close()`` is idempotent, so calling it here in
        addition to an explicit driver ``close()`` is safe.

        ``CuvisPipeline.cleanup`` wraps each node's ``cleanup()`` in a bare
        try/except that only ``logger.warning``s, so a finalize failure here
        (ffmpeg unable to write the ``moov`` trailer, leaving an unplayable file)
        would otherwise be indistinguishable from a benign teardown warning while
        the run still reports success. Surface it explicitly at ERROR level and
        record it on ``_finalize_error`` so callers/tests can detect the truncated
        output, then re-raise so nothing is silently hidden.
        """
        try:
            self.close()
        except RuntimeError as exc:
            self._finalize_error = str(exc)
            logger.error(
                "ToVideoNode failed to finalize {} at pipeline teardown: {}",
                self.output_video_path,
                exc,
            )
            raise
        finally:
            super().cleanup()

    def __del__(self) -> None:
        """Best-effort cleanup; do not rely on this for normal teardown."""
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception as exc:
            logger.debug("Failed to kill ffmpeg during __del__: {}", exc)


# ---------------------------------------------------------------------------
# ToImage — write RGB frame batches to individual image files
# ---------------------------------------------------------------------------
class ToImage(_FrameRenderMixin, Node):
    """Write incoming RGB frames to individual image files, one file per frame.

    Mirrors :class:`ToVideoNode` but emits a standalone image per frame instead
    of an encoded video stream. Each file is written immediately and is complete
    on disk the moment ``forward`` returns, so there is no lazy encoder process
    and no explicit ``close()`` / finalization step (and none of the fragmented
    ``movflags`` playability caveats a streaming video has).

    The output name comes from ``filename_pattern`` with the frame index
    substituted (the ``{frame_id}`` field); the image format is inferred from the
    pattern's file extension (for example ``.png`` or ``.jpg``). When the batch
    carries a ``frame_id`` port, that value drives both the filename and the text
    overlay; otherwise a running per-node counter is used. A pattern without a
    ``{frame_id}`` field writes every frame to the same file (last wins).

    Parameters
    ----------
    output_dir : str
        Directory the image files are written to. Created if missing.
    filename_pattern : str, optional
        ``str.format`` pattern for each file's name, receiving ``frame_id`` as a
        keyword field. The extension selects the image format. Default is
        ``"frame_{frame_id:06d}.png"``.
    frame_rotation : int | None, optional
        Optional frame rotation in degrees; same semantics and accepted values as
        :class:`ToVideoNode`. Default is ``None`` (no rotation).
    overlay_title : str | None, optional
        Optional static title rendered at the top center with its own darkened
        background block. Default is ``None``.
    """

    _category = NodeCategory.SINK
    _tags = frozenset({NodeTag.IMAGE})

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

    OUTPUT_SPECS: dict[str, PortSpec] = {}  # sink node

    def __init__(
        self,
        output_dir: str,
        filename_pattern: str = "frame_{frame_id:06d}.png",
        frame_rotation: int | None = None,
        overlay_title: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError("output_dir must be a non-empty string")
        if not isinstance(filename_pattern, str) or not filename_pattern.strip():
            raise ValueError("filename_pattern must be a non-empty string")
        if not Path(filename_pattern).suffix:
            raise ValueError(
                "filename_pattern must include an image extension (e.g. '.png', '.jpg')"
            )
        valid_rotations = {None, 0, 90, -90, 180, -180, 270, -270}
        if frame_rotation not in valid_rotations:
            raise ValueError(
                "frame_rotation must be one of: None, 0, 90, -90, 180, -180, 270, -270"
            )

        self.output_dir = Path(output_dir)
        self.filename_pattern = filename_pattern
        self.frame_rotation = self._normalize_rotation(frame_rotation)
        self.overlay_title = (
            None
            if overlay_title is None or not str(overlay_title).strip()
            else str(overlay_title).strip()
        )
        self._frame_counter = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(
            output_dir=output_dir,
            filename_pattern=filename_pattern,
            frame_rotation=frame_rotation,
            overlay_title=self.overlay_title,
            **kwargs,
        )

    def _write_frame(self, frame: torch.Tensor, fid: int) -> None:
        """Write one uint8 HWC RGB frame to ``output_dir`` as an image file."""
        rgb_np = np.ascontiguousarray(frame.numpy())
        bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)  # cv2.imwrite expects BGR
        path = self.output_dir / self.filename_pattern.format(frame_id=fid)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), bgr_np):
            raise RuntimeError(f"cv2.imwrite failed to write frame to {path}")

    def forward(
        self,
        rgb_image: torch.Tensor,
        frame_id: torch.Tensor | None = None,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, Any]:
        """Write each incoming RGB frame to its own image file.

        Returns
        -------
        dict
            Empty dict (sink node).
        """
        rgb_u8 = self._to_uint8_batch(rgb_image)

        for b, frame in enumerate(rgb_u8):
            self._draw_title_overlay(frame)
            if frame_id is not None and b < len(frame_id):
                fid = int(frame_id[b].item())
                draw_text(frame, 8, 8, f"frame {fid}", (255, 255, 255), scale=2, bg=True)
            else:
                fid = self._frame_counter
            frame = self._rotate_frame(frame)
            self._write_frame(frame, fid)
            self._frame_counter += 1

        return {}


# ---------------------------------------------------------------------------
# VideoFrameNode — passthrough source node for RGB frames
# ---------------------------------------------------------------------------
class VideoFrameNode(Node):
    """Passthrough source node that receives RGB frames from the batch."""

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.VIDEO, NodeTag.STREAMING})

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
