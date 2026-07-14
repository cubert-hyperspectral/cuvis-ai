"""Image-file sink nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor
from torchvision.io import write_png

from cuvis_ai_core.node import Node


class PngWriter(Node):
    """Write RGB frames to PNG files on disk.

    A sink node (no output ports): it consumes an ``rgb_image`` and writes one
    PNG per frame via :func:`torchvision.io.write_png`, so the final composite
    image drops straight out of the pipeline. Input is the canonical
    ``[B, H, W, 3]`` float32 in ``[0, 1]``; it is scaled to ``uint8`` and
    written channels-first.

    Naming. A single frame with no ``frame_id`` is written to ``output_path``
    verbatim. With a ``frame_id`` (per-frame streaming) the index is appended as
    ``{stem}_{frame_id:06d}{suffix}``; a multi-frame batch is written as
    ``{stem}_{i:06d}{suffix}`` per frame.

    Parameters
    ----------
    output_path : str
        Destination PNG path. Its parent directory is created on construction.
    compression_level : int
        zlib compression level 0-9 forwarded to ``write_png`` (default 6).
    """

    _category = NodeCategory.SINK
    _tags = frozenset({NodeTag.IMAGE, NodeTag.RGB})

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB frames to write [B, H, W, 3] in [0, 1].",
        ),
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(-1,),
            description="Frame index for per-frame file naming.",
            optional=True,
        ),
    }

    OUTPUT_SPECS: dict[str, PortSpec] = {}  # sink node

    def __init__(self, output_path: str, compression_level: int = 6, **kwargs: Any) -> None:
        if not 0 <= compression_level <= 9:
            raise ValueError("compression_level must be in [0, 9]")
        self.output_path = Path(output_path)
        self.compression_level = int(compression_level)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(
            output_path=str(self.output_path),
            compression_level=self.compression_level,
            **kwargs,
        )

    def _path_for(self, i: int, n_frames: int, frame_id: int | None) -> Path:
        """Resolve the destination path for frame ``i`` of a batch of ``n_frames``."""
        if frame_id is not None and n_frames == 1:
            return self.output_path.with_name(
                f"{self.output_path.stem}_{frame_id:06d}{self.output_path.suffix}"
            )
        if n_frames == 1:
            return self.output_path
        return self.output_path.with_name(
            f"{self.output_path.stem}_{i:06d}{self.output_path.suffix}"
        )

    @torch.no_grad()
    def forward(
        self,
        rgb_image: Tensor,
        frame_id: Tensor | None = None,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Write each RGB frame to a PNG file.

        Parameters
        ----------
        rgb_image : Tensor
            ``[B, H, W, 3]`` float32 in ``[0, 1]``.
        frame_id : Tensor or None
            ``(B,)`` int64 frame index used for per-frame naming.

        Returns
        -------
        dict
            Empty dict (sink node).
        """
        n_frames = rgb_image.shape[0]
        fid = int(frame_id.reshape(-1)[0].item()) if frame_id is not None else None
        for i in range(n_frames):
            u8 = (rgb_image[i].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)  # [H, W, 3]
            chw = u8.permute(2, 0, 1).contiguous().cpu()
            write_png(chw, str(self._path_for(i, n_frames, fid)), self.compression_level)
        return {}


__all__ = ["PngWriter"]
