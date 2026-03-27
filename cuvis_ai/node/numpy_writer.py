"""Per-frame numpy feature writer node."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


class NumpyFeatureWriterNode(Node):
    """Save per-frame feature tensors to ``.npy`` files.

    Writes one ``.npy`` file per frame, named
    ``{prefix}_{frame_id:06d}.npy``.  Useful for offline analysis,
    clustering, or evaluation of ReID embeddings.

    Parameters
    ----------
    output_dir : str
        Directory to write ``.npy`` files into.
    prefix : str
        Filename prefix (default ``"features"``).
    """

    INPUT_SPECS = {
        "features": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1),
            description="Feature tensor to save, e.g. embeddings [B, N, D].",
        ),
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(1,),
            description="Frame index for file naming.",
        ),
    }

    OUTPUT_SPECS: dict[str, PortSpec] = {}  # sink node

    def __init__(
        self,
        output_dir: str,
        prefix: str = "features",
        **kwargs: Any,
    ) -> None:
        self.output_dir = str(output_dir)
        self.prefix = str(prefix)
        self._dir_created = False
        super().__init__(output_dir=self.output_dir, prefix=self.prefix, **kwargs)

    @torch.no_grad()
    def forward(self, features: Tensor, frame_id: Tensor, **_: Any) -> dict[str, Tensor]:
        """Write features to a ``.npy`` file.

        Parameters
        ----------
        features : Tensor
            ``[B, N, D]`` float32. Batch dimension is squeezed before saving.
        frame_id : Tensor
            ``(1,)`` int64 scalar frame index.

        Returns
        -------
        dict
            Empty dict (sink node).
        """
        out_dir = Path(self.output_dir)
        if not self._dir_created:
            out_dir.mkdir(parents=True, exist_ok=True)
            self._dir_created = True

        fid = int(frame_id.item())
        # Squeeze batch dim: [B, N, D] → [N, D]
        array = features.squeeze(0).cpu().numpy()
        np.save(out_dir / f"{self.prefix}_{fid:06d}.npy", array)

        return {}


__all__ = ["NumpyFeatureWriterNode"]
