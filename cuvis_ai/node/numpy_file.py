"""Numpy `.npy` source and sink nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_core.node import Node

# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _pad_to_bhwc4(array: np.ndarray) -> np.ndarray:
    """Pad array to 4D BHWC-compatible shape."""
    if array.ndim == 1:
        return array[None, None, None, :]
    if array.ndim == 2:
        return array[:, None, None, :]
    if array.ndim == 3:
        return array[None, ...]
    if array.ndim == 4:
        return array
    raise ValueError(
        f"NpyReader supports arrays with 1-4 dimensions, got shape {array.shape} (ndim={array.ndim})"
    )


class NpyReader(Node):
    """Load a `.npy` file once and return the same tensor every forward call."""

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.METADATA})

    INPUT_SPECS = {
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(1,),
            description="Optional trigger input to emit one output per frame",
            optional=True,
        )
    }

    OUTPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Loaded tensor padded to 4D BHWC-compatible shape",
        )
    }

    def __init__(self, file_path: str, **kwargs: Any) -> None:
        self.file_path = str(Path(file_path))
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"NpyReader input file not found: {path}")

        raw = np.load(path, allow_pickle=False)
        padded = _pad_to_bhwc4(np.asarray(raw, dtype=np.float32))
        tensor = torch.from_numpy(np.ascontiguousarray(padded))

        super().__init__(file_path=self.file_path, **kwargs)
        self.register_buffer("_data_buf", tensor, persistent=True)

    @torch.no_grad()
    def forward(
        self,
        frame_id: torch.Tensor | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Return cached tensor."""
        return {"data": self._data_buf}


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


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

    _category = NodeCategory.SINK
    _tags = frozenset({NodeTag.EMBEDDING, NodeTag.METADATA})

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


__all__ = ["NpyReader", "NumpyFeatureWriterNode"]
