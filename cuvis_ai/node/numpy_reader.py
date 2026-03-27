"""Numpy-backed constant source node."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec


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


__all__ = ["NpyReader"]
