"""Numpy-backed constant source node."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_core.node import Node


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
    """Load a `.npy` file once and return the same tensor every forward call.

    Supports two modes:

    1. **File mode** (file_path is a string):
       - Loads a `.npy` file at initialization
       - Buffer is populated immediately and persisted with pipeline weights

    2. **Buffer mode** (file_path is None):
       - Initializes with an empty buffer [0]
       - Buffer is populated via load_from_array() or statistical_initialization()
       - Useful for pipelines where learned reference vectors are baked into weights
    """

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.METADATA})

    TRAINABLE_BUFFERS = ("_data_buf",)

    INPUT_SPECS = {
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(1,),
            description="Optional trigger input to emit one output per frame",
            optional=True,
        ),
        "signatures": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1),
            description="Port-stream signatures [B,N,C] for statistical_initialization",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Loaded tensor padded to 4D BHWC-compatible shape",
        )
    }

    def __init__(self, file_path: str | None = None, **kwargs: Any) -> None:
        """Initialize NpyReader in either file or buffer mode.

        Parameters
        ----------
        file_path : str | None
            Path to a .npy file (file mode) or None (buffer mode).
        **kwargs
            Passed to parent Node.
        """
        self.file_path = file_path
        self._statistically_initialized = False

        # File mode: load immediately
        if file_path is not None:
            self.file_path = str(Path(file_path))
            path = Path(self.file_path)
            if not path.exists():
                raise FileNotFoundError(f"NpyReader input file not found: {path}")

            raw = np.load(path, allow_pickle=False)
            padded = _pad_to_bhwc4(np.asarray(raw, dtype=np.float32))
            tensor = torch.from_numpy(np.ascontiguousarray(padded))
            self._statistically_initialized = True
        else:
            # Buffer mode: start empty, will be filled via load_from_array() or statistical_initialization()
            tensor = torch.empty(0, dtype=torch.float32)

        super().__init__(file_path=self.file_path, **kwargs)
        self.register_buffer("_data_buf", tensor, persistent=True)

    def load_from_array(self, data: np.ndarray | torch.Tensor) -> None:
        """Populate buffer from a numpy array or torch tensor.

        Parameters
        ----------
        data : np.ndarray | torch.Tensor
            Input array/tensor to load. Shape can be 1-4D; will be padded to [N,1,1,C] or [B,H,W,C].
        """
        if isinstance(data, torch.Tensor):
            # Convert tensor to numpy for padding
            data_np = data.to(torch.float32).cpu().numpy()
        elif isinstance(data, np.ndarray):
            data_np = data.astype(np.float32)
        else:
            raise TypeError(f"Expected ndarray or Tensor, got {type(data)}")

        padded = _pad_to_bhwc4(data_np)
        tensor = torch.from_numpy(np.ascontiguousarray(padded, dtype=np.float32))

        self._data_buf = tensor
        self._statistically_initialized = True

    def statistical_initialization(self, input_stream: list[dict]) -> None:
        """Populate buffer from a port-stream of signature items.

        Processes items with 'signatures' or 'data' keys, squeezes batch dimension,
        and accumulates (mean-averages) multiple items.

        Parameters
        ----------
        input_stream : list[dict]
            Stream items, each with optional 'signatures' or 'data' key containing [B,N,C] or [B,H,W,C].

        Raises
        ------
        RuntimeError
            If no usable tensors are found (all empty or missing keys).
        """
        items = []
        for item in input_stream:
            # Try 'signatures' key first, fall back to 'data'
            tensor = item.get("signatures")
            if tensor is None:
                tensor = item.get("data")
            if tensor is None:
                continue

            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor.astype(np.float32))
            elif isinstance(tensor, torch.Tensor):
                tensor = tensor.to(torch.float32)
            else:
                continue

            # Skip empty tensors (frames with no annotations)
            if tensor.numel() == 0:
                continue

            items.append(tensor)

        if not items:
            raise RuntimeError(
                "statistical_initialization: no usable tensors found in stream. "
                "Expected items with 'signatures' [B,N,C] or 'data' [B,H,W,C] keys."
            )

        # Squeeze batch dimension and stack
        squeezed = [item.squeeze(0) if item.ndim > 0 else item for item in items]

        # Concatenate along first dimension
        concatenated = torch.cat(squeezed, dim=0)

        # If multiple items, compute mean along batch
        if len(items) > 1:
            accumulated = concatenated.mean(dim=0, keepdim=True)
        else:
            accumulated = concatenated

        # Pad to BHWC format
        padded_np = _pad_to_bhwc4(accumulated.cpu().numpy())
        tensor = torch.from_numpy(np.ascontiguousarray(padded_np, dtype=np.float32))

        self._data_buf = tensor
        self._statistically_initialized = True

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Load state dict with special handling for empty placeholder tensors.

        When buffer mode is loaded from a .pt file, the buffer is resized
        to match the incoming state before PyTorch's default loader restores it.
        """
        if "_data_buf" in state_dict and self._data_buf.numel() == 0:
            # Resize empty placeholder to match incoming shape
            incoming = state_dict["_data_buf"]
            self._data_buf = torch.empty_like(incoming)

        super().load_state_dict(state_dict, strict=strict)

    @torch.no_grad()
    def forward(
        self,
        frame_id: torch.Tensor | None = None,  # noqa: ARG002
        signatures: torch.Tensor | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Return cached tensor, ignoring any live input signatures.

        In buffer mode, this always emits the learned signatures stored in _data_buf,
        regardless of what's passed on the port. At inference time, the signatures
        from the training phase (persisted in the .pt file) are used.
        """
        if self._data_buf.numel() == 0:
            raise RuntimeError(
                "NpyReader buffer is empty. In buffer mode (file_path=None), "
                "call load_from_array() or statistical_initialization() before forward()."
            )
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
    def forward(
        self, features: torch.Tensor, frame_id: torch.Tensor, **_: Any
    ) -> dict[str, torch.Tensor]:
        """Write features to a ``.npy`` file.

        Parameters
        ----------
        features : torch.Tensor
            ``[B, N, D]`` float32. Batch dimension is squeezed before saving.
        frame_id : torch.Tensor
            ``(1,)`` int64 scalar frame index.

        Returns
        -------
        dict
            Empty dict (sink node).
        """
        from pathlib import Path

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
