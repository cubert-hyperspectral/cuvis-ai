from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def _coerce_shape(obj: Any, name: str) -> tuple[int, ...]:
    if hasattr(obj, "shape"):
        shape = obj.shape
        if isinstance(shape, torch.Size):
            return tuple(shape)
        if isinstance(shape, Sequence):
            return tuple(int(dim) for dim in shape)
    if isinstance(obj, Sequence):
        try:
            return tuple(int(dim) for dim in obj)
        except Exception as exc:  # pragma: no cover
            raise TypeError(f"Could not interpret {name} sequence as shape: {obj}") from exc
    raise TypeError(f"{name} has no attribute `.shape` and is not a shape sequence.")


def _flatten_bhwc(x: torch.Tensor) -> torch.Tensor:
    B, H, W, C = x.shape
    return x.view(B, H * W, C)
