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


def check_array_shape(x: Any, desired_shape: Sequence[int], name: str = "Tensor") -> None:
    """Validate that x.shape matches desired_shape, where -1 is a wildcard."""
    actual = _coerce_shape(x, name)
    desired = tuple(desired_shape)

    # quick spec sanity
    if any((not isinstance(d, int)) or d < -1 for d in desired):
        raise TypeError("desired_shape entries must be -1 or non-negative ints.")

    # rank check
    if len(actual) != len(desired):
        raise ValueError(
            f"{name} rank mismatch: expected {len(desired)}D per spec {desired}, got {len(actual)}D (shape={actual})."
        )

    # per-dim check
    for i, (a, d) in enumerate(zip(actual, desired)):
        if d != -1 and a != d:
            raise ValueError(
                f"{name} dim {i} mismatch: expected {d}, got {a}. shape={actual}, spec={desired}"
            )
    return True


def _flatten_bhwc(x: torch.Tensor) -> torch.Tensor:
    B, H, W, C = x.shape
    return x.view(B, H * W, C)
