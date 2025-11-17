from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def check_ndim(x: np.ndarray, expected_dims: Sequence[int], name: str = "array") -> None:
    """Check number of dimensions (generic; works for NumPy/Torch tensors already on CPU)."""
    if x.ndim not in expected_dims:
        raise ValueError(
            f"{name} must have ndim in {expected_dims}, got {x.ndim} (shape={x.shape})"
        )


def normalize_per_channel_vectorized(
    array: np.ndarray, range_min: float = 0.0, range_max: float = 1.0
) -> np.ndarray:
    arr = array.astype(np.float32)
    min_vals = arr.min(axis=(0, 1), keepdims=True)
    max_vals = arr.max(axis=(0, 1), keepdims=True)
    denom = np.where(max_vals > min_vals, max_vals - min_vals, 1.0)
    norm = (arr - min_vals) / denom
    norm = norm * (range_max - range_min) + range_min
    return norm


def _resolve_measurement_indices(
    indices: Sequence[int] | Iterable[int] | None, max_index: int = None
) -> list[int]:
    """Coerce, validate and store  indices."""
    print("indices:", indices, "max_index:", max_index)
    if indices is None and max_index is not None:
        resolved = list(range(max_index))
    elif isinstance(indices, range):
        resolved = list(indices)
    else:
        resolved = list(indices)

    if not resolved:
        if max_index is not None and max_index == 0:
            return []
        raise ValueError("At least one index is required.")

    invalid_indices = [idx for idx in resolved if idx < 0 or idx >= max_index]
    if invalid_indices:
        raise IndexError(
            f"Indices {invalid_indices} are out of bounds selection with {max_index} ."
        )

    if len(set(resolved)) != len(resolved):
        raise ValueError("Indices contain duplicates; provide unique indices.")

    return resolved
