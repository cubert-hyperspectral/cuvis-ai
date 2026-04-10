"""Helpers for sampled-fixed false-RGB normalization initialization."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch


def uniform_sample_positions(total_frames: int, sample_fraction: float = 0.05) -> list[int]:
    """Return deterministic uniformly spaced indices in ``[0, total_frames)``."""
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    if not (0.0 < sample_fraction <= 1.0):
        raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")

    sample_count = max(1, int(math.ceil(total_frames * float(sample_fraction))))
    if sample_count >= total_frames:
        return list(range(total_frames))
    if sample_count == 1:
        return [0]
    return [int((i * (total_frames - 1)) // (sample_count - 1)) for i in range(sample_count)]


def build_statistical_sample_stream(
    predict_ds: Any,
    sample_positions: Iterable[int],
) -> Iterable[dict[str, Any]]:
    """Yield sampled inputs in the format expected by ``statistical_initialization``."""
    for pos in sample_positions:
        sample = predict_ds[int(pos)]
        cube = torch.as_tensor(sample["cube"], dtype=torch.float32)
        if cube.ndim != 3:
            raise ValueError(f"Expected sampled cube with shape [H, W, C], got {tuple(cube.shape)}")
        wavelengths = torch.as_tensor(sample["wavelengths"]).flatten().cpu().numpy()
        yield {
            "cube": cube.unsqueeze(0),  # [1, H, W, C]
            "wavelengths": wavelengths,
        }


def initialize_false_rgb_sampled_fixed(
    false_rgb_node: Any,
    predict_ds: Any,
    sample_fraction: float = 0.05,
) -> list[int]:
    """Initialize a false-RGB selector statistically from a deterministic sample."""
    sample_positions = uniform_sample_positions(len(predict_ds), sample_fraction=sample_fraction)
    sample_stream = build_statistical_sample_stream(predict_ds, sample_positions)
    false_rgb_node.statistical_initialization(sample_stream)
    return sample_positions
