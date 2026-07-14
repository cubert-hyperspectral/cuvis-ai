from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.segmentation import IntensityThresholdSegmenter

pytestmark = pytest.mark.unit


def _cube() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [
                    [0.0, 0.2, 0.4, 0.6],
                    [0.1, 0.1, 0.1, 0.1],
                ],
                [
                    [0.9, 1.0, 0.8, 0.7],
                    [0.5, 0.5, 0.5, 0.5],
                ],
            ]
        ],
        dtype=torch.float32,
    )


@torch.no_grad()
def test_intensity_segmenter_mean_reduction() -> None:
    cube = _cube()
    node = IntensityThresholdSegmenter(low=0.2, high=0.6, reduction="mean")
    result = node.forward(cube=cube)

    intensity = cube.numpy().mean(axis=-1)
    expected = ((intensity >= 0.2) & (intensity <= 0.6)).astype(np.int32)

    assert result["mask"].dtype == torch.int32
    assert result["mask"].shape == (1, 2, 2)
    assert torch.equal(result["mask"], torch.from_numpy(expected))


@torch.no_grad()
def test_intensity_segmenter_max_reduction() -> None:
    cube = _cube()
    node = IntensityThresholdSegmenter(low=0.0, high=0.9, reduction="max")
    result = node.forward(cube=cube)

    intensity = cube.numpy().max(axis=-1)
    expected = ((intensity >= 0.0) & (intensity <= 0.9)).astype(np.int32)

    assert torch.equal(result["mask"], torch.from_numpy(expected))


@torch.no_grad()
def test_intensity_segmenter_band_reduction() -> None:
    cube = _cube()
    node = IntensityThresholdSegmenter(low=0.3, high=1.0, reduction="band", band_index=2)
    result = node.forward(cube=cube)

    intensity = cube.numpy()[..., 2]
    expected = ((intensity >= 0.3) & (intensity <= 1.0)).astype(np.int32)

    assert torch.equal(result["mask"], torch.from_numpy(expected))


def test_intensity_segmenter_rejects_bad_reduction() -> None:
    with pytest.raises(ValueError):
        IntensityThresholdSegmenter(reduction="median")


def test_intensity_segmenter_rejects_inverted_interval() -> None:
    # low > high would silently yield an all-zero mask; reject it instead.
    with pytest.raises(ValueError):
        IntensityThresholdSegmenter(low=0.8, high=0.2)
