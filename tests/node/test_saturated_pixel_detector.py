from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.preprocessors import SaturatedPixelDetector

pytestmark = pytest.mark.unit


@torch.no_grad()
def test_saturated_pixel_detector_matches_numpy_reference() -> None:
    cube = torch.tensor(
        [
            [
                [
                    [1.0, 0.5, 1.0, 0.2],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [0.9, 1.2, 0.3, 1.0],
                ],
            ]
        ],
        dtype=torch.float32,
    )

    node = SaturatedPixelDetector(saturation_value=1.0, mask_threshold=0.0)
    result = node.forward(cube=cube)

    cube_np = cube.numpy()
    expected_scores = (cube_np >= 1.0).astype(np.float32).mean(axis=-1, keepdims=True)
    expected_decisions = expected_scores > 0.0

    assert result["scores"].dtype == torch.float32
    assert result["decisions"].dtype == torch.bool
    assert result["scores"].shape == (1, 2, 2, 1)
    assert result["decisions"].shape == (1, 2, 2, 1)

    assert torch.allclose(result["scores"], torch.from_numpy(expected_scores), atol=1e-6, rtol=0.0)
    assert torch.equal(result["decisions"], torch.from_numpy(expected_decisions))


@torch.no_grad()
def test_saturated_pixel_detector_respects_threshold_and_value() -> None:
    cube = torch.tensor([[[[0.5, 0.8, 0.85, 0.9]]]], dtype=torch.float32)

    node = SaturatedPixelDetector(saturation_value=0.8, mask_threshold=0.5)
    result = node.forward(cube=cube)

    cube_np = cube.numpy()
    expected_scores = (cube_np >= 0.8).astype(np.float32).mean(axis=-1, keepdims=True)
    expected_decisions = expected_scores > 0.5

    assert torch.allclose(result["scores"], torch.from_numpy(expected_scores), atol=1e-6, rtol=0.0)
    assert torch.equal(result["decisions"], torch.from_numpy(expected_decisions))
