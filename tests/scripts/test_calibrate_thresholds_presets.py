"""The binary and quantile preset metrics of calibrate-thresholds share one pixel-count rule."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.calibrate_thresholds import _current_preset_binary, _current_preset_quantile

pytestmark = pytest.mark.unit


def _frames() -> tuple[np.ndarray, np.ndarray]:
    # Two 2x2 frames: frame 0 is clean, frame 1 carries one hot anomalous pixel.
    pixel_scores = np.array(
        [[[0.10, 0.20], [0.15, 0.05]], [[0.10, 0.90], [0.20, 0.10]]], dtype=np.float32
    )
    gt_masks = np.zeros_like(pixel_scores, dtype=bool)
    gt_masks[1, 0, 1] = True
    return pixel_scores, gt_masks


def test_current_preset_binary_counts_pixels_at_the_shipped_threshold():
    probabilities, gt_masks = _frames()
    current = _current_preset_binary(probabilities, gt_masks, {"threshold": 0.5})
    assert current["threshold"] == 0.5
    assert current["pixel"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0}


def test_current_preset_binary_defaults_to_one_half():
    probabilities, gt_masks = _frames()
    assert _current_preset_binary(probabilities, gt_masks, {})["threshold"] == 0.5


def test_current_preset_quantile_uses_the_per_frame_cutoff():
    pixel_scores, gt_masks = _frames()
    frame_quantiles = {0.5: np.median(pixel_scores.reshape(2, -1), axis=1)}
    current = _current_preset_quantile(pixel_scores, gt_masks, 0.5, frame_quantiles)
    assert current["quantile"] == 0.5
    # Frame 0 flags 0.20 and 0.15 (two false positives), frame 1 flags 0.90 and 0.20.
    assert current["pixel"] == {"precision": 0.25, "recall": 1.0, "f1": 0.4, "iou": 0.25}
