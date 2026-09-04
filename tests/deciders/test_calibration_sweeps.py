"""Pure-numpy tests for the shared threshold-calibration sweep helpers.

These pin the numerics the deciders and the ``calibrate-thresholds`` CLI both rely on: the
one-pass ``searchsorted`` confusion counts match the plain per-candidate loop bit for bit,
candidates and margins are exact in float32 (the dtype the deciders compare pixel scores
in), the stage-1 image score is the statistic ``TwoStageBinaryDecider.forward`` computes, and
unusable input is refused with a ``CalibrationError`` that names the problem.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.deciders import _calibration as calib

pytestmark = pytest.mark.unit


def _random_split(seed: int = 0, n: int = 4, size: int = 16, positive_rate: float = 0.05):
    """Random float32 scores with ties plus a sparse random mask, as numpy arrays."""
    rng = np.random.default_rng(seed)
    # Integer-valued scores divided by 10 give plenty of exact ties.
    pixel = (rng.integers(0, 40, size=(n, size, size)) / 10).astype(np.float32)
    gt = rng.random((n, size, size)) < positive_rate
    gt[0, 0, 0] = True  # guarantee at least one positive
    gt[1, 0, 0] = False  # and one negative
    return pixel, gt


def _reference_confusions(pixel_scores, gt_masks, candidates):
    """The pre-searchsorted implementation: one full pass per candidate."""
    n = pixel_scores.shape[0]
    flat_scores = pixel_scores.reshape(n, -1)
    flat_gt = gt_masks.reshape(n, -1)
    tp = np.empty((n, candidates.size), dtype=np.float64)
    fp = np.empty_like(tp)
    for j, thr in enumerate(candidates):
        flagged = flat_scores >= thr
        tp[:, j] = (flagged & flat_gt).sum(axis=1)
        fp[:, j] = (flagged & ~flat_gt).sum(axis=1)
    return tp, fp


def _reference_image_score(pixel_scores: torch.Tensor, top_k_fraction: float) -> float:
    """The stage-1 statistic as it was written inline in ``TwoStageBinaryDecider.forward``."""
    flat = pixel_scores.reshape(-1)
    k = max(
        1,
        int(torch.ceil(torch.tensor(flat.numel() * top_k_fraction, dtype=torch.float32)).item()),
    )
    return torch.topk(flat, k).values.mean().item()


# --- margin_below ---------------------------------------------------------------------


def test_margin_below_is_the_plateau_midpoint():
    samples = np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float32)
    assert calib.margin_below(5.0, samples, float32=True) == 2.5
    assert calib.margin_below(5.0, samples, float32=False) == 2.5


def test_margin_below_without_a_lower_sample_returns_the_threshold():
    samples = np.array([5.0, 5.0], dtype=np.float32)
    assert calib.margin_below(5.0, samples, float32=True) == 5.0


def test_margin_below_float32_falls_back_when_the_midpoint_collapses():
    upper = np.float32(1.0)
    lower = np.nextafter(upper, np.float32(0.0))  # adjacent float32 values
    samples = np.array([lower, upper], dtype=np.float32)
    margin = calib.margin_below(float(upper), samples, float32=True)
    assert margin == float(upper)  # no float32 value fits strictly between the two
    assert np.float32(margin) == margin


# --- candidates and confusions ----------------------------------------------------------


def test_pixel_candidates_are_float32_exact_sorted_and_include_the_max():
    pixel, _ = _random_split()
    candidates = calib.pixel_candidates(pixel, 64)
    assert candidates.dtype == np.float32
    assert np.all(np.diff(candidates) > 0)
    assert np.float32(pixel.max()) in candidates
    assert all(np.float32(float(c)) == c for c in candidates)


def test_frame_confusions_match_the_per_candidate_loop():
    for seed in range(3):
        pixel, gt = _random_split(seed=seed)
        candidates = calib.pixel_candidates(pixel, 64)
        tp, fp = calib.frame_confusions(pixel, gt, candidates)
        ref_tp, ref_fp = _reference_confusions(pixel, gt, candidates)
        np.testing.assert_array_equal(tp, ref_tp)
        np.testing.assert_array_equal(fp, ref_fp)


def test_sweep_absolute_margin_lies_strictly_between_adjacent_samples():
    pixel, gt = _random_split(seed=1)
    best = calib.sweep_absolute(pixel, gt, 64)
    threshold, margin = best["threshold"], best["margin_threshold"]
    assert np.float32(threshold) == threshold
    assert np.float32(margin) == margin
    below = pixel[pixel < threshold]
    if below.size:
        assert float(below.max()) < margin <= threshold
    else:
        assert margin == threshold
    # Same flagged set, so the same F1 as the on-point optimum.
    flagged = pixel >= np.float32(margin)
    tp = float((flagged & gt).sum())
    assert calib.prf(tp, float((flagged & ~gt).sum()), float(gt.sum()) - tp)["f1"] == best["f1"]


def test_sweep_two_stage_carries_on_point_and_margin_keys():
    pixel, gt = _random_split(seed=2)
    frame_labels = gt.any(axis=(1, 2))
    image_scores = calib.topk_mean_scores(pixel, 0.01)
    best_image, joint, conditional = calib.sweep_two_stage(
        pixel, gt, image_scores, frame_labels, 64
    )
    assert {"threshold", "margin_threshold", "f1"} <= best_image.keys()
    for row in (joint, conditional):
        assert {"pixel_threshold", "margin_pixel_threshold", "margin_image_threshold"} <= row.keys()
        assert np.float32(row["margin_pixel_threshold"]) == row["margin_pixel_threshold"]
    assert conditional["image_threshold"] == best_image["threshold"]
    assert joint["f1"] >= conditional["f1"]


def test_sweep_ties_resolve_to_the_highest_candidate():
    # Every threshold in (0, 5] separates perfectly: the highest candidate (5.0) must win.
    pixel = np.zeros((2, 4, 4), dtype=np.float32)
    gt = np.zeros((2, 4, 4), dtype=bool)
    pixel[1, :2, :2] = 5.0
    gt[1, :2, :2] = True
    best = calib.sweep_absolute(pixel, gt, 32)
    assert best["threshold"] == 5.0
    assert best["margin_threshold"] == 2.5


# --- stage-1 image score ----------------------------------------------------------------


@pytest.mark.parametrize("top_k_fraction", [0.001, 0.25, 0.5, 1.0])
def test_frame_image_score_matches_the_inline_forward_statistic(top_k_fraction):
    frames = torch.rand(5, 23, 31, generator=torch.Generator().manual_seed(0))
    for frame in frames:
        assert calib.frame_image_score(frame, top_k_fraction) == _reference_image_score(
            frame, top_k_fraction
        )
    stacked = calib.topk_mean_scores(frames.numpy(), top_k_fraction)
    assert stacked.dtype == np.float64
    assert stacked.tolist() == [_reference_image_score(f, top_k_fraction) for f in frames]


def test_topk_count_rounds_the_product_in_float32():
    # 1000 * 0.001 is 1.0000000000000002 in float64 but exactly 1 in float32: k stays 1.
    assert calib.topk_count(1000, 0.001) == 1
    assert calib.topk_count(1, 1e-9) == 1  # never below one pixel


# --- sigmoid space ------------------------------------------------------------------------


def test_sigmoid_float32_matches_torch_and_saturates():
    values = np.array([-40.0, 0.0, 5.0, 20.0, 40.0], dtype=np.float32)
    probs = calib.sigmoid_float32(values)
    assert probs.dtype == np.float32
    np.testing.assert_array_equal(probs, torch.sigmoid(torch.from_numpy(values)).numpy())
    assert probs[3] == 1.0 and probs[4] == 1.0  # float32 saturation above logits of ~17


# --- guards --------------------------------------------------------------------------------


def _scores_and_targets(n=2, size=4):
    scores = torch.zeros(n, size, size, 1)
    targets = torch.zeros(n, size, size, 1, dtype=torch.bool)
    scores[1, 0, 0, 0] = 5.0
    targets[1, 0, 0, 0] = True
    return scores, targets


def test_reduce_scores_targets_accepts_3d_and_4d_layouts():
    scores, targets = _scores_and_targets()
    full, pixel, gt, frame_labels = calib.reduce_scores_targets(scores, targets[..., 0])
    assert full.shape == (2, 4, 4, 1) and pixel.shape == (2, 4, 4)
    assert pixel.dtype == np.float32 and gt.dtype == bool
    assert frame_labels.tolist() == [False, True]


def test_reduce_scores_targets_refuses_shape_mismatch():
    scores, targets = _scores_and_targets()
    with pytest.raises(calib.CalibrationError, match=r"disagree on \[N, H, W\]"):
        calib.reduce_scores_targets(scores, targets[:1])
    with pytest.raises(calib.CalibrationError, match="scores must be"):
        calib.reduce_scores_targets(scores[0, 0], targets[0, 0])  # rank 2: not a frame stack


def test_reduce_scores_targets_refuses_non_finite_scores():
    scores, targets = _scores_and_targets()
    scores[0, 1, 1, 0] = float("nan")
    with pytest.raises(calib.CalibrationError, match="nan or inf"):
        calib.reduce_scores_targets(scores, targets)


def test_reduce_scores_targets_refuses_single_class_targets():
    scores, targets = _scores_and_targets()
    with pytest.raises(calib.CalibrationError, match="no anomalous pixel"):
        calib.reduce_scores_targets(scores, torch.zeros_like(targets))
    with pytest.raises(calib.CalibrationError, match="every pixel anomalous"):
        calib.reduce_scores_targets(scores, torch.ones_like(targets))


# --- auroc ---------------------------------------------------------------------------------


def test_binary_auroc_separable_ties_and_single_class():
    labels = np.array([False, False, True, True])
    assert calib.binary_auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert calib.binary_auroc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == 0.5  # midranks
    assert np.isnan(calib.binary_auroc(np.array([0.1, 0.9]), np.array([True, True])))
