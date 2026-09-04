"""Calibration of the binary deciders against a labelled split.

Each ``calibrate(scores, targets)`` re-fits the decider's own decision rule to F1-max on
labelled validation scores and writes the result to both the live attribute (so ``forward``
uses it) and ``hparams`` (so it serialises into the pipeline yaml). The golden tests build a
cleanly separable synthetic split, calibrate, and assert the recalibrated decider reproduces
the ground truth in ``forward`` - the end-to-end contract - plus the live/hparam/report
consistency that makes the value persist.
"""

import pytest
import torch

from cuvis_ai.node.deciders.binary_decider import BinaryDecider, QuantileBinaryDecider
from cuvis_ai.node.deciders.two_stage_decider import TwoStageBinaryDecider

pytestmark = pytest.mark.unit

HIGH = 5.0  # raw anomaly score; sigmoid(5) is well clear of sigmoid(0)


def _frame_level_split(n_normal: int = 3, n_anom: int = 3, size: int = 8):
    """Frames with a bright 2x2 anomaly block (anomalous) or all-zero (normal).

    Returns ``(scores [N,H,W,1] float32, targets [N,H,W,1] bool)`` with normal frames first.
    """
    total = n_normal + n_anom
    scores = torch.zeros(total, size, size, 1)
    targets = torch.zeros(total, size, size, 1, dtype=torch.bool)
    for i in range(n_normal, total):
        scores[i, 1:3, 1:3, 0] = HIGH
        targets[i, 1:3, 1:3, 0] = True
    return scores, targets


def test_binary_decider_calibrate_reproduces_gt():
    scores, targets = _frame_level_split()
    decider = BinaryDecider(threshold=0.5)
    report = decider.calibrate(scores, targets)

    assert report["f1"] == pytest.approx(1.0)
    # forward thresholds sigmoid(logits); after calibration it must reproduce the GT exactly.
    decisions = decider.forward(logits=scores)["decisions"]
    assert torch.equal(decisions, targets)
    # the shipped default 0.5 would flag every pixel (sigmoid(0)=0.5>=0.5) - calibration moved it
    assert decider.threshold > 0.5


def test_quantile_decider_calibrate_finds_fraction():
    # every frame: 10 anomalous pixels (of 100) at HIGH, rest zero -> optimal cutoff flags 10%.
    n, size = 4, 10
    scores = torch.zeros(n, size, size, 1)
    targets = torch.zeros(n, size, size, 1, dtype=torch.bool)
    scores[:, 0, :, 0] = HIGH  # top row = 10 pixels
    targets[:, 0, :, 0] = True
    decider = QuantileBinaryDecider(quantile=0.5)
    report = decider.calibrate(scores, targets)

    assert report["f1"] == pytest.approx(1.0)
    decisions = decider.forward(logits=scores)["decisions"]
    assert torch.equal(decisions, targets)


def test_two_stage_decider_calibrate_gate_and_pixel():
    scores, targets = _frame_level_split()
    decider = TwoStageBinaryDecider(image_threshold=0.5, pixel_threshold=None)
    report = decider.calibrate(scores, targets)

    assert report["image_f1"] == pytest.approx(1.0)
    assert report["pixel_f1"] == pytest.approx(1.0)
    # stage 2 moved off the quantile fallback onto an absolute cutoff.
    assert decider.pixel_threshold is not None
    decisions = decider.forward(logits=scores)["decisions"]
    assert torch.equal(decisions, targets)


@pytest.mark.parametrize(
    ("decider", "keys"),
    [
        (BinaryDecider(threshold=0.5), ["threshold"]),
        (QuantileBinaryDecider(quantile=0.5), ["quantile"]),
        (TwoStageBinaryDecider(image_threshold=0.5), ["image_threshold", "pixel_threshold"]),
    ],
)
def test_calibrate_updates_live_attr_and_hparams(decider, keys):
    """The live attribute, ``hparams`` (yaml source) and the report all agree post-calibrate."""
    scores, targets = _frame_level_split()
    report = decider.calibrate(scores, targets)
    for key in keys:
        live = getattr(decider, key)
        assert decider.hparams[key] == live  # yaml will carry the calibrated value
        assert report[key]["new"] == live  # report matches what was written


def test_calibrate_survives_hparam_reconstruction():
    """Rebuilding a decider from its post-calibration hparams keeps the calibrated values."""
    scores, targets = _frame_level_split()
    decider = TwoStageBinaryDecider(image_threshold=0.5, pixel_threshold=None)
    decider.calibrate(scores, targets)
    rebuilt = TwoStageBinaryDecider(
        image_threshold=decider.hparams["image_threshold"],
        top_k_fraction=decider.hparams["top_k_fraction"],
        quantile=decider.hparams["quantile"],
        pixel_threshold=decider.hparams["pixel_threshold"],
    )
    assert rebuilt.image_threshold == decider.image_threshold
    assert rebuilt.pixel_threshold == decider.pixel_threshold
    assert torch.equal(rebuilt.forward(logits=scores)["decisions"], targets)
