"""Calibration to a precision or recall target.

The deciders calibrate to the F1 optimum unless their ``calibration_objective`` hparam names a
target: ``precision`` or ``recall`` at ``calibration_level`` ``pixels`` or ``frames``. These tests
pin the selection rule on hand-built rows, the hparam validation, the parity of the
single-threshold deciders with torchmetrics' fixed-point metrics (which search every distinct
score, so they also prove the refinement finds the exact point on a split larger than the
candidate grid), the frame-level definition against ``forward``, the unreachable case, and the
report numbers a client reads.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchmetrics.classification import BinaryPrecisionAtFixedRecall, BinaryRecallAtFixedPrecision

from cuvis_ai.node.deciders import _calibration as calib
from cuvis_ai.node.deciders.binary_decider import BinaryDecider, QuantileBinaryDecider
from cuvis_ai.node.deciders.two_stage_decider import TwoStageBinaryDecider

pytestmark = pytest.mark.unit

DECIDERS = [BinaryDecider, QuantileBinaryDecider, TwoStageBinaryDecider]


def _row(**values):
    """A sweep row with every metric the selection reads, zero unless given."""
    row = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "frame_precision": 0.0,
        "frame_recall": 0.0,
        "frame_f1": 0.0,
        "image_threshold": 0.0,
        "pixel_threshold": 0.0,
    }
    row.update(values)
    return row


def _logits(seed: int = 0, n: int = 4, size: int = 16, positive_rate: float = 0.1):
    """Distinct float32 logits where anomalies score higher on average, as tensors."""
    rng = np.random.default_rng(seed)
    scores = torch.from_numpy(rng.normal(size=(n, size, size, 1)).astype(np.float32))
    targets = torch.from_numpy(rng.random((n, size, size, 1)) < positive_rate)
    scores[targets] += 1.5
    targets[0, 0, 0, 0] = True
    targets[1, 0, 0, 0] = False
    return scores, targets


def _frame_prf(decisions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Frame-level precision and recall as the evaluation counts them: any pixel flags a frame."""
    flagged = decisions.flatten(1).any(dim=1)
    labels = targets.flatten(1).any(dim=1)
    tp = float((flagged & labels).sum())
    fp = float((flagged & ~labels).sum())
    fn = float((~flagged & labels).sum())
    return calib.prf(tp, fp, fn)


def _pixel_prf(decisions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    tp = float((decisions & targets).sum())
    fp = float((decisions & ~targets).sum())
    fn = float((~decisions & targets).sum())
    return calib.prf(tp, fp, fn)


# --- the selection rule on hand-built rows -------------------------------------------------


def test_recall_target_takes_the_most_precise_row_at_or_above_the_target():
    rows = [
        _row(recall=0.95, precision=0.5, pixel_threshold=1.0),
        _row(recall=0.92, precision=0.7, pixel_threshold=2.0),
        _row(recall=0.85, precision=0.9, pixel_threshold=3.0),
    ]
    row, reached = calib.select_calibration_row(
        rows, calib.CalibrationObjective("recall", 0.9, "pixels")
    )
    assert reached
    assert row["pixel_threshold"] == 2.0


def test_precision_target_takes_the_highest_recall_row_at_or_above_the_target():
    rows = [
        _row(recall=0.95, precision=0.5, pixel_threshold=1.0),
        _row(recall=0.92, precision=0.7, pixel_threshold=2.0),
        _row(recall=0.85, precision=0.9, pixel_threshold=3.0),
    ]
    row, reached = calib.select_calibration_row(
        rows, calib.CalibrationObjective("precision", 0.6, "pixels")
    )
    assert reached
    assert row["pixel_threshold"] == 2.0


def test_ties_resolve_by_the_other_level_then_the_higher_thresholds():
    rows = [
        _row(recall=0.9, precision=0.8, frame_f1=0.5, image_threshold=1.0, pixel_threshold=2.0),
        _row(recall=0.9, precision=0.8, frame_f1=0.9, image_threshold=1.0, pixel_threshold=1.0),
        _row(recall=0.9, precision=0.8, frame_f1=0.9, image_threshold=2.0, pixel_threshold=1.0),
    ]
    row, _ = calib.select_calibration_row(rows, calib.CalibrationObjective("recall", 0.9, "pixels"))
    assert row["frame_f1"] == 0.9
    assert (row["image_threshold"], row["pixel_threshold"]) == (2.0, 1.0)


def test_unreachable_recall_lands_on_the_highest_recall_row_and_says_so():
    rows = [
        _row(recall=0.95, precision=0.5, pixel_threshold=1.0),
        _row(recall=0.95, precision=0.6, pixel_threshold=2.0),
        _row(recall=0.85, precision=0.9, pixel_threshold=3.0),
    ]
    row, reached = calib.select_calibration_row(
        rows, calib.CalibrationObjective("recall", 0.99, "pixels")
    )
    assert not reached
    assert row["pixel_threshold"] == 2.0  # the most precise of the highest-recall rows


def test_unreachable_precision_lands_on_the_most_precise_row():
    rows = [
        _row(recall=0.95, precision=0.5, pixel_threshold=1.0),
        _row(recall=0.85, precision=0.9, pixel_threshold=3.0),
        _row(recall=0.80, precision=0.9, pixel_threshold=4.0),
    ]
    row, reached = calib.select_calibration_row(
        rows, calib.CalibrationObjective("precision", 0.99, "pixels")
    )
    assert not reached
    assert row["pixel_threshold"] == 3.0  # the highest recall among the most precise rows


def test_frame_level_reads_the_frame_columns():
    rows = [
        _row(frame_recall=0.9, frame_precision=0.5, recall=0.1, precision=0.1, pixel_threshold=1.0),
        _row(frame_recall=0.9, frame_precision=0.8, recall=0.1, precision=0.1, pixel_threshold=2.0),
    ]
    row, reached = calib.select_calibration_row(
        rows, calib.CalibrationObjective("recall", 0.9, "frames")
    )
    assert reached
    assert row["pixel_threshold"] == 2.0


# --- the hparams ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", DECIDERS)
def test_a_target_objective_needs_a_target(cls):
    with pytest.raises(ValueError, match="calibration_target"):
        cls(calibration_objective="recall")


@pytest.mark.parametrize("cls", DECIDERS)
@pytest.mark.parametrize("target", [-0.1, 1.5, float("nan")])
def test_a_target_outside_the_unit_interval_is_refused(cls, target):
    with pytest.raises(ValueError, match="calibration_target"):
        cls(calibration_objective="precision", calibration_target=target)


@pytest.mark.parametrize("cls", DECIDERS)
def test_an_unknown_objective_is_refused(cls):
    with pytest.raises(ValueError, match="calibration_objective"):
        cls(calibration_objective="accuracy", calibration_target=0.9)


@pytest.mark.parametrize("cls", DECIDERS)
def test_an_unknown_level_is_refused(cls):
    with pytest.raises(ValueError, match="calibration_level"):
        cls(calibration_objective="recall", calibration_target=0.9, calibration_level="images")


def test_the_quantile_decider_refuses_the_frame_level():
    with pytest.raises(ValueError, match="calibration_level"):
        QuantileBinaryDecider(
            calibration_objective="recall", calibration_target=0.9, calibration_level="frames"
        )


@pytest.mark.parametrize("cls", DECIDERS)
def test_the_objective_defaults_to_f1_and_persists_in_the_hparams(cls):
    default = cls()
    assert default.hparams["calibration_objective"] == "f1"
    assert default.hparams["calibration_target"] is None
    assert default.hparams["calibration_level"] == "pixels"
    decider = cls(calibration_objective="recall", calibration_target=0.9)
    rebuilt = cls(**decider.hparams)
    assert rebuilt.hparams["calibration_objective"] == "recall"
    assert rebuilt.hparams["calibration_target"] == 0.9


# --- the deciders against torchmetrics and forward ------------------------------------------


def test_binary_decider_recall_target_matches_torchmetrics_and_forward():
    scores, targets = _logits()
    decider = BinaryDecider(calibration_objective="recall", calibration_target=0.9)
    report = decider.calibrate(scores, targets)

    probabilities = torch.sigmoid(scores).reshape(-1)
    labels = targets.reshape(-1).int()
    precision, _ = BinaryPrecisionAtFixedRecall(min_recall=0.9)(probabilities, labels)
    assert report["objective"] == "recall" and report["level"] == "pixels"
    assert report["target"] == 0.9 and report["reached"] == 1.0
    assert report["pixel_recall"] >= 0.9
    assert report["pixel_precision"] == pytest.approx(float(precision), abs=1e-6)

    decisions = decider.forward(logits=scores)["decisions"]
    pixel = _pixel_prf(decisions, targets)
    assert pixel["recall"] == pytest.approx(report["pixel_recall"])
    assert pixel["precision"] == pytest.approx(report["pixel_precision"])
    assert decider.hparams["threshold"] == decider.threshold


def test_binary_decider_precision_target_on_a_split_larger_than_the_grid_is_exact():
    """18432 distinct scores: the coarse grid alone misses the point, the refinement finds it."""
    scores, targets = _logits(seed=3, n=2, size=96)
    decider = BinaryDecider(calibration_objective="precision", calibration_target=0.8)
    report = decider.calibrate(scores, targets)

    probabilities = torch.sigmoid(scores).reshape(-1)
    labels = targets.reshape(-1).int()
    recall, _ = BinaryRecallAtFixedPrecision(min_precision=0.8)(probabilities, labels)
    assert report["reached"] == 1.0
    assert report["pixel_precision"] >= 0.8
    assert report["pixel_recall"] == pytest.approx(float(recall), abs=1e-6)


def test_two_stage_frame_recall_target_counts_frames_like_the_evaluation():
    scores, targets = _logits(seed=1, n=8, positive_rate=0.02)
    targets[0] = False
    targets[1] = False
    decider = TwoStageBinaryDecider(
        calibration_objective="recall", calibration_target=0.8, calibration_level="frames"
    )
    report = decider.calibrate(scores, targets)

    decisions = decider.forward(logits=scores)["decisions"]
    frame = _frame_prf(decisions, targets)
    pixel = _pixel_prf(decisions, targets)
    assert report["reached"] == 1.0 and report["frame_recall"] >= 0.8
    assert report["frame_recall"] == pytest.approx(frame["recall"])
    assert report["frame_precision"] == pytest.approx(frame["precision"])
    assert report["pixel_recall"] == pytest.approx(pixel["recall"])
    assert report["pixel_precision"] == pytest.approx(pixel["precision"])
    assert report["frames"] == 8
    assert report["anomalous_frames"] == int(targets.flatten(1).any(dim=1).sum())
    assert decider.hparams["image_threshold"] == decider.image_threshold
    assert decider.hparams["pixel_threshold"] == decider.pixel_threshold


def test_two_stage_frame_target_grid_is_exact_in_the_frame_flags():
    """A frame target is reached whenever any cutoff reaches it: the grid holds every frame maximum."""
    scores, targets = _logits(seed=2, n=6, positive_rate=0.02)
    targets[0] = False
    decider = TwoStageBinaryDecider(
        calibration_objective="precision", calibration_target=1.0, calibration_level="frames"
    )
    report = decider.calibrate(scores, targets)
    # The anomalous frames hold the highest pixel scores (shifted by 1.5), so a gate that admits
    # only them and a cutoff at their maximum gives frame precision 1.0.
    assert report["reached"] == 1.0
    assert report["frame_precision"] == pytest.approx(1.0)
    decisions = decider.forward(logits=scores)["decisions"]
    assert _frame_prf(decisions, targets)["precision"] == pytest.approx(1.0)


def test_quantile_decider_pixel_recall_target():
    scores, targets = _logits(seed=4, n=4, size=12)
    decider = QuantileBinaryDecider(calibration_objective="recall", calibration_target=0.7)
    report = decider.calibrate(scores, targets)
    decisions = decider.forward(logits=scores)["decisions"]
    pixel = _pixel_prf(decisions, targets)
    assert report["reached"] == 1.0 and report["pixel_recall"] >= 0.7
    assert pixel["recall"] == pytest.approx(report["pixel_recall"])
    assert pixel["precision"] == pytest.approx(report["pixel_precision"])
    assert decider.hparams["quantile"] == decider.quantile


def test_an_unreachable_precision_target_lands_on_the_most_precise_point_and_says_so():
    rng = np.random.default_rng(5)
    scores = torch.from_numpy(rng.normal(size=(4, 16, 16, 1)).astype(np.float32))
    targets = torch.from_numpy(rng.random((4, 16, 16, 1)) < 0.1)  # no separation at all
    # The five highest scores sit on normal pixels, so every cutoff that flags an anomalous
    # pixel also flags them and no cutoff reaches a precision of 0.99.
    scores[0, 0, :5, 0] = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0])
    targets[0, 0, :5, 0] = False
    decider = BinaryDecider(calibration_objective="precision", calibration_target=0.99)
    report = decider.calibrate(scores, targets)

    probabilities = torch.sigmoid(scores).reshape(-1).numpy()
    labels = targets.reshape(-1).numpy()
    best = 0.0
    for cutoff in np.unique(probabilities):
        flagged = probabilities >= cutoff
        best = max(best, flagged[labels].sum() / flagged.sum())
    assert report["reached"] == 0.0 and report["target"] == 0.99
    assert report["pixel_precision"] == pytest.approx(best)
    assert report["pixel_precision"] < 0.99


def test_the_f1_objective_reports_both_levels_too():
    scores, targets = _logits(seed=6, n=6, positive_rate=0.03)
    targets[0] = False
    decider = TwoStageBinaryDecider()
    report = decider.calibrate(scores, targets)
    decisions = decider.forward(logits=scores)["decisions"]
    frame = _frame_prf(decisions, targets)
    assert report["objective"] == "f1"
    assert "reached" not in report and "target" not in report
    assert report["frame_recall"] == pytest.approx(frame["recall"])
    assert report["frame_precision"] == pytest.approx(frame["precision"])
    assert report["frames"] == 6
    assert "summary" not in report
    # The counts behind the rates are the ones forward's decisions give, as integers.
    flagged = decisions.flatten(1).any(dim=1)
    labels = targets.flatten(1).any(dim=1)
    assert report["frame_tp"] == int((flagged & labels).sum())
    assert report["frame_fp"] == int((flagged & ~labels).sum())
    assert report["frame_fn"] == int((~flagged & labels).sum())
    assert report["pixel_tp"] == int((decisions & targets).sum())
    assert report["pixel_fp"] == int((decisions & ~targets).sum())
    assert report["pixel_fn"] == int((~decisions & targets).sum())
    assert report["frame_tp"] + report["frame_fn"] == report["anomalous_frames"]
    assert all(isinstance(report[key], int) for key in ("pixel_tp", "frame_tp", "frame_fn"))
