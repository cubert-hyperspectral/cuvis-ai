"""Pure-function tests for the calibrate-thresholds report helpers (no data, no model)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cuvis_ai_schemas.training import CalibrationConfig

from cuvis_ai.node.deciders.binary_decider import BinaryDecider, QuantileBinaryDecider
from cuvis_ai.node.deciders.two_stage_decider import TwoStageBinaryDecider
from scripts.calibrate_thresholds import (
    _calibration_block,
    _current_preset_two_stage,
    _print_report,
    _target_point,
)

pytestmark = pytest.mark.unit


def _fixture() -> dict[str, np.ndarray]:
    # Two 2x2 frames: frame 0 is a clean frame with low scores, frame 1 carries one hot
    # anomalous pixel. Image scores are the per-frame maxima.
    pixel_scores = np.array(
        [[[0.10, 0.20], [0.15, 0.05]], [[0.10, 0.90], [0.20, 0.10]]], dtype=np.float32
    )
    gt_masks = np.zeros_like(pixel_scores, dtype=bool)
    gt_masks[1, 0, 1] = True
    frame_labels = gt_masks.any(axis=(1, 2))
    image_scores = pixel_scores.max(axis=(1, 2))
    frame_quantiles = {0.5: np.median(pixel_scores.reshape(2, -1), axis=1)}
    return {
        "pixel_scores": pixel_scores,
        "gt_masks": gt_masks,
        "image_scores": image_scores,
        "frame_labels": frame_labels,
        "frame_quantiles": frame_quantiles,
    }


def test_current_preset_two_stage_gate_off_scores_every_frame():
    f = _fixture()
    current = _current_preset_two_stage(
        f["pixel_scores"],
        f["gt_masks"],
        f["image_scores"],
        f["frame_labels"],
        {"image_threshold": None, "quantile": 0.5, "pixel_threshold": None},
        f["frame_quantiles"],
    )
    assert current["image_threshold"] is None
    assert current["image"] is None
    assert current["stage2"] == {"mode": "quantile", "quantile": 0.5}
    # Gate off: both frames flag their pixels at or above the per-frame median.
    # Frame 0 median 0.125 -> 0.15, 0.20 flagged (2 fp); frame 1 median 0.15 -> 0.90, 0.20
    # flagged (1 tp, 1 fp).
    assert current["pixel"]["recall"] == 1.0
    assert current["pixel"]["precision"] == pytest.approx(1 / 4)


def test_current_preset_two_stage_finite_gate_blanks_clean_frame():
    f = _fixture()
    current = _current_preset_two_stage(
        f["pixel_scores"],
        f["gt_masks"],
        f["image_scores"],
        f["frame_labels"],
        {"image_threshold": 0.5, "quantile": 0.5, "pixel_threshold": None},
        f["frame_quantiles"],
    )
    assert current["image_threshold"] == 0.5
    assert current["image"]["f1"] == 1.0  # the gate separates the two frames perfectly
    # Only frame 1 passes the gate: 1 tp, 1 fp.
    assert current["pixel"]["precision"] == pytest.approx(1 / 2)
    assert current["pixel"]["recall"] == 1.0


def _two_stage_report(current: dict) -> dict:
    return {
        "split": "val",
        "frames": 2,
        "anomalous_frames": 1,
        "decider": {"node": "decider", "class_name": "TwoStageBinaryDecider", "mode": "two_stage"},
        "score_source": "dinomaly.scores",
        "image": {
            "auroc": 1.0,
            "f1_max": {"f1": 1.0, "threshold": 0.5, "precision": 1.0, "recall": 1.0},
        },
        "pixel": {
            "joint_optimum": {"f1": 1.0, "image_threshold": 0.5, "pixel_threshold": 0.6},
            "conditional_on_image_f1max": {
                "f1": 1.0,
                "pixel_threshold": 0.6,
                "precision": 1.0,
                "recall": 1.0,
                "iou": 1.0,
            },
        },
        "current_preset": current,
        "calibrated_decider_hparams": {
            "image_threshold": 0.5,
            "top_k_fraction": 0.001,
            "pixel_threshold": 0.6,
        },
    }


def test_print_report_says_gate_off_for_null_image_threshold(capsys):
    current = {
        "image_threshold": None,
        "stage2": {"mode": "quantile", "quantile": 0.995},
        "image": None,
        "pixel": {"precision": 0.25, "recall": 1.0, "f1": 0.4, "iou": 0.25},
    }
    _print_report(_two_stage_report(current))
    out = capsys.readouterr().out
    assert "current preset (gate off, quantile=0.995): pixel F1=0.4000" in out
    assert "image_threshold: 0.5" in out  # the calibrated value to paste stays printed


def test_print_report_keeps_image_metrics_for_finite_gate(capsys):
    current = {
        "image_threshold": 0.5,
        "stage2": {"mode": "absolute", "pixel_threshold": 0.6},
        "image": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0},
        "pixel": {"precision": 0.5, "recall": 1.0, "f1": 0.6667, "iou": 0.5},
    }
    _print_report(_two_stage_report(current))
    out = capsys.readouterr().out
    assert "current preset (image_threshold=0.5, pixel_threshold=0.6): image F1=1.0000" in out


# --- a precision or recall target ----------------------------------------------------------


def _scored_frames(seed: int = 1, n: int = 8, size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Raw ``[N, H, W]`` scores where anomalous pixels score higher; frames 0 and 1 are clean."""
    rng = np.random.default_rng(seed)
    pixel_scores = rng.normal(size=(n, size, size)).astype(np.float32)
    gt_masks = rng.random((n, size, size)) < 0.02
    gt_masks[:2] = False
    gt_masks[2:, 0, 0] = True
    pixel_scores[gt_masks] += 1.5
    return pixel_scores, gt_masks


def test_flags_without_a_target_keep_the_f1_sweep():
    assert _calibration_block("f1", None, "pixels") is None


def test_flags_with_a_target_become_the_trainrun_block():
    assert _calibration_block("recall", 0.9, "frames") == CalibrationConfig(
        objective="recall", target=0.9, level="frames"
    )


@pytest.mark.parametrize(
    ("objective", "target", "level"),
    [
        ("recall", None, "pixels"),
        ("f1", 0.9, "pixels"),
        ("f1", None, "frames"),
        ("precision", 1.5, "pixels"),
    ],
)
def test_flags_the_trainrun_block_refuses_are_refused(objective, target, level):
    with pytest.raises(ValueError):
        _calibration_block(objective, target, level)


def test_the_target_point_is_the_deciders_own_calibration():
    pixel_scores, gt_masks = _scored_frames()
    block = CalibrationConfig(objective="recall", target=0.8, level="frames")
    decider = TwoStageBinaryDecider()
    point, hparams = _target_point(
        decider, "two_stage", pixel_scores, gt_masks, block, num_candidates=64, top_k_fraction=0.002
    )
    expected = TwoStageBinaryDecider(top_k_fraction=0.002).calibrate(
        torch.from_numpy(pixel_scores[..., None]),
        torch.from_numpy(gt_masks[..., None]),
        objective=block,
        num_candidates=64,
    )
    assert point == expected
    assert point["reached"] == 1.0 and point["frame_recall"] >= 0.8
    assert hparams == {
        "image_threshold": decider.image_threshold,
        "top_k_fraction": 0.002,
        "pixel_threshold": decider.pixel_threshold,
    }


def test_the_target_point_of_a_binary_decider_ships_its_threshold():
    pixel_scores, gt_masks = _scored_frames(seed=2)
    decider = BinaryDecider()
    point, hparams = _target_point(
        decider,
        "binary",
        pixel_scores,
        gt_masks,
        CalibrationConfig(objective="precision", target=0.5),
        num_candidates=64,
        top_k_fraction=0.001,
    )
    assert (point["objective"], point["level"]) == ("precision", "pixels")
    assert hparams == {"threshold": decider.threshold}


def test_the_quantile_decider_refuses_a_frame_level_target():
    pixel_scores, gt_masks = _scored_frames(seed=3)
    with pytest.raises(ValueError, match="frames"):
        _target_point(
            QuantileBinaryDecider(),
            "quantile",
            pixel_scores,
            gt_masks,
            CalibrationConfig(objective="recall", target=0.9, level="frames"),
            num_candidates=64,
            top_k_fraction=0.001,
        )


def test_print_report_names_the_target_and_whether_it_was_reached(capsys):
    report = {
        "split": "val",
        "frames": 8,
        "anomalous_frames": 6,
        "decider": {"node": "decider", "class_name": "BinaryDecider", "mode": "binary"},
        "score_source": "rx.scores",
        "objective": {"objective": "recall", "target": 0.9, "level": "frames"},
        "target_point": {
            "reached": 0.0,
            "frame_precision": 0.75,
            "frame_recall": 0.5,
            "frame_tp": 3,
            "frame_fp": 1,
            "frame_fn": 3,
            "pixel_precision": 0.4,
            "pixel_recall": 0.25,
            "pixel_tp": 10,
            "pixel_fp": 15,
            "pixel_fn": 30,
        },
        "calibrated_decider_hparams": {"threshold": 0.7},
    }
    _print_report(report)
    out = capsys.readouterr().out
    assert "target: recall 0.9 at the frame level, not reached" in out
    assert "frames: precision 0.750, recall 0.500 (tp 3, fp 1, fn 3)" in out
    assert "pixels: precision 0.400, recall 0.250 (tp 10, fp 15, fn 30)" in out
    assert "threshold: 0.7" in out
