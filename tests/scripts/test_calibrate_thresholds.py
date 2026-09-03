"""Pure-function tests for the calibrate-thresholds report helpers (no data, no model)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.calibrate_thresholds import _current_preset_two_stage, _print_report

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
