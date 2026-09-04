"""Calibration of the binary deciders against a labelled split.

Each ``calibrate(scores, targets)`` re-fits the decider's own decision rule to F1-max on
labelled validation scores and writes the result to both the live attribute (so ``forward``
uses it) and ``hparams`` (so it serialises into the pipeline yaml). The golden tests build a
cleanly separable synthetic split, calibrate, and assert the recalibrated decider reproduces
the ground truth in ``forward`` - the end-to-end contract - plus the live/hparam/report
consistency that makes the value persist. The numeric tests check that what the sweep scored
is what ``forward`` decides (float32 pixel space, float64 image gate, sigmoid saturation),
and that the ``calibrate-thresholds`` CLI ships the same values as the methods.
"""

import numpy as np
import pytest
import torch

from cuvis_ai.node.deciders import _calibration as calib
from cuvis_ai.node.deciders.binary_decider import BinaryDecider, QuantileBinaryDecider
from cuvis_ai.node.deciders.two_stage_decider import TwoStageBinaryDecider
from cuvis_ai_core.pipeline.factory import PipelineBuilder

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


def _random_split(seed: int = 0, n: int = 4, size: int = 16):
    """Random float32 logits with ties and a sparse random mask, as tensors."""
    rng = np.random.default_rng(seed)
    scores = torch.from_numpy(
        (rng.integers(-20, 20, size=(n, size, size, 1)) / 10).astype(np.float32)
    )
    targets = torch.from_numpy(rng.random((n, size, size, 1)) < 0.05)
    targets[0, 0, 0, 0] = True
    targets[1, 0, 0, 0] = False
    return scores, targets


def _f1(decisions: torch.Tensor, targets: torch.Tensor) -> float:
    tp = float((decisions & targets).sum())
    fp = float((decisions & ~targets).sum())
    fn = float((~decisions & targets).sum())
    return calib.prf(tp, fp, fn)["f1"]


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
    # The sweep ran in float32 sigmoid space; the value is the plateau midpoint between the two
    # probability levels and is exact in float32.
    high = float(torch.sigmoid(torch.tensor(HIGH)))
    assert decider.threshold == pytest.approx((0.5 + high) / 2, abs=1e-6)
    assert np.float32(decider.threshold) == decider.threshold
    assert report["on_point_threshold"] == pytest.approx(high)


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


def test_quantile_decider_reduce_dims_equivalent_to_default_still_calibrates():
    # reduce_dims=[1, 2] on single-channel input reduces the same values as the default.
    scores, targets = _frame_level_split()
    decider = QuantileBinaryDecider(quantile=0.5, reduce_dims=[1, 2])
    report = decider.calibrate(scores, targets)
    assert report is not None
    assert decider.hparams["quantile"] == decider.quantile


def test_quantile_decider_refuses_reduce_dims_it_cannot_reproduce():
    scores, targets = _frame_level_split()
    scores = scores.repeat(1, 1, 1, 3)  # three channels: reduce_dims=[1, 2] keeps the C axis
    decider = QuantileBinaryDecider(quantile=0.5, reduce_dims=[1, 2])
    with pytest.raises(calib.CalibrationError, match="reduce_dims"):
        decider.calibrate(scores, targets)
    assert decider.quantile == 0.5 and decider.hparams["quantile"] == 0.5


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
    # Both values sit at the midpoint of the plateau between the normal (0) and anomalous (5)
    # levels, not on the anomalous samples themselves.
    assert decider.image_threshold == 2.5
    assert decider.pixel_threshold == 2.5
    assert report["on_point"] == {"image_threshold": HIGH, "pixel_threshold": HIGH}


def test_two_stage_decider_default_gate_off_gets_calibrated():
    """The CuvisNEXT training presets ship both thresholds as null; calibration fills them."""
    scores, targets = _frame_level_split()
    decider = TwoStageBinaryDecider()
    report = decider.calibrate(scores, targets)

    assert report["image_threshold"]["old"] is None
    assert report["pixel_threshold"]["old"] is None
    assert decider.image_threshold == 2.5 and decider.pixel_threshold == 2.5
    assert report["joint"]["f1"] >= report["pixel_f1"]
    assert {"image_threshold", "pixel_threshold"} <= report["joint"].keys()
    assert torch.equal(decider.forward(logits=scores)["decisions"], targets)


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


def test_calibrated_values_survive_the_pipeline_yaml_round_trip(tmp_path):
    """save_to_file writes the calibrated hparams; a pipeline built from that yaml has them."""
    scores, targets = _frame_level_split()
    config = {
        "metadata": {"name": "calibrated"},
        "nodes": [
            {
                "name": "decider",
                "class_name": ("cuvis_ai.node.deciders.two_stage_decider.TwoStageBinaryDecider"),
                "hparams": {"image_threshold": 0.5, "pixel_threshold": None},
            }
        ],
        "connections": [],
    }
    pipeline = PipelineBuilder().build_from_config(config)
    decider = next(node for node in pipeline.nodes if node.name == "decider")
    decider.calibrate(scores, targets)

    yaml_path = tmp_path / "calibrated.yaml"
    pipeline.save_to_file(yaml_path, save_weights=False)
    text = yaml_path.read_text(encoding="utf-8")
    assert "image_threshold: 2.5" in text and "pixel_threshold: 2.5" in text

    rebuilt = PipelineBuilder().build_from_config(yaml_path)
    node = next(n for n in rebuilt.nodes if n.name == "decider")
    assert node.image_threshold == 2.5 and node.pixel_threshold == 2.5
    assert torch.equal(node.forward(logits=scores)["decisions"], targets)


def test_forward_reproduces_the_sweep_on_random_scores():
    """Float32 pixel space and float64 gate: forward's F1 equals the F1 the sweep reported."""
    scores, targets = _random_split(seed=3)
    two_stage = TwoStageBinaryDecider(image_threshold=0.5, top_k_fraction=0.02)
    report = two_stage.calibrate(scores, targets)
    assert np.float32(two_stage.pixel_threshold) == two_stage.pixel_threshold
    decisions = two_stage.forward(logits=scores)["decisions"]
    assert _f1(decisions, targets) == pytest.approx(report["pixel_f1"], abs=1e-12)

    binary = BinaryDecider(threshold=0.5)
    report = binary.calibrate(scores, targets)
    assert np.float32(binary.threshold) == binary.threshold
    decisions = binary.forward(logits=scores)["decisions"]
    assert _f1(decisions, targets) == pytest.approx(report["f1"], abs=1e-12)


def test_binary_decider_saturated_logits_report_what_forward_can_do():
    """Above logits of ~17 float32 sigmoid is exactly 1.0: the sweep must not claim to separate."""
    scores = torch.full((2, 4, 4, 1), 20.0)
    targets = torch.zeros(2, 4, 4, 1, dtype=torch.bool)
    scores[1, :2, :2, 0] = 40.0
    targets[1, :2, :2, 0] = True
    decider = BinaryDecider(threshold=0.5)
    report = decider.calibrate(scores, targets)
    decisions = decider.forward(logits=scores)["decisions"]
    assert report["f1"] < 1.0
    assert _f1(decisions, targets) == pytest.approx(report["f1"], abs=1e-12)


def test_cli_ships_the_same_values_as_the_decider_methods():
    """The calibrate-thresholds CLI and calibrate() are one implementation."""
    from scripts.calibrate_thresholds import _binary_hparams, _two_stage_hparams

    scores, targets = _random_split(seed=4)
    _, pixel, gt, frame_labels = calib.reduce_scores_targets(scores, targets)

    two_stage = TwoStageBinaryDecider(image_threshold=0.5, top_k_fraction=0.02)
    two_stage.calibrate(scores, targets)
    image_scores = calib.topk_mean_scores(pixel, 0.02)
    best_image, _joint, conditional = calib.sweep_two_stage(
        pixel, gt, image_scores, frame_labels, 256
    )
    cli = _two_stage_hparams(best_image, conditional, 0.02)
    assert cli["image_threshold"] == two_stage.hparams["image_threshold"]
    assert cli["pixel_threshold"] == two_stage.hparams["pixel_threshold"]

    binary = BinaryDecider(threshold=0.5)
    binary.calibrate(scores, targets)
    best = calib.sweep_absolute(calib.sigmoid_float32(pixel), gt, 256)
    assert _binary_hparams(best)["threshold"] == binary.hparams["threshold"]
