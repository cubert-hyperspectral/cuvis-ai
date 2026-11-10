"""DecisionOutcomeOverlay paints hits, false alarms and misses of a binary decision onto RGB.

Non-square frames (H=6, W=10) catch an H/W transposition. The decision and the target
rectangles overlap, so one frame holds all three outcomes and untouched background.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from cuvis_ai.node.anomaly_visualization import DecisionOutcomeOverlay

pytestmark = pytest.mark.unit

B, H, W = 1, 6, 10
GREY = 0.5
HIT, FALSE_ALARM, MISS = (1.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)


def _frame() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grey RGB, decisions on columns 2..5, targets on columns 4..7, both on rows 1..3."""
    rgb = torch.full((B, H, W, 3), GREY, dtype=torch.float32)
    decisions = torch.zeros(B, H, W, 1, dtype=torch.bool)
    targets = torch.zeros(B, H, W, 1, dtype=torch.bool)
    decisions[0, 1:4, 2:6] = True
    targets[0, 1:4, 4:8] = True
    return rgb, decisions, targets


def _expected(alpha: float, with_targets: bool = True) -> torch.Tensor:
    """Hand-built expected frame: each outcome blended with its colour at ``alpha``."""
    out = torch.full((B, H, W, 3), GREY, dtype=torch.float32)

    def paint(rows: slice, cols: slice, colour: tuple[float, float, float]) -> None:
        out[0, rows, cols] = (1.0 - alpha) * GREY + alpha * torch.tensor(colour)

    if with_targets:
        paint(slice(1, 4), slice(2, 4), FALSE_ALARM)  # decided, not annotated
        paint(slice(1, 4), slice(4, 6), HIT)  # decided and annotated
        paint(slice(1, 4), slice(6, 8), MISS)  # annotated, not decided
    else:
        paint(slice(1, 4), slice(2, 6), FALSE_ALARM)  # no ground truth: every kept pixel
    return out


def test_solid_overlay_paints_each_outcome_in_its_colour() -> None:
    rgb, decisions, targets = _frame()
    out = DecisionOutcomeOverlay()(rgb_image=rgb, decisions=decisions, targets=targets)
    assert torch.allclose(out["rgb_image"], _expected(alpha=1.0), atol=1e-6)


def test_alpha_blends_colour_with_image() -> None:
    rgb, decisions, targets = _frame()
    out = DecisionOutcomeOverlay(alpha=0.5)(rgb_image=rgb, decisions=decisions, targets=targets)
    assert torch.allclose(out["rgb_image"], _expected(alpha=0.5), atol=1e-6)


def test_without_targets_every_decision_is_painted_as_false_alarm() -> None:
    rgb, decisions, _ = _frame()
    out = DecisionOutcomeOverlay()(rgb_image=rgb, decisions=decisions)
    assert torch.allclose(out["rgb_image"], _expected(alpha=1.0, with_targets=False), atol=1e-6)


def test_custom_colours() -> None:
    rgb, decisions, targets = _frame()
    node = DecisionOutcomeOverlay(
        hit_color=(0.0, 0.0, 1.0), false_alarm_color=(1.0, 0.0, 1.0), miss_color=(0.0, 1.0, 1.0)
    )
    out = node(rgb_image=rgb, decisions=decisions, targets=targets)["rgb_image"]
    assert torch.equal(out[0, 2, 4], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(out[0, 2, 2], torch.tensor([1.0, 0.0, 1.0]))
    assert torch.equal(out[0, 2, 7], torch.tensor([0.0, 1.0, 1.0]))
    assert torch.equal(out[0, 0, 0], torch.tensor([GREY, GREY, GREY]))


def test_port_contract_and_input_untouched() -> None:
    rgb, decisions, targets = _frame()
    before = rgb.clone()
    out = DecisionOutcomeOverlay()(rgb_image=rgb, decisions=decisions, targets=targets)
    spec = DecisionOutcomeOverlay.OUTPUT_SPECS["rgb_image"]
    assert set(out) == set(DecisionOutcomeOverlay.OUTPUT_SPECS)
    assert out["rgb_image"].shape == (B, H, W, 3)
    assert out["rgb_image"].dtype == spec.dtype
    assert torch.equal(rgb, before)


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_alpha_outside_unit_range_is_refused(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        DecisionOutcomeOverlay(alpha=alpha)


@pytest.mark.parametrize("colour", [(1.0, 0.0), (1.0, 0.0, 2.0)])
def test_colour_must_be_three_channels_in_unit_range(colour: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="hit_color"):
        DecisionOutcomeOverlay(hit_color=colour)


@pytest.mark.parametrize("argument", ["hit_color", "false_alarm_color", "miss_color"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_colour_is_refused(argument: str, bad: float) -> None:
    """A NaN channel passes a plain range check, and NaN times a zero outcome mask is NaN, so
    one bad colour would turn every pixel of every frame into NaN."""
    with pytest.raises(ValueError, match=argument):
        DecisionOutcomeOverlay(**{argument: (bad, 0.0, 0.0)})


class _ExportProbe(torch.nn.Module):
    """The node with tensors in and a tensor out, for torch.export."""

    def __init__(self, node: DecisionOutcomeOverlay) -> None:
        super().__init__()
        self.node = node

    def forward(
        self, rgb: torch.Tensor, decisions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return self.node(rgb_image=rgb, decisions=decisions, targets=targets)["rgb_image"]


def test_export_smoke() -> None:
    node = DecisionOutcomeOverlay(alpha=0.6).eval()
    rgb, decisions, targets = _frame()
    program = torch.export.export(_ExportProbe(node), (rgb, decisions, targets))

    other_rgb = torch.rand(B, H, W, 3)
    other_dec = torch.rand(B, H, W, 1) > 0.5
    other_tgt = torch.rand(B, H, W, 1) > 0.5
    traced = program.module()(other_rgb, other_dec, other_tgt)
    eager = node(rgb_image=other_rgb, decisions=other_dec, targets=other_tgt)["rgb_image"]
    assert torch.allclose(traced, eager, atol=1e-6)


@pytest.mark.slow
def test_onnx_runtime_matches_eager_with_targets() -> None:
    """The registry's ONNX contract builds only the required inputs, so it never exports the
    ``targets`` branch; this exports it and checks ONNX Runtime against eager."""
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    node = DecisionOutcomeOverlay(alpha=0.6).eval()
    probe = _ExportProbe(node).eval()
    rgb, decisions, targets = _frame()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "decision_outcome_overlay.onnx"
        torch.onnx.export(
            probe,
            (rgb, decisions, targets),
            str(path),
            dynamo=True,
            opset_version=18,
            external_data=False,
        )
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        names = [inp.name for inp in session.get_inputs()]
        (got,) = session.run(
            None, dict(zip(names, (rgb.numpy(), decisions.numpy(), targets.numpy()), strict=True))
        )

    np.testing.assert_allclose(got, _expected(alpha=0.6).numpy(), atol=1e-6)
