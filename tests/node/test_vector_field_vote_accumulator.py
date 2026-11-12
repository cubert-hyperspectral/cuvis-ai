"""Tests for VectorFieldVoteAccumulator: a plain Python loop reference, order handling,
dropped out-of-bounds/non-finite votes, half-to-even rounding, gradients, port contract,
export smoke, dynamo ONNX plus ONNX Runtime parity, and pipeline save/reload."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.node.vote_accumulation import VectorFieldVoteAccumulator
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

pytestmark = pytest.mark.unit


def _loop_reference(
    vectors: torch.Tensor, weights: torch.Tensor | None, order: str = "xy"
) -> torch.Tensor:
    """Plain Python double loop: the node's own reference, not its vectorised form."""
    b, h, w, _ = vectors.shape
    out = torch.zeros(b, h, w, 1, dtype=torch.float32)
    for bi in range(b):
        for y in range(h):
            for x in range(w):
                v0 = float(vectors[bi, y, x, 0])
                v1 = float(vectors[bi, y, x, 1])
                dx, dy = (v0, v1) if order == "xy" else (v1, v0)
                wgt = 1.0 if weights is None else float(weights[bi, y, x, 0])
                if not (math.isfinite(dx) and math.isfinite(dy) and math.isfinite(wgt)):
                    continue
                ty = y + _round_half_even(dy)
                tx = x + _round_half_even(dx)
                if not (0 <= ty < h and 0 <= tx < w):
                    continue
                out[bi, ty, tx, 0] += wgt
    return out


def _round_half_even(v: float) -> int:
    """Python's round() already rounds halves to even for floats; make the intent explicit."""
    return int(round(v))


def _vectors(seed: int, b: int = 2, h: int = 5, w: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(b, h, w, 2, generator=g) * 6.0 - 3.0).to(torch.float32)


# ---------------------------------------------------------------------------
# 1. Golden reference (loop) and basic maps
# ---------------------------------------------------------------------------


def test_matches_loop_reference_on_small_non_square_frames():
    vectors = _vectors(seed=0, b=2, h=5, w=7)
    node = VectorFieldVoteAccumulator()
    out = node(vectors=vectors)["scores"]
    expected = _loop_reference(vectors, None, order="xy")
    torch.testing.assert_close(out, expected)


def test_zero_field_gives_a_map_of_ones():
    b, h, w = 1, 4, 6
    vectors = torch.zeros(b, h, w, 2)
    out = VectorFieldVoteAccumulator()(vectors=vectors)["scores"]
    assert torch.equal(out, torch.ones(b, h, w, 1))


def test_all_vectors_point_at_one_pixel_gives_h_times_w_there():
    b, h, w = 1, 4, 5
    target_y, target_x = 2, 3
    rows = torch.arange(h).view(1, h, 1, 1).expand(b, h, w, 1)
    cols = torch.arange(w).view(1, 1, w, 1).expand(b, h, w, 1)
    dy = (target_y - rows).to(torch.float32)
    dx = (target_x - cols).to(torch.float32)
    vectors = torch.cat([dx, dy], dim=-1)
    out = VectorFieldVoteAccumulator()(vectors=vectors)["scores"]
    expected = torch.zeros(b, h, w, 1)
    expected[0, target_y, target_x, 0] = float(h * w)
    torch.testing.assert_close(out, expected)


def test_exact_expected_map_positions_and_values():
    """A hand-built tiny case checked position-by-position, not only by total."""
    # 3x3 frame; pixel (0,0) votes for (1,1); pixel (0,1) votes for (1,1) too (collision);
    # pixel (2,2) votes out of range.
    vectors = torch.zeros(1, 3, 3, 2)
    vectors[0, 0, 0] = torch.tensor([1.0, 1.0])  # dx, dy -> target (1, 1)
    vectors[0, 0, 1] = torch.tensor([0.0, 1.0])  # dx=0, dy=1 -> target row 1, col 1
    vectors[0, 2, 2] = torch.tensor([5.0, 5.0])  # out of range, dropped
    out = VectorFieldVoteAccumulator()(vectors=vectors)["scores"]
    # Every pixel but (0,0), (0,1) and (2,2) casts its own self-vote (zero vector).
    # (0,0) and (0,1) are redirected onto (1,1) instead of voting for themselves.
    # (2,2)'s out-of-range vote is dropped outright (no self-vote, no redirected vote).
    expected = torch.ones(1, 3, 3, 1)
    expected[0, 0, 0, 0] = 0.0
    expected[0, 0, 1, 0] = 0.0
    expected[0, 2, 2, 0] = 0.0
    expected[0, 1, 1, 0] = 1.0 + 2.0  # its own self-vote plus the two redirected onto it
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# 2. Dropped votes: out of bounds, non-finite
# ---------------------------------------------------------------------------


def test_out_of_bounds_and_non_finite_votes_dropped_sum_equals_votes_kept():
    b, h, w = 1, 4, 4
    vectors = torch.zeros(b, h, w, 2)
    vectors[0, 0, 0] = torch.tensor([100.0, 100.0])  # out of bounds
    vectors[0, 1, 1] = torch.tensor([float("nan"), 0.0])  # non-finite
    vectors[0, 2, 2] = torch.tensor([0.0, float("inf")])  # non-finite
    # the rest (4*4 - 3 = 13) vote for themselves (zero vector)
    out = VectorFieldVoteAccumulator()(vectors=vectors)["scores"]
    assert float(out.sum()) == 13.0


def test_non_finite_weight_drops_the_vote_too():
    b, h, w = 1, 1, 3
    vectors = torch.zeros(b, h, w, 2)
    weights = torch.tensor([[[[1.0], [float("nan")], [2.0]]]])
    out = VectorFieldVoteAccumulator()(vectors=vectors, weights=weights)["scores"]
    assert float(out.sum()) == pytest.approx(3.0)  # 1.0 (col0) + 0 (col1 dropped) + 2.0 (col2)


def test_huge_finite_offset_is_dropped():
    vectors = torch.zeros(1, 3, 3, 2)
    vectors[0, 1, 1] = torch.tensor([1e8, 1e8])
    out = VectorFieldVoteAccumulator()(vectors=vectors)["scores"]
    expected = torch.ones(1, 3, 3, 1)
    expected[0, 1, 1, 0] = 0.0  # its self-vote is replaced by the huge, out-of-range one
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# 3. Order handling
# ---------------------------------------------------------------------------


def test_xy_vs_yx_with_asymmetric_vectors():
    b, h, w = 1, 5, 7
    vectors = torch.zeros(b, h, w, 2)
    vectors[0, 1, 1] = torch.tensor([2.0, 0.0])  # channel0=2, channel1=0
    xy_out = VectorFieldVoteAccumulator(order="xy")(vectors=vectors)["scores"]
    yx_out = VectorFieldVoteAccumulator(order="yx")(vectors=vectors)["scores"]
    # xy: dx=2, dy=0 -> target (row=1, col=3), which also receives its own self-vote.
    # yx: dx=0, dy=2 -> target (row=3, col=1), likewise.
    assert xy_out[0, 1, 3, 0] == 2.0
    assert yx_out[0, 3, 1, 0] == 2.0
    assert not torch.equal(xy_out, yx_out)


@pytest.mark.parametrize("bad_order", ["zy", "", "XY", 3])
def test_invalid_order_raises(bad_order):
    with pytest.raises(ValueError, match="order"):
        VectorFieldVoteAccumulator(order=bad_order)


# ---------------------------------------------------------------------------
# 4. Weights
# ---------------------------------------------------------------------------


def test_weighted_votes_match_loop_reference():
    vectors = _vectors(seed=1, b=2, h=4, w=5)
    g = torch.Generator().manual_seed(2)
    weights = torch.rand(2, 4, 5, 1, generator=g).to(torch.float32)
    out = VectorFieldVoteAccumulator()(vectors=vectors, weights=weights)["scores"]
    expected = _loop_reference(vectors, weights, order="xy")
    torch.testing.assert_close(out, expected)


def test_weighted_collisions_sum_the_weights():
    vectors = torch.zeros(1, 2, 2, 2)
    vectors[0, 0, 0] = torch.tensor([1.0, 1.0])  # -> (1,1)
    vectors[0, 0, 1] = torch.tensor([0.0, 1.0])  # -> (1,1)
    weights = torch.tensor([[[[2.0], [3.0]], [[0.5], [0.5]]]])
    out = VectorFieldVoteAccumulator()(vectors=vectors, weights=weights)["scores"]
    # (1,1) collects both votes above plus its own self-vote (weight 0.5, zero vector).
    assert out[0, 1, 1, 0] == pytest.approx(2.0 + 3.0 + 0.5)


@pytest.mark.parametrize("bad_shape", [(2, 4, 5, 2), (2, 4, 6, 1), (2, 5, 5, 1), (1, 4, 5, 1)])
def test_weights_with_wrong_shape_raises(bad_shape):
    vectors = _vectors(seed=3, b=2, h=4, w=5)
    weights = torch.ones(*bad_shape, dtype=torch.float32)
    with pytest.raises(ValueError, match="weights"):
        VectorFieldVoteAccumulator()(vectors=vectors, weights=weights)


# ---------------------------------------------------------------------------
# 5. Batch independence, halves round to even
# ---------------------------------------------------------------------------


def test_batch_elements_do_not_mix():
    b, h, w = 2, 4, 4
    vectors = torch.zeros(b, h, w, 2)
    vectors[0, 0, 0] = torch.tensor([1.0, 1.0])  # (0,0) joins (1,1)'s self-vote
    vectors[1, 0, 1] = torch.tensor([2.0, 2.0])  # (0,1) joins (2,3)'s self-vote
    out = VectorFieldVoteAccumulator()(vectors=vectors)["scores"]
    assert out[0, 1, 1, 0] == 2.0  # (1,1)'s own self-vote plus the one from (0,0)
    assert out[1, 2, 3, 0] == 2.0  # (2,3)'s own self-vote plus the one from (0,1)
    assert (
        out[1, 1, 1, 0] == 1.0
    )  # batch 1's own self-vote only; batch 0's extra vote does not leak


@pytest.mark.parametrize(
    ("source", "offset", "expected_target", "lands_elsewhere"),
    [
        (2, 0.5, 2, False),  # round(0.5) -> 0 (nearest even): target == source
        (2, 1.5, 4, True),  # round(1.5) -> 2 (nearest even): target = 2 + 2 = 4
        (3, 2.5, 5, True),  # round(2.5) -> 2 (nearest even): target = 3 + 2 = 5
        (3, -0.5, 3, False),  # round(-0.5) -> 0 (nearest even): target == source
        (4, 1.5, 6, True),  # round(1.5) -> 2 (nearest even): target = 4 + 2 = 6
    ],
)
def test_halves_round_to_even(source, offset, expected_target, lands_elsewhere):
    h = w = 9
    vectors = torch.zeros(1, h, w, 2)
    vectors[0, source, source] = torch.tensor([offset, 0.0])  # move along columns only
    out = VectorFieldVoteAccumulator()(vectors=vectors)["scores"]
    # The target also casts its own self-vote (zero vector) unless target == source,
    # in which case the redirected vote and the self-vote are the same single vote.
    expected = 2.0 if lands_elsewhere else 1.0
    assert out[0, source, expected_target, 0] == expected


# ---------------------------------------------------------------------------
# 6. Gradients
# ---------------------------------------------------------------------------


def test_gradient_of_dropped_weight_is_zero():
    vectors = torch.zeros(1, 1, 3, 2)
    vectors[0, 0, 0] = torch.tensor([100.0, 0.0])  # out of range: dropped
    weights = torch.ones(1, 1, 3, 1, requires_grad=True)
    out = VectorFieldVoteAccumulator()(vectors=vectors, weights=weights)["scores"]
    (out.sum() * torch.tensor([1.0, 2.0, 3.0]).view(1, 1, 3, 1)).sum().backward()
    assert weights.grad is not None
    assert float(weights.grad[0, 0, 0, 0]) == 0.0


def test_gradient_of_kept_weight_matches_non_uniform_loss():
    vectors = torch.zeros(1, 1, 2, 2)  # both pixels vote for themselves
    weights = torch.ones(1, 1, 2, 1, requires_grad=True)
    out = VectorFieldVoteAccumulator()(vectors=vectors, weights=weights)["scores"]
    coeffs = torch.tensor([1.0, 2.0]).view(1, 1, 2, 1)
    (out * coeffs).sum().backward()
    assert weights.grad is not None
    torch.testing.assert_close(weights.grad, coeffs)


def test_no_gradient_flows_to_vectors():
    """vectors feeds only the rounding path, so the output's graph (if any) never reaches
    back to it: either the output needs no grad at all, or an explicit grad request for
    vectors comes back as None (unused) or all zero."""
    vectors = torch.zeros(1, 1, 2, 2, requires_grad=True)
    out = VectorFieldVoteAccumulator()(vectors=vectors)["scores"]
    loss = out.sum()
    if not loss.requires_grad:
        return
    (grad,) = torch.autograd.grad(loss, vectors, allow_unused=True, retain_graph=True)
    assert grad is None or torch.all(grad == 0.0)


# ---------------------------------------------------------------------------
# 7. Port contract
# ---------------------------------------------------------------------------


def test_port_contract_shape_and_dtype():
    vectors = _vectors(seed=4, b=1, h=3, w=3)
    out = VectorFieldVoteAccumulator()(vectors=vectors)
    assert "scores" in out
    assert out["scores"].shape == (1, 3, 3, 1)
    assert out["scores"].dtype == torch.float32


def test_category_and_tags_are_set():
    node = VectorFieldVoteAccumulator()
    assert node.get_category() == NodeCategory.TRANSFORM
    assert NodeTag.ONNX in node.get_tags()


# ---------------------------------------------------------------------------
# 8. Export smoke, dynamo ONNX + ONNX Runtime parity
# ---------------------------------------------------------------------------


class _ExportProbeNoWeights(torch.nn.Module):
    def __init__(self, node: VectorFieldVoteAccumulator) -> None:
        super().__init__()
        self.node = node

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.node(vectors=vectors)["scores"]


class _ExportProbeWithWeights(torch.nn.Module):
    def __init__(self, node: VectorFieldVoteAccumulator) -> None:
        super().__init__()
        self.node = node

    def forward(self, vectors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return self.node(vectors=vectors, weights=weights)["scores"]


def test_export_smoke_matches_forward_without_weights():
    node = VectorFieldVoteAccumulator().eval()
    example = _vectors(seed=5, b=1, h=4, w=4)
    program = torch.export.export(_ExportProbeNoWeights(node), (example,))
    other = _vectors(seed=6, b=1, h=4, w=4)
    traced = program.module()(other)
    expected = node(vectors=other)["scores"]
    torch.testing.assert_close(traced, expected)


def test_export_smoke_matches_forward_with_weights_connected():
    node = VectorFieldVoteAccumulator().eval()
    example = _vectors(seed=7, b=1, h=4, w=4)
    example_w = torch.rand(1, 4, 4, 1, generator=torch.Generator().manual_seed(8))
    program = torch.export.export(_ExportProbeWithWeights(node), (example, example_w))
    other = _vectors(seed=9, b=1, h=4, w=4)
    other_w = torch.rand(1, 4, 4, 1, generator=torch.Generator().manual_seed(10))
    traced = program.module()(other, other_w)
    expected = node(vectors=other, weights=other_w)["scores"]
    torch.testing.assert_close(traced, expected)


@pytest.mark.slow
def test_dynamo_export_matches_onnx_runtime_collision_heavy_weighted_case():
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    b, h, w = 1, 5, 5
    g = torch.Generator().manual_seed(11)
    # Small-magnitude vectors so many pixels collide into the same few targets.
    vectors = (torch.rand(b, h, w, 2, generator=g) * 2.0 - 1.0).to(torch.float32)
    weights = torch.rand(b, h, w, 1, generator=torch.Generator().manual_seed(12)).to(torch.float32)

    node = VectorFieldVoteAccumulator().eval()
    probe = _ExportProbeWithWeights(node).eval()
    with torch.no_grad():
        eager = probe(vectors, weights)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vote_accumulator.onnx"
        torch.onnx.export(probe, (vectors, weights), str(path), dynamo=True, opset_version=18)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (got,) = session.run(
            None,
            {
                session.get_inputs()[0].name: vectors.numpy(),
                session.get_inputs()[1].name: weights.numpy(),
            },
        )
    np.testing.assert_allclose(got, eager.numpy(), atol=1e-5, rtol=1e-4)
    reference = _loop_reference(vectors, weights, order="xy").numpy()
    np.testing.assert_allclose(got, reference, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# 9. Pipeline save / reload
# ---------------------------------------------------------------------------


class _VectorSource(Node):
    """Module-scope test source emitting a fixed vector field, for pipeline reload tests."""

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.TORCH})
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"vectors": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 2))}

    def forward(self, **_) -> dict[str, torch.Tensor]:
        return {"vectors": _vectors(seed=42, b=1, h=4, w=4)}


def test_pipeline_save_and_reload_round_trips(tmp_path):
    pipe = CuvisPipeline("vote_accumulator_reload")
    src = _VectorSource(name="src")
    node = VectorFieldVoteAccumulator(name="votes", order="xy")
    pipe.connect(src.outputs.vectors, node.inputs.vectors)
    before = pipe.forward(batch={}, stage=ExecutionStage.INFERENCE)[("votes", "scores")]
    path = tmp_path / "vote_accumulator.yaml"
    pipe.save_to_file(str(path))
    rebuilt = PipelineBuilder().build_from_config(str(path))
    after = rebuilt.forward(batch={}, stage=ExecutionStage.INFERENCE)[("votes", "scores")]
    torch.testing.assert_close(after, before)
