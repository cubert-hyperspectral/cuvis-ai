"""Tests for LocalMaximaSuppression: a plain loop reference, plateau tie rule, borders,
windows, chained use after VectorFieldVoteAccumulator, port contract, export smoke, dynamo
ONNX plus ONNX Runtime parity for both tie rules, and pipeline save/reload."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.node.spatial import LocalMaximaSuppression
from cuvis_ai.node.vote_accumulation import VectorFieldVoteAccumulator
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

pytestmark = pytest.mark.unit


def _window_hw(window: int | list[int]) -> tuple[int, int]:
    if isinstance(window, int):
        return window, window
    return int(window[0]), int(window[1])


def _loop_reference(
    scores: torch.Tensor, window: int | list[int], keep_plateaus: bool
) -> torch.Tensor:
    """Plain Python triple loop (channel, row, col) with explicit earlier/later rectangles,
    independent of the node's own vectorised implementation."""
    b, h, w, c = scores.shape
    wh, ww = _window_hw(window)
    rh, rw = wh // 2, ww // 2
    out = torch.zeros_like(scores)
    for bi in range(b):
        for ci in range(c):
            plane = scores[bi, :, :, ci]
            for y in range(h):
                for x in range(w):
                    val = float(plane[y, x])
                    y0, y1 = max(0, y - rh), min(h - 1, y + rh)
                    x0, x1 = max(0, x - rw), min(w - 1, x + rw)
                    window_vals = [
                        float(plane[yy, xx]) for yy in range(y0, y1 + 1) for xx in range(x0, x1 + 1)
                    ]
                    if val != max(window_vals):
                        continue
                    if keep_plateaus:
                        out[bi, y, x, ci] = val
                        continue
                    earlier_vals = [
                        float(plane[yy, xx])
                        for yy in range(y0, y1 + 1)
                        for xx in range(x0, x1 + 1)
                        if (yy, xx) < (y, x)
                    ]
                    if not earlier_vals or val > max(earlier_vals):
                        out[bi, y, x, ci] = val
    return out


def _scores(seed: int, b: int = 1, h: int = 7, w: int = 9, c: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, h, w, c, generator=g).to(torch.float32)


# ---------------------------------------------------------------------------
# 1. Golden reference (loop)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("keep_plateaus", [False, True])
@pytest.mark.parametrize("window", [3, 5, [3, 5], [5, 3]])
def test_matches_loop_reference(window, keep_plateaus):
    scores = _scores(seed=0, b=2, h=8, w=10, c=2)
    node = LocalMaximaSuppression(window=window, keep_plateaus=keep_plateaus)
    out = node(scores=scores)["scores"]
    expected = _loop_reference(scores, window, keep_plateaus)
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# 2. Single peak, two close peaks
# ---------------------------------------------------------------------------


def test_single_peak_survives_and_rest_is_zero():
    scores = torch.zeros(1, 7, 7, 1)
    scores[0, 3, 3, 0] = 5.0
    out = LocalMaximaSuppression(window=3)(scores=scores)["scores"]
    expected = torch.zeros(1, 7, 7, 1)
    expected[0, 3, 3, 0] = 5.0
    torch.testing.assert_close(out, expected)


def test_two_peaks_closer_than_window_only_the_larger_survives():
    scores = torch.zeros(1, 7, 7, 1)
    scores[0, 3, 3, 0] = 5.0
    scores[0, 3, 4, 0] = 3.0  # within the 3x3 window of (3,3)
    out = LocalMaximaSuppression(window=3)(scores=scores)["scores"]
    expected = torch.zeros(1, 7, 7, 1)
    expected[0, 3, 3, 0] = 5.0
    torch.testing.assert_close(out, expected)


def test_two_peaks_farther_than_window_both_survive():
    scores = torch.zeros(1, 9, 9, 1)
    scores[0, 1, 1, 0] = 5.0
    scores[0, 7, 7, 0] = 3.0
    out = LocalMaximaSuppression(window=3)(scores=scores)["scores"]
    expected = torch.zeros(1, 9, 9, 1)
    expected[0, 1, 1, 0] = 5.0
    expected[0, 7, 7, 0] = 3.0
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# 3. Plateau tie rule
# ---------------------------------------------------------------------------


def test_2x2_plateau_under_3x3_window_collapses_to_first_point():
    scores = torch.zeros(1, 5, 5, 1)
    scores[0, 2, 2, 0] = scores[0, 2, 3, 0] = scores[0, 3, 2, 0] = scores[0, 3, 3, 0] = 4.0
    out = LocalMaximaSuppression(window=3, keep_plateaus=False)(scores=scores)["scores"]
    expected = torch.zeros(1, 5, 5, 1)
    expected[0, 2, 2, 0] = 4.0  # first in row-major order
    torch.testing.assert_close(out, expected)


def test_2x2_plateau_under_3x3_window_keeps_all_four_with_keep_plateaus():
    scores = torch.zeros(1, 5, 5, 1)
    scores[0, 2, 2, 0] = scores[0, 2, 3, 0] = scores[0, 3, 2, 0] = scores[0, 3, 3, 0] = 4.0
    out = LocalMaximaSuppression(window=3, keep_plateaus=True)(scores=scores)["scores"]
    expected = torch.zeros(1, 5, 5, 1)
    expected[0, 2, 2, 0] = expected[0, 2, 3, 0] = expected[0, 3, 2, 0] = expected[0, 3, 3, 0] = 4.0
    torch.testing.assert_close(out, expected)


def test_long_plateau_in_a_row_keeps_only_the_first_point():
    scores = torch.zeros(1, 3, 9, 1)
    scores[0, 1, 2:7, 0] = 2.0  # a run of five equal values in the middle row
    out = LocalMaximaSuppression(window=3)(scores=scores)["scores"]
    expected = _loop_reference(scores, window=3, keep_plateaus=False)
    torch.testing.assert_close(out, expected)
    assert float(expected[0, 1, 2, 0]) == 2.0
    assert bool((expected[0, 1, 3:7, 0] == 0.0).all())


def test_bent_plateau_can_leave_several_points():
    """An L-shaped plateau with a one-cell gap at its elbow (so the bend sits outside a 3x3
    window's reach from the arm's own first point) leaves one surviving point per arm, even
    though the plateau is still a single connected region under 8-connectivity; the node
    makes no stronger promise than the loop reference."""
    scores = torch.zeros(1, 9, 9, 1)
    for row in range(1, 6):
        scores[0, row, 1, 0] = 3.0  # vertical arm, column 1
    for col in range(3, 8):
        scores[0, 5, col, 0] = 3.0  # horizontal arm, starting two columns past the elbow
    out = LocalMaximaSuppression(window=3)(scores=scores)["scores"]
    expected = _loop_reference(scores, window=3, keep_plateaus=False)
    torch.testing.assert_close(out, expected)
    assert int((expected > 0).sum()) == 2
    assert float(expected[0, 1, 1, 0]) == 3.0
    assert float(expected[0, 5, 3, 0]) == 3.0


def test_branching_plateau_matches_loop_reference():
    scores = torch.zeros(1, 9, 9, 1)
    scores[0, 4, 1:8, 0] = 1.5  # horizontal bar
    scores[0, 1:8, 4, 0] = 1.5  # crossing vertical bar
    out = LocalMaximaSuppression(window=3)(scores=scores)["scores"]
    expected = _loop_reference(scores, window=3, keep_plateaus=False)
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# 4. Borders, corners, non-square windows
# ---------------------------------------------------------------------------


def test_border_peak_survives_against_minus_infinity_padding():
    scores = torch.zeros(1, 5, 5, 1)
    scores[0, 0, 0, 0] = 1.0  # corner: the window is mostly off-image
    out = LocalMaximaSuppression(window=3)(scores=scores)["scores"]
    expected = torch.zeros(1, 5, 5, 1)
    expected[0, 0, 0, 0] = 1.0
    torch.testing.assert_close(out, expected)


def test_negative_peak_at_the_border_keeps_its_value():
    scores = torch.full((1, 5, 5, 1), -5.0)
    scores[0, 0, 2, 0] = -1.0  # less negative than its neighbours: the local maximum
    out = LocalMaximaSuppression(window=3)(scores=scores)["scores"]
    expected = _loop_reference(scores, window=3, keep_plateaus=False)
    torch.testing.assert_close(out, expected)
    assert float(expected[0, 0, 2, 0]) == -1.0


def test_non_square_window_matches_loop_reference():
    scores = _scores(seed=1, b=1, h=6, w=11, c=1)
    out = LocalMaximaSuppression(window=[3, 7])(scores=scores)["scores"]
    expected = _loop_reference(scores, window=[3, 7], keep_plateaus=False)
    torch.testing.assert_close(out, expected)


def test_window_larger_than_image_matches_loop_reference():
    scores = _scores(seed=2, b=1, h=3, w=3, c=1)
    out = LocalMaximaSuppression(window=9)(scores=scores)["scores"]
    expected = _loop_reference(scores, window=9, keep_plateaus=False)
    torch.testing.assert_close(out, expected)
    assert int((expected > 0).sum()) == 1  # only the frame's global maximum survives


# ---------------------------------------------------------------------------
# 5. 1x1 window identity, singleton axes, all-zero map
# ---------------------------------------------------------------------------


def test_1x1_window_is_identity():
    scores = _scores(seed=3, b=1, h=4, w=4, c=1)
    out = LocalMaximaSuppression(window=1)(scores=scores)["scores"]
    torch.testing.assert_close(out, scores)


def test_1x1_window_is_identity_with_keep_plateaus():
    scores = _scores(seed=4, b=1, h=4, w=4, c=1)
    out = LocalMaximaSuppression(window=1, keep_plateaus=True)(scores=scores)["scores"]
    torch.testing.assert_close(out, scores)


def test_singleton_height_axis_matches_loop_reference():
    scores = _scores(seed=5, b=1, h=1, w=8, c=1)
    out = LocalMaximaSuppression(window=3)(scores=scores)["scores"]
    expected = _loop_reference(scores, window=3, keep_plateaus=False)
    torch.testing.assert_close(out, expected)


def test_singleton_width_axis_matches_loop_reference():
    scores = _scores(seed=6, b=1, h=8, w=1, c=1)
    out = LocalMaximaSuppression(window=3)(scores=scores)["scores"]
    expected = _loop_reference(scores, window=3, keep_plateaus=False)
    torch.testing.assert_close(out, expected)


def test_all_zero_map_every_pixel_is_a_tied_maximum_only_first_survives():
    scores = torch.zeros(1, 4, 4, 1)
    out = LocalMaximaSuppression(window=3)(scores=scores)["scores"]
    expected = _loop_reference(scores, window=3, keep_plateaus=False)
    torch.testing.assert_close(out, expected)
    assert float(expected[0, 0, 0, 0]) == 0.0  # the "surviving" value is still zero
    # exactly the cells that are the unique row-major-first tied max of their own window
    assert bool((expected == 0.0).all())  # every value in the map is 0 either way


# ---------------------------------------------------------------------------
# 6. Several channels, batch independence
# ---------------------------------------------------------------------------


def test_channels_do_not_mix():
    scores = torch.zeros(1, 5, 5, 2)
    scores[0, 2, 2, 0] = 9.0
    scores[0, 2, 2, 1] = 1.0
    scores[0, 1, 1, 1] = 9.0  # different peak location on channel 1
    out = LocalMaximaSuppression(window=3)(scores=scores)["scores"]
    expected = _loop_reference(scores, window=3, keep_plateaus=False)
    torch.testing.assert_close(out, expected)
    assert float(expected[0, 2, 2, 0]) == 9.0
    assert float(expected[0, 1, 1, 1]) == 9.0
    assert float(expected[0, 2, 2, 1]) == 0.0


def test_batch_elements_do_not_mix():
    scores = torch.zeros(2, 5, 5, 1)
    scores[0, 2, 2, 0] = 9.0
    scores[1, 1, 1, 0] = 9.0
    out = LocalMaximaSuppression(window=3)(scores=scores)["scores"]
    expected = _loop_reference(scores, window=3, keep_plateaus=False)
    torch.testing.assert_close(out, expected)
    assert float(expected[0, 1, 1, 0]) == 0.0
    assert float(expected[1, 2, 2, 0]) == 0.0


# ---------------------------------------------------------------------------
# 7. Chained after VectorFieldVoteAccumulator
# ---------------------------------------------------------------------------


def test_chained_after_vote_accumulator_surviving_peaks_equal_centroids():
    """A synthetic field where every pixel's vector points to its nearest of two known
    centroids; after voting and suppression, exactly two peaks survive, at those centroids."""
    h, w = 15, 15
    centroids = [(4, 4), (11, 10)]
    rows = torch.arange(h).view(h, 1).expand(h, w)
    cols = torch.arange(w).view(1, w).expand(h, w)
    dists = torch.stack(
        [torch.sqrt((rows - cy) ** 2.0 + (cols - cx) ** 2.0) for cy, cx in centroids]
    )
    nearest = dists.argmin(dim=0)  # [h, w], 0 or 1
    dy = torch.zeros(h, w)
    dx = torch.zeros(h, w)
    for idx, (cy, cx) in enumerate(centroids):
        mask = nearest == idx
        dy[mask] = (cy - rows[mask]).to(torch.float32)
        dx[mask] = (cx - cols[mask]).to(torch.float32)
    vectors = torch.stack([dx, dy], dim=-1).unsqueeze(0)

    votes = VectorFieldVoteAccumulator()(vectors=vectors)["scores"]
    peaks = LocalMaximaSuppression(window=5)(scores=votes)["scores"]
    surviving = (peaks[0, :, :, 0] > 0).nonzero(as_tuple=False)
    assert surviving.shape[0] == len(centroids)
    found = {(int(r), int(c)) for r, c in surviving}
    assert found == set(centroids)


# ---------------------------------------------------------------------------
# 8. Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_window", [2, 4, 0, -1, -3])
def test_even_or_non_positive_int_window_raises(bad_window):
    with pytest.raises(ValueError, match="window"):
        LocalMaximaSuppression(window=bad_window)


@pytest.mark.parametrize("bad_window", [[2, 3], [3, 2], [3, 4, 5], [3], [0, 3]])
def test_invalid_list_window_raises(bad_window):
    with pytest.raises(ValueError, match="window"):
        LocalMaximaSuppression(window=bad_window)


def test_non_integer_window_raises():
    with pytest.raises(ValueError, match="window"):
        LocalMaximaSuppression(window=3.0)


def test_non_bool_keep_plateaus_raises():
    with pytest.raises(ValueError, match="keep_plateaus"):
        LocalMaximaSuppression(keep_plateaus="yes")


# ---------------------------------------------------------------------------
# 9. Port contract
# ---------------------------------------------------------------------------


def test_port_contract_shape_and_dtype():
    scores = _scores(seed=7, b=1, h=5, w=5, c=3)
    out = LocalMaximaSuppression(window=3)(scores=scores)
    assert "scores" in out
    assert out["scores"].shape == (1, 5, 5, 3)
    assert out["scores"].dtype == torch.float32


def test_category_and_tags_are_set():
    node = LocalMaximaSuppression()
    assert node.get_category() == NodeCategory.TRANSFORM
    assert NodeTag.ONNX in node.get_tags()


# ---------------------------------------------------------------------------
# 10. Export smoke, dynamo ONNX + ONNX Runtime parity for both tie rules
# ---------------------------------------------------------------------------


class _ExportProbe(torch.nn.Module):
    def __init__(self, node: LocalMaximaSuppression) -> None:
        super().__init__()
        self.node = node

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return self.node(scores=scores)["scores"]


@pytest.mark.parametrize("keep_plateaus", [False, True])
def test_export_smoke_matches_forward(keep_plateaus):
    node = LocalMaximaSuppression(window=3, keep_plateaus=keep_plateaus).eval()
    example = _scores(seed=8, b=1, h=6, w=6, c=1)
    program = torch.export.export(_ExportProbe(node), (example,))
    other = _scores(seed=9, b=1, h=6, w=6, c=1)
    traced = program.module()(other)
    expected = node(scores=other)["scores"]
    torch.testing.assert_close(traced, expected)


@pytest.mark.slow
@pytest.mark.parametrize("keep_plateaus", [False, True])
def test_dynamo_export_matches_onnx_runtime(keep_plateaus):
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    scores = torch.zeros(1, 6, 6, 1)
    scores[0, 2, 2, 0] = scores[0, 2, 3, 0] = scores[0, 3, 2, 0] = scores[0, 3, 3, 0] = 4.0
    scores[0, 0, 0, 0] = 2.0  # a border tie-breaker

    node = LocalMaximaSuppression(window=3, keep_plateaus=keep_plateaus).eval()
    probe = _ExportProbe(node).eval()
    with torch.no_grad():
        eager = probe(scores)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"local_maxima_{keep_plateaus}.onnx"
        torch.onnx.export(probe, (scores,), str(path), dynamo=True, opset_version=18)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (got,) = session.run(None, {session.get_inputs()[0].name: scores.numpy()})
    np.testing.assert_allclose(got, eager.numpy(), atol=1e-5, rtol=1e-4)
    reference = _loop_reference(scores, window=3, keep_plateaus=keep_plateaus).numpy()
    np.testing.assert_allclose(got, reference, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# 11. Pipeline save / reload
# ---------------------------------------------------------------------------


class _ScoreSource(Node):
    """Module-scope test source emitting a fixed score map, for pipeline reload tests."""

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.TORCH})
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"scores": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))}

    def forward(self, **_) -> dict[str, torch.Tensor]:
        return {"scores": _scores(seed=55, b=1, h=5, w=5, c=1)}


def test_pipeline_save_and_reload_round_trips(tmp_path):
    pipe = CuvisPipeline("local_maxima_reload")
    src = _ScoreSource(name="src")
    node = LocalMaximaSuppression(name="peaks", window=3)
    pipe.connect(src.outputs.scores, node.inputs.scores)
    before = pipe.forward(batch={}, stage=ExecutionStage.INFERENCE)[("peaks", "scores")]
    path = tmp_path / "local_maxima.yaml"
    pipe.save_to_file(str(path))
    rebuilt = PipelineBuilder().build_from_config(str(path))
    after = rebuilt.forward(batch={}, stage=ExecutionStage.INFERENCE)[("peaks", "scores")]
    torch.testing.assert_close(after, before)
