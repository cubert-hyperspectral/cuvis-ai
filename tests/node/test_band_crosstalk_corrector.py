"""Tests for BandCrosstalkCorrector: identity start, matrix product reference, trace
normalization, clipping, freeze/unfreeze, gradient flow and export parity."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from cuvis_ai.node.pretreatments.crosstalk import BandCrosstalkCorrector

pytestmark = pytest.mark.unit

B, H, W, C = 2, 4, 5, 6


def _cube(seed: int = 0, b: int = B, h: int = H, w: int = W, c: int = C) -> torch.Tensor:
    return torch.rand(b, h, w, c, generator=torch.Generator().manual_seed(seed))


def _nonsymmetric_matrix(c: int = C, seed: int = 1) -> torch.Tensor:
    """A square matrix with a non-symmetric, non-trivial structure that catches a
    transposed einsum (row/column swap changes the result)."""
    g = torch.Generator().manual_seed(seed)
    return torch.eye(c) + 0.1 * torch.randn(c, c, generator=g)


# ---------------------------------------------------------------------------
# 1. Golden reference
# ---------------------------------------------------------------------------


def test_identity_at_construction_equals_relu_input() -> None:
    """A freshly constructed node applies the identity matrix: output is relu(input)."""
    node = BandCrosstalkCorrector(input_channels=C)
    cube = _cube(seed=2) * 2.0 - 1.0  # allow negative values so relu has an effect
    out = node.forward(cube=cube)["cube"]
    expected = torch.relu(cube)
    torch.testing.assert_close(out, expected)


def test_matrix_product_reference_catches_transposed_einsum() -> None:
    """Output equals an explicit matrix product against a non-symmetric matrix."""
    node = BandCrosstalkCorrector(input_channels=C, clip_negative=False)
    matrix = _nonsymmetric_matrix()
    with torch.no_grad():
        node.weights.copy_(matrix)
    cube = _cube(seed=3)
    out = node.forward(cube=cube)["cube"]
    # out[b,h,w,d] = sum_c matrix[d,c] * cube[b,h,w,c]
    expected = torch.einsum("dc,bhwc->bhwd", matrix, cube)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)


def test_normalize_trace_rescales_to_trace_c_and_matches_manual_rescale() -> None:
    """With normalize_trace, the applied matrix has trace C and matches a manual rescale."""
    node = BandCrosstalkCorrector(input_channels=C, clip_negative=False, normalize_trace=True)
    matrix = _nonsymmetric_matrix()
    with torch.no_grad():
        node.weights.copy_(matrix)
    cube = _cube(seed=4)
    out = node.forward(cube=cube)["cube"]

    scale = C / torch.trace(matrix)
    rescaled = matrix * scale
    assert torch.allclose(torch.trace(rescaled), torch.tensor(float(C)), atol=1e-5)
    expected = torch.einsum("dc,bhwc->bhwd", rescaled, cube)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)


def test_normalize_trace_off_uses_raw_product_so_ignored_flag_fails() -> None:
    """With normalize_trace off, the raw (unscaled) product is used; this differs from
    the trace-rescaled product whenever the matrix's trace is not already C, so an
    implementation that silently always rescales fails this test."""
    node = BandCrosstalkCorrector(input_channels=C, clip_negative=False, normalize_trace=False)
    matrix = _nonsymmetric_matrix()
    with torch.no_grad():
        node.weights.copy_(matrix)
    cube = _cube(seed=5)
    out = node.forward(cube=cube)["cube"]

    raw_expected = torch.einsum("dc,bhwc->bhwd", matrix, cube)
    torch.testing.assert_close(out, raw_expected, atol=1e-6, rtol=1e-5)

    scale = C / torch.trace(matrix)
    rescaled_expected = torch.einsum("dc,bhwc->bhwd", matrix * scale, cube)
    assert not torch.allclose(out, rescaled_expected, atol=1e-4)


def test_negative_trace_rescales_correctly() -> None:
    """normalize_trace holds for a negative trace too: the rescaled matrix has trace C."""
    matrix = -torch.eye(C) + 0.05 * torch.randn(C, C, generator=torch.Generator().manual_seed(6))
    assert torch.trace(matrix) < 0
    node = BandCrosstalkCorrector(input_channels=C, clip_negative=False, normalize_trace=True)
    with torch.no_grad():
        node.weights.copy_(matrix)
    cube = _cube(seed=7)
    out = node.forward(cube=cube)["cube"]

    scale = C / torch.trace(matrix)
    expected = torch.einsum("dc,bhwc->bhwd", matrix * scale, cube)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. Clipping
# ---------------------------------------------------------------------------


def test_clip_negative_true_output_is_non_negative() -> None:
    """With clip_negative on, every output value is >= 0 even with negative inputs."""
    node = BandCrosstalkCorrector(input_channels=C, clip_negative=True)
    matrix = _nonsymmetric_matrix()
    with torch.no_grad():
        node.weights.copy_(matrix)
    cube = _cube(seed=8) * 4.0 - 2.0
    out = node.forward(cube=cube)["cube"]
    assert torch.all(out >= 0.0)


def test_clip_negative_false_negative_values_pass_through() -> None:
    """With clip_negative off, negative products are not clamped."""
    node = BandCrosstalkCorrector(input_channels=C, clip_negative=False)
    matrix = -torch.eye(C)
    with torch.no_grad():
        node.weights.copy_(matrix)
    cube = _cube(seed=9)  # strictly positive input
    out = node.forward(cube=cube)["cube"]
    assert torch.all(out <= 0.0)
    assert torch.any(out < 0.0)


# ---------------------------------------------------------------------------
# 3. Trace-magnitude guard
# ---------------------------------------------------------------------------


def test_near_zero_trace_raises_value_error() -> None:
    """A trace whose magnitude is below eps raises ValueError instead of NaN or inf."""
    node = BandCrosstalkCorrector(input_channels=C, normalize_trace=True, eps=1e-6)
    # Build a matrix whose trace is exactly zero.
    matrix = _nonsymmetric_matrix()
    matrix = matrix - torch.eye(C) * (torch.trace(matrix) / C)
    with torch.no_grad():
        node.weights.copy_(matrix)
    assert abs(float(torch.trace(matrix))) < 1e-6
    cube = _cube(seed=10)
    with pytest.raises(ValueError, match="trace"):
        node.forward(cube=cube)


def test_zero_trace_state_dict_loads_and_forward_then_raises() -> None:
    """A state dict with a zero-trace matrix loads without error; forward then raises."""
    node = BandCrosstalkCorrector(input_channels=C, normalize_trace=True, eps=1e-6)
    matrix = _nonsymmetric_matrix()
    matrix = matrix - torch.eye(C) * (torch.trace(matrix) / C)
    with torch.no_grad():
        node.weights.copy_(matrix)
    state = node.state_dict()

    fresh = BandCrosstalkCorrector(input_channels=C, normalize_trace=True, eps=1e-6)
    fresh.load_state_dict(state)  # must not raise

    cube = _cube(seed=11)
    with pytest.raises(ValueError, match="trace"):
        fresh.forward(cube=cube)


def test_trace_guard_not_applied_when_normalize_trace_off() -> None:
    """A zero-trace matrix is fine when normalize_trace is off (no rescale, no guard)."""
    node = BandCrosstalkCorrector(input_channels=C, normalize_trace=False)
    matrix = _nonsymmetric_matrix()
    matrix = matrix - torch.eye(C) * (torch.trace(matrix) / C)
    with torch.no_grad():
        node.weights.copy_(matrix)
    cube = _cube(seed=12)
    out = node.forward(cube=cube)["cube"]  # should not raise
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 4. Port contract
# ---------------------------------------------------------------------------


def test_port_contract_shape_and_dtype() -> None:
    node = BandCrosstalkCorrector(input_channels=C)
    cube = _cube(seed=13)
    out = node.forward(cube=cube)
    assert "cube" in out
    tensor = out["cube"]
    assert tensor.shape == (B, H, W, C)
    assert tensor.dtype == torch.float32


def test_weights_buffer_is_identity_at_construction() -> None:
    node = BandCrosstalkCorrector(input_channels=C)
    torch.testing.assert_close(node.weights, torch.eye(C))


# ---------------------------------------------------------------------------
# 5. Freeze / unfreeze and gradients
# ---------------------------------------------------------------------------


def test_freeze_unfreeze_cycle() -> None:
    node = BandCrosstalkCorrector(input_channels=C)
    before = node.weights.clone()

    node.unfreeze()
    assert isinstance(node.weights, torch.nn.Parameter)
    assert node.weights.requires_grad
    torch.testing.assert_close(node.weights.detach(), before)

    node.freeze()
    assert not isinstance(node.weights, torch.nn.Parameter)
    torch.testing.assert_close(node.weights, before)


def test_gradients_reach_weights_after_unfreeze() -> None:
    node = BandCrosstalkCorrector(input_channels=C)
    node.unfreeze()
    cube = _cube(seed=14)
    out = node.forward(cube=cube)["cube"]
    loss = out.mean()
    loss.backward()
    assert node.weights.grad is not None
    assert torch.isfinite(node.weights.grad).all()
    assert torch.any(node.weights.grad != 0.0)


@pytest.mark.parametrize("normalize_trace", [False, True])
def test_optimizer_step_lowers_loss_and_outputs_stay_finite(normalize_trace: bool) -> None:
    torch.manual_seed(0)
    node = BandCrosstalkCorrector(input_channels=C, normalize_trace=normalize_trace)
    node.unfreeze()
    optimizer = torch.optim.SGD(node.parameters(), lr=0.01)

    cube = _cube(seed=15)
    target = torch.zeros_like(cube)

    out = node.forward(cube=cube)["cube"]
    loss_before = torch.nn.functional.mse_loss(out, target)
    loss_before_value = float(loss_before.detach())
    loss_before.backward()
    optimizer.step()
    optimizer.zero_grad()

    out_after = node.forward(cube=cube)["cube"]
    loss_after = torch.nn.functional.mse_loss(out_after, target)

    assert torch.isfinite(out_after).all()
    assert float(loss_after.detach()) < loss_before_value


# ---------------------------------------------------------------------------
# 6. Export parity
# ---------------------------------------------------------------------------


class _ExportProbe(torch.nn.Module):
    """The node, or its export_module() adapter, with a tensor in and a tensor out."""

    def __init__(self, node: BandCrosstalkCorrector) -> None:
        super().__init__()
        adapter = node.export_module()
        self.module = adapter if adapter is not None else node

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        return self.module(cube=cube)["cube"]


@pytest.mark.parametrize("normalize_trace", [False, True])
@pytest.mark.parametrize("clip_negative", [False, True])
def test_export_smoke_matches_forward(normalize_trace: bool, clip_negative: bool) -> None:
    """torch.export traces the adapter and reproduces forward on a second example."""
    node = BandCrosstalkCorrector(
        input_channels=C, normalize_trace=normalize_trace, clip_negative=clip_negative
    )
    with torch.no_grad():
        node.weights.copy_(_nonsymmetric_matrix())
    node.eval()
    example = _cube(seed=16) * 2.0 - 1.0
    program = torch.export.export(_ExportProbe(node), (example,))

    other = _cube(seed=17) * 2.0 - 1.0
    traced = program.module()(other)
    expected = node.forward(cube=other)["cube"]
    torch.testing.assert_close(traced, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("normalize_trace", [False, True])
@pytest.mark.parametrize("clip_negative", [False, True])
def test_dynamo_export_matches_onnx_runtime(normalize_trace: bool, clip_negative: bool) -> None:
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    node = BandCrosstalkCorrector(
        input_channels=C, normalize_trace=normalize_trace, clip_negative=clip_negative
    )
    with torch.no_grad():
        node.weights.copy_(_nonsymmetric_matrix())
    node.eval()
    probe = _ExportProbe(node).eval()
    cube = _cube(seed=18) * 2.0 - 1.0
    with torch.no_grad():
        eager = probe(cube)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "crosstalk.onnx"
        torch.onnx.export(probe, (cube,), str(path), dynamo=True, opset_version=18)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (got,) = session.run(None, {session.get_inputs()[0].name: cube.numpy()})
    np.testing.assert_allclose(got, eager.numpy(), atol=1e-5, rtol=1e-4)
