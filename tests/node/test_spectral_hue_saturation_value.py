"""Tests for SpectralHueSaturationValue: independent numpy float64 reference per
index, both hue modes, invariance properties, degenerate inputs, gradients, export
parity and pipeline reload."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage

from cuvis_ai.node.spectral_hue import SpectralHueSaturationValue
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from tests.fixtures.mock_nodes import SeededTensorSource

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Independent numpy float64 reference (loops, not the node's vectorised form).
# ---------------------------------------------------------------------------


def _cartesian_coeffs_loop(x: np.ndarray) -> np.ndarray:
    """``c_i`` for ``i = 1..C-1`` from the raw-sum formula, one index at a time."""
    c = len(x)
    out = np.zeros(c - 1, dtype=np.float64)
    for i in range(1, c):
        n_i = c - i + 1
        suffix_sum = sum(x[j - 1] for j in range(i + 1, c + 1))
        out[i - 1] = np.sqrt((n_i - 1) / n_i) * x[i - 1] - suffix_sum / np.sqrt((n_i - 1) * n_i)
    return out


def _atan2_zero_y(y: float, x: float) -> float:
    """``atan2`` with the axis convention: at y == 0, 0 for x >= 0 and pi for x < 0."""
    if y == 0.0:
        return 0.0 if x >= 0.0 else float(np.pi)
    return float(np.arctan2(y, x))


def _polar_angles_loop(coeffs: np.ndarray) -> np.ndarray:
    """``phi_k`` for ``k = 1..C-2`` (``C-2`` angles total), one index at a time."""
    m = len(coeffs)  # m = C - 1
    angles = np.zeros(m - 1, dtype=np.float64)
    for k in range(1, m - 1):  # k = 1..C-3
        tail = coeffs[k:]
        tail_norm = float(np.sqrt(np.sum(tail**2)))
        angles[k - 1] = _atan2_zero_y(tail_norm, coeffs[k - 1])
    angles[m - 2] = _atan2_zero_y(coeffs[m - 1], coeffs[m - 2])
    return angles


def _saturation_value_loop(x: np.ndarray) -> tuple[float, float]:
    return float(np.max(x) - np.min(x)), float(np.mean(x))


def _reference_for_spectrum(x: np.ndarray, eps: float) -> dict[str, np.ndarray]:
    """Cartesian hue, polar angles, saturation and value for one spectrum, numpy float64."""
    coeffs = _cartesian_coeffs_loop(x)
    norm = max(float(np.sqrt(np.sum(coeffs**2))), eps)
    cartesian = coeffs / norm
    polar = _polar_angles_loop(coeffs)
    saturation, value = _saturation_value_loop(x)
    return {
        "cartesian": cartesian,
        "polar": polar,
        "saturation": saturation,
        "value": value,
    }


def _random_cube(
    batches: int, height: int, width: int, channels: int, seed: int = 0
) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    return torch.randn((batches, height, width, channels), generator=rng, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 1. Golden reference against the independent numpy loop, per pixel.
# ---------------------------------------------------------------------------


def test_cartesian_matches_float64_loop_reference_per_pixel() -> None:
    """Every pixel of a 61-band float64 cube matches the independently written loop."""
    c = 61
    rng = np.random.default_rng(0)
    cube = rng.standard_normal((2, 3, 4, c))
    node = SpectralHueSaturationValue(hue="cartesian", eps=1e-8)
    out = node.forward(cube=torch.from_numpy(cube))
    for b in range(2):
        for h in range(3):
            for w in range(4):
                expected = _reference_for_spectrum(cube[b, h, w], eps=1e-8)
                np.testing.assert_allclose(
                    out["hue"][b, h, w].numpy(), expected["cartesian"], atol=1e-12
                )
                assert out["saturation"][b, h, w, 0].item() == pytest.approx(
                    expected["saturation"], abs=1e-12
                )
                assert out["value"][b, h, w, 0].item() == pytest.approx(
                    expected["value"], abs=1e-12
                )


def test_polar_matches_float64_loop_reference_per_pixel() -> None:
    c = 61
    rng = np.random.default_rng(1)
    cube = rng.standard_normal((2, 3, 4, c))
    node = SpectralHueSaturationValue(hue="polar", eps=1e-8)
    out = node.forward(cube=torch.from_numpy(cube))
    for b in range(2):
        for h in range(3):
            for w in range(4):
                expected = _reference_for_spectrum(cube[b, h, w], eps=1e-8)
                np.testing.assert_allclose(
                    out["hue"][b, h, w].numpy(), expected["polar"], atol=1e-10
                )


# ---------------------------------------------------------------------------
# 2. Equality with U^T * PerPixelUnitNorm(x) for a Helmert matrix built in the test.
# ---------------------------------------------------------------------------


def _helmert_matrix(c: int) -> np.ndarray:
    """``C x (C-1)`` orthonormal Helmert basis matrix ``U`` of the mean-zero hyperplane."""
    u = np.zeros((c, c - 1), dtype=np.float64)
    for idx in range(1, c):
        i = idx
        n_i = c - i + 1
        u[i - 1, idx - 1] = np.sqrt((n_i - 1) / n_i)
        for j in range(i + 1, c + 1):
            u[j - 1, idx - 1] = -1.0 / np.sqrt((n_i - 1) * n_i)
    return u


def test_cartesian_equals_helmert_transpose_times_unit_norm() -> None:
    """hue == U^T @ PerPixelUnitNorm(x) for the Helmert matrix built independently here."""
    c = 16
    rng = np.random.default_rng(2)
    x = rng.standard_normal(c)
    u = _helmert_matrix(c)

    centered = x - x.mean()
    unit = centered / max(float(np.linalg.norm(centered)), 1e-8)
    expected = u.T @ unit

    node = SpectralHueSaturationValue(hue="cartesian", eps=1e-8)
    cube = torch.from_numpy(x.reshape(1, 1, 1, c))
    out = node.forward(cube=cube)["hue"][0, 0, 0].numpy()
    np.testing.assert_allclose(out, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. Invariance: hue unchanged under positive gain and constant offset, both modes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hue_mode", ["cartesian", "polar"])
def test_hue_unchanged_under_positive_gain_and_offset(hue_mode: str) -> None:
    x = _random_cube(2, 3, 4, 10, seed=5)
    node = SpectralHueSaturationValue(hue=hue_mode, eps=1e-8)
    out = node.forward(cube=x)["hue"]
    transformed = x * 2.5 + 3.0
    out_transformed = node.forward(cube=transformed)["hue"]
    torch.testing.assert_close(out_transformed, out, atol=1e-4, rtol=1e-4)


def test_saturation_scales_with_gain_and_ignores_offset() -> None:
    x = _random_cube(2, 3, 4, 10, seed=6)
    node = SpectralHueSaturationValue(hue="cartesian", eps=1e-8)
    base = node.forward(cube=x)["saturation"]
    scaled = node.forward(cube=x * 2.5 + 3.0)["saturation"]
    torch.testing.assert_close(scaled, base * 2.5, atol=1e-4, rtol=1e-4)


def test_value_follows_gain_and_offset() -> None:
    x = _random_cube(2, 3, 4, 10, seed=7)
    node = SpectralHueSaturationValue(hue="cartesian", eps=1e-8)
    base = node.forward(cube=x)["value"]
    scaled = node.forward(cube=x * 2.5 + 3.0)["value"]
    torch.testing.assert_close(scaled, base * 2.5 + 3.0, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 4. Polar angle ranges and reconstruction of the Cartesian unit vector.
# ---------------------------------------------------------------------------


def test_polar_angles_in_their_ranges() -> None:
    c = 9
    rng = np.random.default_rng(8)
    cube = torch.from_numpy(rng.standard_normal((1, 1, 1, c)))
    node = SpectralHueSaturationValue(hue="polar", eps=1e-8)
    angles = node.forward(cube=cube)["hue"][0, 0, 0]
    assert angles.shape == (c - 2,)
    # phi_1 .. phi_{C-3}: [0, pi]
    for k in range(c - 3):
        assert 0.0 <= angles[k].item() <= float(np.pi) + 1e-6
    # phi_{C-2}: (-pi, pi]
    assert -float(np.pi) < angles[-1].item() <= float(np.pi) + 1e-6


def test_polar_angles_rebuild_cartesian_unit_vector() -> None:
    c = 11
    rng = np.random.default_rng(9)
    x = rng.standard_normal(c)
    cartesian_node = SpectralHueSaturationValue(hue="cartesian", eps=1e-8)
    polar_node = SpectralHueSaturationValue(hue="polar", eps=1e-8)
    cube = torch.from_numpy(x.reshape(1, 1, 1, c))
    unit = cartesian_node.forward(cube=cube)["hue"][0, 0, 0].numpy()
    angles = polar_node.forward(cube=cube)["hue"][0, 0, 0].numpy()

    m = len(unit)  # m = C - 1
    recon = np.zeros(m, dtype=np.float64)
    sin_prod = 1.0
    for idx in range(m - 2):
        recon[idx] = sin_prod * np.cos(angles[idx])
        sin_prod *= np.sin(angles[idx])
    recon[-2] = sin_prod * np.cos(angles[-1])
    recon[-1] = sin_prod * np.sin(angles[-1])
    np.testing.assert_allclose(recon, unit, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Flat and all-zero spectra; float32 exactness.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hue_mode", ["cartesian", "polar"])
@pytest.mark.parametrize("level", [0.3, 1.0, 1000.0])
def test_flat_spectrum_gives_exactly_zero_hue_float32(hue_mode: str, level: float) -> None:
    """A float32 flat spectrum at several levels gives exactly zero hue in both modes."""
    c = 61
    x = torch.full((1, 1, 1, c), level, dtype=torch.float32)
    node = SpectralHueSaturationValue(hue=hue_mode, eps=1e-8)
    out = node.forward(cube=x)["hue"]
    assert torch.equal(out, torch.zeros_like(out))


def test_small_signal_on_large_constant_level_keeps_its_direction_float32() -> None:
    """Weighted differences keep a small spectral signal under a large constant level.

    The signal is small relative to the constant level (1e-1 against 1000.0) but
    large enough that its own hue is not itself dominated by float32 noise, so
    the comparison isolates the level's effect rather than noise in the signal.
    """
    c = 61
    rng = np.random.default_rng(10)
    signal = rng.standard_normal(c).astype(np.float32) * 1e-1
    level = np.float32(1000.0)
    x_signal = torch.from_numpy(signal.reshape(1, 1, 1, c))
    x_with_level = torch.from_numpy((signal + level).reshape(1, 1, 1, c))

    node = SpectralHueSaturationValue(hue="cartesian", eps=1e-8)
    hue_signal = node.forward(cube=x_signal)["hue"]
    hue_with_level = node.forward(cube=x_with_level)["hue"]
    torch.testing.assert_close(hue_with_level, hue_signal, atol=1e-3, rtol=1e-3)
    assert torch.any(hue_with_level != 0.0)


def test_all_zero_spectrum_stays_finite() -> None:
    c = 20
    x = torch.zeros((1, 2, 2, c))
    node = SpectralHueSaturationValue(hue="cartesian", eps=1e-8)
    out = node.forward(cube=x)
    assert torch.isfinite(out["hue"]).all()
    assert torch.equal(out["hue"], torch.zeros_like(out["hue"]))
    assert torch.equal(out["saturation"], torch.zeros_like(out["saturation"]))
    assert torch.equal(out["value"], torch.zeros_like(out["value"]))


# ---------------------------------------------------------------------------
# 6. Two channels (minimum for cartesian).
# ---------------------------------------------------------------------------


def test_two_channels_cartesian() -> None:
    x = torch.tensor([[[[3.0, 7.0]]]])  # [1, 1, 1, 2]
    node = SpectralHueSaturationValue(hue="cartesian", eps=1e-8)
    out = node.forward(cube=x)
    assert out["hue"].shape == (1, 1, 1, 1)
    assert torch.isfinite(out["hue"]).all()
    assert abs(out["hue"][0, 0, 0, 0].item()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 7. atan2 axis cases in eager mode.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("coeffs", "expected"),
    [
        # last angle: y is the final coordinate
        ([1.0, 0.0], [0.0]),
        ([-1.0, 0.0], [np.pi]),
        ([-1.0, -0.0], [np.pi]),
        ([0.0, 0.0], [0.0]),
        # leading angle: y is the norm of the coordinates after x
        ([2.0, 0.0, 0.0], [0.0, 0.0]),
        ([-2.0, 0.0, 0.0], [np.pi, 0.0]),
        ([0.0, -3.0, 0.0], [np.pi / 2, np.pi]),
    ],
)
def test_polar_angles_at_axis_points_follow_the_convention(
    coeffs: list[float], expected: list[float]
) -> None:
    """Where y == 0 the node returns 0 for x >= 0 and pi for x < 0, never -pi."""
    coefficients = torch.tensor(coeffs, dtype=torch.float64).reshape(1, 1, 1, -1)
    angles = SpectralHueSaturationValue._polar_angles(coefficients)[0, 0, 0]
    np.testing.assert_allclose(angles.numpy(), np.array(expected), atol=1e-12)
    np.testing.assert_allclose(_polar_angles_loop(np.array(coeffs)), np.array(expected), atol=1e-12)


# ---------------------------------------------------------------------------
# 8. Refusals: too few channels, NaN/infinite/non-positive eps.
# ---------------------------------------------------------------------------


def test_forward_raises_for_fewer_than_two_channels_cartesian() -> None:
    node = SpectralHueSaturationValue(hue="cartesian", eps=1e-8)
    with pytest.raises(ValueError):
        node.forward(cube=torch.randn(1, 1, 1, 1))


@pytest.mark.parametrize("channels", [1, 2])
def test_forward_raises_for_fewer_than_three_channels_polar(channels: int) -> None:
    node = SpectralHueSaturationValue(hue="polar", eps=1e-8)
    with pytest.raises(ValueError):
        node.forward(cube=torch.randn(1, 1, 1, channels))


@pytest.mark.parametrize("eps", [0.0, -1e-8, float("nan"), float("inf")])
def test_non_positive_or_non_finite_eps_raises(eps: float) -> None:
    with pytest.raises(ValueError, match="eps"):
        SpectralHueSaturationValue(hue="cartesian", eps=eps)


def test_invalid_hue_mode_raises() -> None:
    with pytest.raises(ValueError, match="hue"):
        SpectralHueSaturationValue(hue="not-a-mode")


# ---------------------------------------------------------------------------
# 9. Port contract.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("hue_mode", "channels"), [("cartesian", 8), ("polar", 8)])
def test_port_contract_shapes_and_dtypes(hue_mode: str, channels: int) -> None:
    node = SpectralHueSaturationValue(hue=hue_mode, eps=1e-8)
    x = _random_cube(2, 3, 4, channels, seed=11)
    out = node.forward(cube=x)
    expected_hue_channels = channels - 1 if hue_mode == "cartesian" else channels - 2
    assert out["hue"].shape == (2, 3, 4, expected_hue_channels)
    assert out["saturation"].shape == (2, 3, 4, 1)
    assert out["value"].shape == (2, 3, 4, 1)
    for key in ("hue", "saturation", "value"):
        assert out[key].dtype == torch.float32


# ---------------------------------------------------------------------------
# 10. Finite gradients, including through the masks at degenerate inputs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hue_mode", ["cartesian", "polar"])
def test_gradients_are_finite_for_a_generic_spectrum(hue_mode: str) -> None:
    x = _random_cube(1, 2, 2, 10, seed=12).requires_grad_(True)
    node = SpectralHueSaturationValue(hue=hue_mode, eps=1e-8)
    out = node.forward(cube=x)
    loss = out["hue"].sum() + out["saturation"].sum() + out["value"].sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("hue_mode", ["cartesian", "polar"])
def test_gradients_are_finite_for_a_flat_spectrum(hue_mode: str) -> None:
    x = torch.full((1, 1, 1, 10), 2.0, requires_grad=True)
    node = SpectralHueSaturationValue(hue=hue_mode, eps=1e-8)
    out = node.forward(cube=x)
    loss = out["hue"].sum() + out["saturation"].sum() + out["value"].sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("hue_mode", ["cartesian", "polar"])
def test_gradients_are_finite_for_a_spectrum_with_a_zero_tail(hue_mode: str) -> None:
    x = torch.cat([torch.randn(1, 1, 1, 6), torch.zeros(1, 1, 1, 4)], dim=-1).requires_grad_(True)
    node = SpectralHueSaturationValue(hue=hue_mode, eps=1e-8)
    out = node.forward(cube=x)
    loss = out["hue"].sum() + out["saturation"].sum() + out["value"].sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# 11. Export smoke.
# ---------------------------------------------------------------------------


class _ExportProbe(torch.nn.Module):
    """The node, or its export_module() adapter, with a tensor in and tensors out."""

    def __init__(self, node: SpectralHueSaturationValue) -> None:
        super().__init__()
        adapter = node.export_module()
        self.module = adapter if adapter is not None else node

    def forward(self, cube: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.module(cube=cube)
        return out["hue"], out["saturation"], out["value"]


@pytest.mark.parametrize("hue_mode", ["cartesian", "polar"])
def test_export_smoke_matches_forward(hue_mode: str) -> None:
    node = SpectralHueSaturationValue(hue=hue_mode, eps=1e-8).eval()
    example = _random_cube(1, 2, 2, 8, seed=13)
    program = torch.export.export(_ExportProbe(node), (example,))

    other = _random_cube(1, 2, 2, 8, seed=14)
    traced_hue, traced_sat, traced_val = program.module()(other)
    expected = node.forward(cube=other)
    torch.testing.assert_close(traced_hue, expected["hue"], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(traced_sat, expected["saturation"], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(traced_val, expected["value"], atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# 12. Dynamo ONNX plus ONNX Runtime parity against the numpy reference, both modes.
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("hue_mode", ["cartesian", "polar"])
def test_dynamo_export_matches_onnx_runtime_and_numpy_reference(hue_mode: str) -> None:
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    c = 16
    node = SpectralHueSaturationValue(hue=hue_mode, eps=1e-8).eval()
    probe = _ExportProbe(node).eval()
    rng = np.random.default_rng(15)
    cube_np = rng.standard_normal((1, 2, 2, c)).astype(np.float32)
    cube = torch.from_numpy(cube_np)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"spectral_hue_{hue_mode}.onnx"
        torch.onnx.export(probe, (cube,), str(path), dynamo=True, opset_version=18)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got_hue, got_sat, got_val = session.run(None, {session.get_inputs()[0].name: cube_np})

    for b in range(1):
        for h in range(2):
            for w in range(2):
                expected = _reference_for_spectrum(cube_np[b, h, w].astype(np.float64), eps=1e-8)
                key = "cartesian" if hue_mode == "cartesian" else "polar"
                np.testing.assert_allclose(got_hue[b, h, w], expected[key], atol=1e-4, rtol=1e-3)
                assert got_sat[b, h, w, 0] == pytest.approx(expected["saturation"], abs=1e-4)
                assert got_val[b, h, w, 0] == pytest.approx(expected["value"], abs=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize(("x_sign", "expected_last_angle"), [(1.0, 0.0), (-1.0, np.pi)])
def test_dynamo_export_atan2_axis_case_matches_eager(
    x_sign: float, expected_last_angle: float
) -> None:
    """ONNX's atan2(0, x < 0) differs from eager's; the mask keeps both at the convention."""
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    c = 5
    node = SpectralHueSaturationValue(hue="polar", eps=1e-8).eval()
    probe = _ExportProbe(node).eval()
    # The last two channels are equal, so the final coordinate (y of the last angle) is
    # exactly zero; the sign of the leading channels sets the sign of its partner x.
    base = torch.zeros(1, 1, 1, c)
    base[0, 0, 0, :-2] = torch.tensor([1.0, 2.0, 3.0]) * x_sign
    with torch.no_grad():
        eager = node.forward(cube=base)["hue"]
    assert eager[0, 0, 0, -1].item() == pytest.approx(expected_last_angle, abs=1e-6)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spectral_hue_atan2_axis.onnx"
        torch.onnx.export(probe, (base,), str(path), dynamo=True, opset_version=18)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got_hue, _, _ = session.run(None, {session.get_inputs()[0].name: base.numpy()})
    np.testing.assert_allclose(got_hue, eager.numpy(), atol=1e-4, rtol=1e-3)


# Spectra whose first hue coordinate is exactly zero and carries a minus sign (the
# coordinates are a negative scale times a sum that cancels). atan2(y, -0.0) is +-pi/2
# in eager, but its ONNX lowering divides y by x and gets the other sign.
_NEGATIVE_ZERO_SPECTRA = [
    [2.0, 3.0, 1.0],
    [3.0, 2.0, 4.0],
    [1.0, 2.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, -1.0, 0.0],
]


@pytest.mark.parametrize("spectrum", _NEGATIVE_ZERO_SPECTRA)
def test_polar_angles_for_a_negative_zero_coordinate_match_the_reference(
    spectrum: list[float],
) -> None:
    cube = torch.tensor(spectrum, dtype=torch.float64).reshape(1, 1, 1, -1)
    coefficients = SpectralHueSaturationValue._cartesian_coefficients(cube)[0, 0, 0]
    assert coefficients[0].item() == 0.0
    assert bool(torch.signbit(coefficients[0]))
    angles = SpectralHueSaturationValue(hue="polar")(cube=cube)["hue"][0, 0, 0].numpy()
    expected = _reference_for_spectrum(np.array(spectrum), eps=1e-8)["polar"]
    np.testing.assert_allclose(angles, expected, atol=1e-12)
    assert abs(angles[0]) == pytest.approx(np.pi / 2, abs=1e-12)


@pytest.mark.parametrize("zero", [0.0, -0.0])
@pytest.mark.parametrize("y_value", [2.0, -4.0])
def test_gradient_at_a_zero_coordinate_is_that_of_atan2(zero: float, y_value: float) -> None:
    """d/dx atan2(y, x) = -y / (x^2 + y^2) and d/dy = x / (x^2 + y^2), at x = +-0."""
    x = torch.tensor([zero], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([y_value], dtype=torch.float64, requires_grad=True)
    angle = SpectralHueSaturationValue._safe_atan2(y, x)
    angle.sum().backward()
    assert angle.item() == pytest.approx(np.sign(y_value) * np.pi / 2, abs=1e-12)
    assert x.grad.item() == pytest.approx(-1.0 / y_value, abs=1e-12)
    assert y.grad.item() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("spectrum", _NEGATIVE_ZERO_SPECTRA)
def test_dynamo_export_matches_eager_for_a_negative_zero_coordinate(
    spectrum: list[float],
) -> None:
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    node = SpectralHueSaturationValue(hue="polar", eps=1e-8).eval()
    probe = _ExportProbe(node).eval()
    cube = torch.tensor(spectrum, dtype=torch.float32).reshape(1, 1, 1, -1)
    with torch.no_grad():
        eager = node.forward(cube=cube)["hue"]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spectral_hue_negative_zero.onnx"
        torch.onnx.export(probe, (cube,), str(path), dynamo=True, opset_version=18)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got_hue, _, _ = session.run(None, {session.get_inputs()[0].name: cube.numpy()})
    np.testing.assert_allclose(got_hue, eager.numpy(), atol=1e-5)


# ---------------------------------------------------------------------------
# 13. Pipeline save and reload.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hue_mode", ["cartesian", "polar"])
def test_pipeline_round_trips_through_yaml(tmp_path, hue_mode: str) -> None:
    pipe = CuvisPipeline("spectral_hue_reload")
    src = SeededTensorSource(name="src", shape=[2, 3, 4, 7], seed=42)
    node = SpectralHueSaturationValue(name="hue_node", hue=hue_mode, eps=1e-8)
    pipe.connect(src.outputs.data, node.inputs.cube)
    before = pipe.forward(batch={}, stage=ExecutionStage.INFERENCE)[("hue_node", "hue")]

    path = tmp_path / "spectral_hue.yaml"
    pipe.save_to_file(str(path))
    rebuilt = PipelineBuilder().build_from_config(str(path))
    after = rebuilt.forward(batch={}, stage=ExecutionStage.INFERENCE)[("hue_node", "hue")]

    assert torch.equal(after, before)
