from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import savgol_filter

from cuvis_ai.node.pretreatments import SavitzkyGolay

pytestmark = pytest.mark.unit


@torch.no_grad()
@pytest.mark.parametrize("deriv", [0, 1, 2])
def test_savgol_matches_scipy_on_interior(deriv: int) -> None:
    rng = np.random.default_rng(0)
    window_length, polyorder, delta = 11, 2, 1.0
    x = rng.random((1, 2, 3, 21)).astype(np.float64)
    wavelengths = np.arange(21, dtype=np.int32)  # uniform 1 nm spacing

    ref = savgol_filter(
        x, window_length, polyorder, deriv=deriv, delta=delta, axis=-1, mode="nearest"
    )

    node = SavitzkyGolay(
        window_length=window_length, polyorder=polyorder, deriv=deriv, delta=delta, mode="nearest"
    )
    got = node.forward(cube=torch.tensor(x, dtype=torch.float32), wavelengths=wavelengths)["cube"]

    pad = window_length // 2
    interior = slice(pad, 21 - pad)
    assert torch.allclose(
        got[..., interior],
        torch.tensor(ref[..., interior], dtype=torch.float32),
        atol=1e-5,
    )


@torch.no_grad()
@pytest.mark.parametrize("mode", ["nearest", "mirror", "constant"])
@pytest.mark.parametrize("deriv", [0, 1])
def test_savgol_derivative_uses_wavelength_spacing(mode: str, deriv: int) -> None:
    # The node is built with the default delta=1.0, but a connected wavelengths
    # port with 4 nm uniform spacing must drive the derivative magnitude, so the
    # result matches scipy with delta=4.0.
    rng = np.random.default_rng(1)
    window_length, polyorder, spacing = 7, 2, 4
    x = rng.random((1, 2, 3, 21)).astype(np.float64)
    wavelengths = np.arange(0, 21 * spacing, spacing, dtype=np.int32)

    ref = savgol_filter(
        x, window_length, polyorder, deriv=deriv, delta=float(spacing), axis=-1, mode=mode, cval=0.0
    )

    node = SavitzkyGolay(window_length=window_length, polyorder=polyorder, deriv=deriv, mode=mode)
    got = node.forward(cube=torch.tensor(x, dtype=torch.float32), wavelengths=wavelengths)["cube"]

    assert torch.allclose(got, torch.tensor(ref, dtype=torch.float32), atol=1e-5)


@torch.no_grad()
def test_savgol_derivative_falls_back_to_delta_without_wavelengths() -> None:
    # With no wavelengths port the derivative uses the delta parameter.
    rng = np.random.default_rng(2)
    window_length, polyorder, deriv, delta = 7, 2, 1, 4.0
    x = rng.random((1, 2, 3, 21)).astype(np.float64)

    ref = savgol_filter(
        x, window_length, polyorder, deriv=deriv, delta=delta, axis=-1, mode="nearest"
    )

    node = SavitzkyGolay(
        window_length=window_length, polyorder=polyorder, deriv=deriv, delta=delta, mode="nearest"
    )
    got = node.forward(cube=torch.tensor(x, dtype=torch.float32))["cube"]

    assert torch.allclose(got, torch.tensor(ref, dtype=torch.float32), atol=1e-5)


@torch.no_grad()
def test_savgol_warns_on_non_uniform_spacing() -> None:
    from loguru import logger

    wavelengths = np.array([400, 405, 412, 420, 431, 445, 462, 482, 505, 531], dtype=np.int32)
    cube = torch.rand(1, 2, 3, wavelengths.size)

    warnings_seen: list[str] = []
    sink_id = logger.add(lambda m: warnings_seen.append(m.record["message"]), level="WARNING")
    try:
        node = SavitzkyGolay(window_length=7, polyorder=2, deriv=1)
        node.forward(cube=cube, wavelengths=wavelengths)
    finally:
        logger.remove(sink_id)

    assert any("non-uniform" in w for w in warnings_seen)


@torch.no_grad()
def test_savgol_smoothing_ignores_non_uniform_spacing() -> None:
    from loguru import logger

    wavelengths = np.array([400, 405, 412, 420, 431, 445, 462, 482, 505, 531], dtype=np.int32)
    cube = torch.rand(1, 2, 3, wavelengths.size)

    warnings_seen: list[str] = []
    sink_id = logger.add(lambda m: warnings_seen.append(m.record["message"]), level="WARNING")
    try:
        node = SavitzkyGolay(window_length=7, polyorder=2, deriv=0)  # smoothing
        node.forward(cube=cube, wavelengths=wavelengths)
    finally:
        logger.remove(sink_id)

    assert not any("non-uniform" in w for w in warnings_seen)


def test_savgol_rejects_even_window_length() -> None:
    with pytest.raises(ValueError):
        SavitzkyGolay(window_length=10)


def test_savgol_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        SavitzkyGolay(mode="wrap")


@torch.no_grad()
def test_savgol_preserves_shape() -> None:
    node = SavitzkyGolay()
    cube = torch.rand(2, 4, 5, 21)
    out = node.forward(cube=cube, wavelengths=np.arange(21, dtype=np.int32))["cube"]
    assert out.shape == cube.shape


@torch.no_grad()
def test_savgol_spacing_rescale_degenerate_wavelengths_are_noop() -> None:
    # Fewer than two bands or all-equal wavelengths give no usable spacing, so the
    # rescale factor falls back to 1.0 (no change to the derivative).
    node = SavitzkyGolay(window_length=7, polyorder=2, deriv=1)
    cube = torch.rand(1, 1, 1, 7)
    assert node._spacing_rescale(np.array([500], dtype=np.int32), cube) == 1.0
    assert node._spacing_rescale(np.array([500, 500, 500], dtype=np.int32), cube) == 1.0
