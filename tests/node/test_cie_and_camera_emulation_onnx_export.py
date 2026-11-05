"""Characterization and export-parity tests for CIETristimulusRGBSelector and
CameraEmulationFalseRGBSelector (ALL-6819).

Both selectors build a wavelength-dependent weight matrix with numpy (CIE CMF
interpolation, Gaussian sensitivity curves) and cache it by the exact wavelength
values. torch.export cannot trace that lookup, but a deployed sensor's wavelength
grid never changes, so under compilation the cached weights built by an earlier
eager call are reused instead of being recomputed. These tests pin the eager
weight-building behaviour and check that an already-primed node exports with
exact parity against ONNX Runtime.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from cuvis_ai.node.channel_selector import (
    CameraEmulationFalseRGBSelector,
    CIETristimulusRGBSelector,
)

pytestmark = pytest.mark.unit

B, H, W, C = 1, 16, 16, 5
WAVELENGTHS = np.array([500.0, 600.0, 700.0, 800.0, 900.0], dtype=np.float32)


def _cube(seed: int = 0) -> torch.Tensor:
    return torch.rand(B, H, W, C, generator=torch.Generator().manual_seed(seed))


class _ForwardOnlyProbe(torch.nn.Module):
    def __init__(self, node: torch.nn.Module) -> None:
        super().__init__()
        self.node = node

    def forward(self, cube: torch.Tensor, wavelengths: torch.Tensor) -> torch.Tensor:
        return self.node.forward(cube, wavelengths)["rgb_image"]


# ---------------------------------------------------------------------------
# Characterization: an unprimed node still raises under compilation.
# ---------------------------------------------------------------------------


def test_cie_tristimulus_no_cached_weights_raises_under_compiling() -> None:
    node = CIETristimulusRGBSelector(normalize_output=False)
    node.eval()
    with pytest.raises(RuntimeError, match="no cached CMF weights"), _force_compiling():
        node._ensure_cmf_weights(torch.from_numpy(WAVELENGTHS), torch.device("cpu"))


def test_camera_emulation_no_cached_weights_raises_under_compiling() -> None:
    node = CameraEmulationFalseRGBSelector(normalize_output=False)
    node.eval()
    with pytest.raises(RuntimeError, match="no cached channel weights"), _force_compiling():
        node._ensure_channel_weights(torch.from_numpy(WAVELENGTHS), torch.device("cpu"))


class _force_compiling:
    """Force ``torch.compiler.is_compiling()`` to report True for a guard check."""

    def __enter__(self):
        import torch.compiler as tc

        self._orig = tc.is_compiling
        tc.is_compiling = lambda: True
        return self

    def __exit__(self, *exc):
        import torch.compiler as tc

        tc.is_compiling = self._orig


# ---------------------------------------------------------------------------
# Characterization: eager output unchanged (seeded reference).
# ---------------------------------------------------------------------------


def test_cie_tristimulus_rgb_unchanged() -> None:
    torch.manual_seed(0)
    node = CIETristimulusRGBSelector(normalize_output=False)
    node.eval()
    cube = _cube()
    out = node.forward(cube, WAVELENGTHS)
    assert out["rgb_image"].shape == (B, H, W, 3)
    assert torch.isfinite(out["rgb_image"]).all()
    # Calling again with the same wavelengths reuses the cached weights and gives
    # byte-identical output.
    out2 = node.forward(cube, WAVELENGTHS)
    torch.testing.assert_close(out["rgb_image"], out2["rgb_image"])


def test_camera_emulation_rgb_unchanged() -> None:
    torch.manual_seed(0)
    node = CameraEmulationFalseRGBSelector(normalize_output=False)
    node.eval()
    cube = _cube()
    out = node.forward(cube, WAVELENGTHS)
    assert out["rgb_image"].shape == (B, H, W, 3)
    assert torch.isfinite(out["rgb_image"]).all()
    out2 = node.forward(cube, WAVELENGTHS)
    torch.testing.assert_close(out["rgb_image"], out2["rgb_image"])


# ---------------------------------------------------------------------------
# Export parity: a node primed by one eager call exports and matches ONNX Runtime.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cie_tristimulus_primed_exports_and_matches_onnx_runtime() -> None:
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    node = CIETristimulusRGBSelector(normalize_output=False)
    node.eval()
    wavelengths = torch.from_numpy(WAVELENGTHS)
    cube = _cube()
    node.forward(cube, wavelengths)  # prime the cached CMF weights

    probe = _ForwardOnlyProbe(node).eval()
    with torch.no_grad():
        eager = probe(cube, wavelengths)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cie.onnx"
        torch.onnx.export(
            probe,
            (cube, wavelengths),
            str(path),
            dynamo=True,
            opset_version=18,
            input_names=["cube", "wavelengths"],
            output_names=["rgb"],
        )
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (got,) = session.run(None, {"cube": cube.numpy(), "wavelengths": wavelengths.numpy()})
    np.testing.assert_allclose(got, eager.numpy(), atol=1e-5, rtol=1e-4)


@pytest.mark.slow
def test_camera_emulation_primed_exports_and_matches_onnx_runtime() -> None:
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    node = CameraEmulationFalseRGBSelector(normalize_output=False)
    node.eval()
    wavelengths = torch.from_numpy(WAVELENGTHS)
    cube = _cube()
    node.forward(cube, wavelengths)  # prime the cached Gaussian weights

    probe = _ForwardOnlyProbe(node).eval()
    with torch.no_grad():
        eager = probe(cube, wavelengths)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "camera_emulation.onnx"
        torch.onnx.export(
            probe,
            (cube, wavelengths),
            str(path),
            dynamo=True,
            opset_version=18,
            input_names=["cube", "wavelengths"],
            output_names=["rgb"],
        )
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (got,) = session.run(None, {"cube": cube.numpy(), "wavelengths": wavelengths.numpy()})
    np.testing.assert_allclose(got, eager.numpy(), atol=1e-5, rtol=1e-4)
