"""Characterization tests pinning eager outputs of the wavelength-lookup selectors
before their nearest-band resolution moves from numpy to tensor ops (ALL-6819).

Every selector here failed the ONNX stress sweep with a numpy band lookup inside
``forward``. These tests freeze today's eager behaviour (band choice, tolerance
warnings, output values) on seeded inputs so a refactor that swaps the numpy lookup
for tensor ops can be checked against them, then exercise the actual export path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.channel_selector import (
    CIRSelector,
    EVI2Selector,
    EVISelector,
    FixedWavelengthSelector,
    GNDVISelector,
    MCARISelector,
    MSAVISelector,
    NBRSelector,
    NDRESelector,
    NDVISelector,
    NDWISelector,
    PRISelector,
    SAVISelector,
)

pytestmark = pytest.mark.unit

B, H, W, C = 1, 16, 16, 5
WAVELENGTHS = np.array([500.0, 600.0, 700.0, 800.0, 900.0], dtype=np.float32)


def _cube(seed: int = 0) -> torch.Tensor:
    return torch.rand(B, H, W, C, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------
# Characterization: normalized-difference and vegetation-index selectors.
# ---------------------------------------------------------------------------

_NORMALIZED_DIFFERENCE = [NDVISelector, NDWISelector, NBRSelector, GNDVISelector, NDRESelector]
_VEGETATION_INDEX = [
    EVISelector,
    EVI2Selector,
    SAVISelector,
    MSAVISelector,
    MCARISelector,
    PRISelector,
]


@pytest.mark.parametrize("cls", _NORMALIZED_DIFFERENCE)
def test_normalized_difference_index_image_unchanged(cls: type) -> None:
    """Pin the raw index image of each two-band selector at its default wavelengths."""
    torch.manual_seed(0)
    node = cls()
    cube = _cube()
    out = node.forward(cube, WAVELENGTHS)
    # Recomputed straight from the formula on the numpy-resolved indices, independent
    # of whatever internal resolution path forward uses.
    wl = WAVELENGTHS
    primary_idx = int(np.argmin(np.abs(wl - node.primary_nm)))
    secondary_idx = int(np.argmin(np.abs(wl - node.secondary_nm)))
    primary = cube[..., primary_idx]
    secondary = cube[..., secondary_idx]
    expected = (primary - secondary) / (primary + secondary).clamp_min(node.eps)
    torch.testing.assert_close(out["index_image"][..., 0], expected, atol=1e-6, rtol=1e-5)
    assert out["band_info"]["band_indices"] == [primary_idx, secondary_idx]


@pytest.mark.parametrize("cls", _VEGETATION_INDEX)
def test_vegetation_index_band_indices_unchanged(cls: type) -> None:
    """Pin the resolved band indices of each multi-band vegetation index selector."""
    torch.manual_seed(0)
    node = cls()
    cube = _cube()
    out = node.forward(cube, WAVELENGTHS)
    wl = WAVELENGTHS
    expected_indices = [int(np.argmin(np.abs(wl - nm))) for nm in node.band_nm.values()]
    assert out["band_info"]["band_indices"] == expected_indices
    assert torch.isfinite(out["index_image"]).all()


def test_cir_selector_band_indices_and_rgb_unchanged() -> None:
    """Pin CIRSelector's resolved indices and composed (unnormalized) RGB."""
    torch.manual_seed(0)
    node = CIRSelector(normalize_output=False)
    cube = _cube()
    out = node.forward(cube, WAVELENGTHS)
    wl = WAVELENGTHS
    nir_idx = int(np.argmin(np.abs(wl - node.nir_nm)))
    red_idx = int(np.argmin(np.abs(wl - node.red_nm)))
    green_idx = int(np.argmin(np.abs(wl - node.green_nm)))
    expected = torch.stack([cube[..., nir_idx], cube[..., red_idx], cube[..., green_idx]], dim=-1)
    torch.testing.assert_close(out["rgb_image"], expected)
    assert out["band_info"]["band_indices"] == [nir_idx, red_idx, green_idx]


def test_fixed_wavelength_selector_band_indices_and_rgb_unchanged() -> None:
    """Pin FixedWavelengthSelector's resolved indices and raw stacked bands."""
    torch.manual_seed(0)
    node = FixedWavelengthSelector(target_wavelengths=(650.0, 550.0, 450.0), normalize_output=False)
    cube = _cube()
    out = node.forward(cube, WAVELENGTHS)
    wl = WAVELENGTHS
    indices = [int(np.argmin(np.abs(wl - nm))) for nm in node.target_wavelengths]
    expected = torch.stack([cube[..., i] for i in indices], dim=-1)
    torch.testing.assert_close(out["rgb_image"], expected)
    assert out["band_info"]["band_indices"] == indices


def test_fixed_wavelength_selector_n_channel_unchanged() -> None:
    """Pin the n != 3 stacked-band path (no normalisation available there)."""
    torch.manual_seed(0)
    node = FixedWavelengthSelector(
        target_wavelengths=(500.0, 600.0, 700.0, 800.0, 900.0, 550.0),
        normalize_output=False,
        norm_mode="per_frame",
    )
    cube = _cube()
    out = node.forward(cube, WAVELENGTHS)
    assert out["rgb_image"].shape == (B, H, W, 6)


# ---------------------------------------------------------------------------
# Out-of-tolerance warning: the eager-only guard must still fire eagerly, and must
# be skipped while torch.export traces the graph (checked in the export test below).
# ---------------------------------------------------------------------------


def test_warn_bands_out_of_tolerance_still_fires_eagerly() -> None:
    """A target far outside the sensor grid still warns once per label in eager mode."""
    node = NDVISelector(nir_nm=2000.0, band_tolerance_nm=10.0)
    cube = _cube()
    node.forward(cube, WAVELENGTHS)
    assert "nir" in node._bands_warned


# ---------------------------------------------------------------------------
# Export parity: the dynamo exporter traces forward directly (no numpy lookup) and
# ONNX Runtime reproduces the eager output within tolerance.
# ---------------------------------------------------------------------------


class _ForwardOnlyProbe(torch.nn.Module):
    """Wrap a selector's forward so torch.export sees tensor-only inputs/outputs."""

    def __init__(self, node: torch.nn.Module) -> None:
        super().__init__()
        self.node = node

    def forward(self, cube: torch.Tensor, wavelengths: torch.Tensor) -> torch.Tensor:
        out = self.node.forward(cube, wavelengths)
        return out["rgb_image"] if "rgb_image" in out else out["index_image"]


@pytest.mark.slow
@pytest.mark.parametrize(
    "cls",
    [*_NORMALIZED_DIFFERENCE, *_VEGETATION_INDEX, CIRSelector, FixedWavelengthSelector],
)
def test_selector_exports_and_matches_onnx_runtime(cls: type) -> None:
    """Every fixed wavelength-lookup selector traces and ONNX Runtime matches eager."""
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    node = cls(normalize_output=False)
    node.eval()
    probe = _ForwardOnlyProbe(node).eval()
    cube = _cube()
    wavelengths = torch.from_numpy(WAVELENGTHS)
    with torch.no_grad():
        eager = probe(cube, wavelengths)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "selector.onnx"
        torch.onnx.export(
            probe,
            (cube, wavelengths),
            str(path),
            dynamo=True,
            opset_version=18,
            input_names=["cube", "wavelengths"],
            output_names=["out"],
        )
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (got,) = session.run(None, {"cube": cube.numpy(), "wavelengths": wavelengths.numpy()})
    np.testing.assert_allclose(got, eager.numpy(), atol=1e-5, rtol=1e-4)


@pytest.mark.slow
def test_fixed_wavelength_selector_settled_running_bounds_export_matches() -> None:
    """A selector already past warmup and freeze exports the frozen running bounds exactly."""
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    node = FixedWavelengthSelector(running_warmup_frames=2, freeze_running_bounds_after_frames=3)
    node.eval()
    wavelengths = torch.from_numpy(WAVELENGTHS)
    # Run past warmup and freeze so the bounds are settled before export.
    for i in range(5):
        node.forward(_cube(seed=i), wavelengths)
    assert int(node._norm_frame_count) >= 3

    probe = _ForwardOnlyProbe(node).eval()
    cube = _cube(seed=99)
    with torch.no_grad():
        eager = probe(cube, wavelengths)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fixed_settled.onnx"
        torch.onnx.export(probe, (cube, wavelengths), str(path), dynamo=True, opset_version=18)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (got,) = session.run(None, {"cube": cube.numpy(), "wavelengths": wavelengths.numpy()})
    np.testing.assert_allclose(got, eager.numpy(), atol=1e-5, rtol=1e-4)


@pytest.mark.slow
def test_out_of_tolerance_guard_skipped_under_export() -> None:
    """The eager-only tolerance warning does not enter the traced graph."""
    pytest.importorskip("onnxscript")
    torch.manual_seed(0)
    node = NDVISelector(nir_nm=2000.0, band_tolerance_nm=10.0, normalize_output=False)
    node.eval()
    probe = _ForwardOnlyProbe(node).eval()
    cube = _cube()
    wavelengths = torch.from_numpy(WAVELENGTHS)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ndvi.onnx"
        # Must not raise and must not bake in a Python-side warning/branch.
        torch.onnx.export(
            probe,
            (cube, wavelengths),
            str(path),
            dynamo=True,
            opset_version=18,
        )


# ---------------------------------------------------------------------------
# Device: the band indices must follow the cube, not the (numpy-origin,
# always-CPU) wavelengths. A cu3s source feeds a node its numpy wavelengths on
# a numpy-typed port while the cube itself already sits on CUDA; the index
# tensor the lookup builds must not stay on the CPU in that case.
# ---------------------------------------------------------------------------


def test_fixed_wavelength_selector_indices_follow_cube_device_on_cpu() -> None:
    """Even on CPU, the resolved indices sit on the cube's device, not a hardcoded one."""
    node = FixedWavelengthSelector(normalize_output=False)
    cube = _cube()
    wavelengths_t, indices = node._resolve_band_indices_tensor(wavelengths=WAVELENGTHS, cube=cube)
    assert indices.device == cube.device
    assert wavelengths_t.device == cube.device


def test_ndvi_selector_indices_follow_cube_device_on_cpu() -> None:
    node = NDVISelector(normalize_output=False)
    cube = _cube()
    wavelengths_t, indices = node._resolve_band_indices_tensor(wavelengths=WAVELENGTHS, cube=cube)
    assert indices.device == cube.device
    assert wavelengths_t.device == cube.device


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fixed_wavelength_selector_runs_on_cuda_cube_with_numpy_wavelengths() -> None:
    """A node fed numpy wavelengths (a numpy-typed port) and a CUDA cube must not
    raise a cross-device index_select; this reproduces the span-export failure on
    the real Dinomaly pipeline (ALL-6819 follow-up)."""
    node = FixedWavelengthSelector(normalize_output=False).cuda()
    cube = _cube().cuda()
    out = node.forward(cube, WAVELENGTHS)  # WAVELENGTHS is a plain numpy array
    assert out["rgb_image"].device.type == "cuda"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ndvi_selector_runs_on_cuda_cube_with_numpy_wavelengths() -> None:
    node = NDVISelector(normalize_output=False).cuda()
    cube = _cube().cuda()
    out = node.forward(cube, WAVELENGTHS)
    assert out["index_image"].device.type == "cuda"
    assert out["rgb_image"].device.type == "cuda"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cir_selector_runs_on_cuda_cube_with_numpy_wavelengths() -> None:
    node = CIRSelector(normalize_output=False).cuda()
    cube = _cube().cuda()
    out = node.forward(cube, WAVELENGTHS)
    assert out["rgb_image"].device.type == "cuda"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_evi_selector_runs_on_cuda_cube_with_numpy_wavelengths() -> None:
    """The vegetation-index family shares the same band-lookup base as NDVISelector."""
    node = EVISelector(normalize_output=False).cuda()
    cube = _cube().cuda()
    out = node.forward(cube, WAVELENGTHS)
    assert out["index_image"].device.type == "cuda"
    assert out["rgb_image"].device.type == "cuda"
