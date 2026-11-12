"""Characterization and export-parity tests for MinMaxNormalizer and PercentileNormalizer
(ALL-6819). Both failed the ONNX stress sweep on a data-dependent fitted-state guard
(``.item()`` on MinMaxNormalizer, a NaN-bounds check on PercentileNormalizer) that only
exists to raise before an unfitted node runs; this pins the raise still firing eagerly and
checks the fitted (frozen) path exports with exact parity against ONNX Runtime.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from cuvis_ai.node.normalization import MinMaxNormalizer, PercentileNormalizer, PerPixelUnitNorm

pytestmark = pytest.mark.unit

B, H, W, C = 1, 16, 16, 5


def _data(seed: int = 0) -> torch.Tensor:
    return torch.rand(B, H, W, C, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------
# Characterization: the eager guard still raises on an unfitted node.
# ---------------------------------------------------------------------------


def test_minmax_normalizer_raises_when_unfitted() -> None:
    node = MinMaxNormalizer(use_running_stats=True)
    node.eval()
    with pytest.raises(RuntimeError, match="requires statistical_initialization"):
        node.forward(_data())


def test_percentile_normalizer_raises_when_unfitted_statistical() -> None:
    node = PercentileNormalizer(n_channels=C, norm_mode="statistical")
    node.eval()
    with pytest.raises(RuntimeError, match="requires statistical_initialization"):
        node.forward(_data())


def test_minmax_normalizer_fitted_output_unchanged() -> None:
    """Pin the fitted-state min-max scaling formula on seeded data."""
    node = MinMaxNormalizer(use_running_stats=True)
    node.eval()
    node.statistical_initialization(iter([{"data": _data(seed=1)}]))
    data = _data(seed=2)
    out = node.forward(data)["normalized"]
    ranges = torch.clamp(node.running_max - node.running_min, min=node.eps)
    expected = (data - node.running_min) / ranges
    torch.testing.assert_close(out, expected)


def test_percentile_normalizer_fitted_statistical_output_unchanged() -> None:
    """Pin the fitted statistical-mode scaling formula on seeded data."""
    node = PercentileNormalizer(n_channels=C, norm_mode="statistical")
    node.eval()
    node.statistical_initialization(iter([{"data": _data(seed=1)}]))
    data = _data(seed=2)
    out = node.forward(data)["normalized"]
    channels = data.shape[-1]
    lo = node.running_min.reshape(1, 1, 1, channels)
    hi = node.running_max.reshape(1, 1, 1, channels)
    expected = ((data - lo) / (hi - lo).clamp_min(node.eps)).clamp_(0.0, 1.0)
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# Export parity.
# ---------------------------------------------------------------------------


class _ForwardOnlyProbe(torch.nn.Module):
    def __init__(self, node: torch.nn.Module) -> None:
        super().__init__()
        self.node = node

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.node.forward(data)["normalized"]


@pytest.mark.slow
def test_minmax_normalizer_fitted_exports_and_matches_onnx_runtime() -> None:
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    node = MinMaxNormalizer(use_running_stats=True)
    node.eval()
    node.statistical_initialization(iter([{"data": _data(seed=1)}]))
    probe = _ForwardOnlyProbe(node).eval()
    data = _data(seed=2)
    with torch.no_grad():
        eager = probe(data)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "minmax.onnx"
        torch.onnx.export(probe, (data,), str(path), dynamo=True, opset_version=18)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (got,) = session.run(None, {session.get_inputs()[0].name: data.numpy()})
    np.testing.assert_allclose(got, eager.numpy(), atol=1e-5, rtol=1e-4)


@pytest.mark.slow
def test_percentile_normalizer_statistical_exports_and_matches_onnx_runtime() -> None:
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    node = PercentileNormalizer(n_channels=C, norm_mode="statistical")
    node.eval()
    node.statistical_initialization(iter([{"data": _data(seed=1)}]))
    probe = _ForwardOnlyProbe(node).eval()
    data = _data(seed=2)
    with torch.no_grad():
        eager = probe(data)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "percentile.onnx"
        torch.onnx.export(probe, (data,), str(path), dynamo=True, opset_version=18)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (got,) = session.run(None, {session.get_inputs()[0].name: data.numpy()})
    np.testing.assert_allclose(got, eager.numpy(), atol=1e-5, rtol=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize("center", [True, False])
def test_per_pixel_unit_norm_exports_and_matches_onnx_runtime(center: bool) -> None:
    """Dynamo ONNX export plus ONNX Runtime parity, for both center modes."""
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    node = PerPixelUnitNorm(eps=1e-8, center=center)
    node.eval()
    probe = _ForwardOnlyProbe(node).eval()
    data = _data(seed=3)
    with torch.no_grad():
        eager = probe(data)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"per_pixel_unit_norm_center_{center}.onnx"
        torch.onnx.export(probe, (data,), str(path), dynamo=True, opset_version=18)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (got,) = session.run(None, {session.get_inputs()[0].name: data.numpy()})
    np.testing.assert_allclose(got, eager.numpy(), atol=1e-5, rtol=1e-4)


@pytest.mark.slow
def test_percentile_normalizer_settled_running_mode_exports_and_matches() -> None:
    """RUNNING mode, once past warmup and freeze, traces the frozen bounds exactly."""
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    node = PercentileNormalizer(
        n_channels=C,
        norm_mode="running",
        running_warmup_frames=2,
        freeze_running_bounds_after_frames=3,
    )
    node.eval()
    for i in range(5):
        node.forward(_data(seed=i))
    assert int(node._norm_frame_count) >= 3

    probe = _ForwardOnlyProbe(node).eval()
    data = _data(seed=99)
    with torch.no_grad():
        eager = probe(data)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "percentile_running.onnx"
        torch.onnx.export(probe, (data,), str(path), dynamo=True, opset_version=18)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (got,) = session.run(None, {session.get_inputs()[0].name: data.numpy()})
    np.testing.assert_allclose(got, eager.numpy(), atol=1e-5, rtol=1e-4)
