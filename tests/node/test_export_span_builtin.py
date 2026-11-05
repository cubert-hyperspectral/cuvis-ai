"""A toy pipeline of builtin nodes (MinMaxNormalizer, FixedWavelengthSelector) exported
through cuvis_ai_core.pipeline.export.export_deployment with the wavelengths baked in as
a constant (ALL-6819). Checks the whole export path end to end: parity against ONNX
Runtime inside the exporter's own check, and the deploy folder loading back through
ExportedSpan and reproducing the canonical pipeline's output.

Fast: a 4x4 cube, a handful of frames. Marked unit, not slow.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.node.channel_selector import FixedWavelengthSelector
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

pytestmark = pytest.mark.unit
export = pytest.importorskip("cuvis_ai_core.pipeline.export")
pytest.importorskip("onnxruntime")

WAVELENGTHS = [450.0, 500.0, 550.0, 600.0, 650.0]
_CUBE = PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))
_WAVELENGTHS_SPEC = PortSpec(dtype=np.int32, shape=(-1,))


class _Source(Node):
    """Feeds a fixed cube and the wavelength grid, like a cu3s loader (numpy wavelengths).

    Reads the cube from a ``.pt`` file by path (an hparam-safe string) instead of holding
    the tensor itself as an hparam, which a deploy yaml cannot serialize.
    """

    OUTPUT_SPECS = {"cube": _CUBE, "wavelengths": _WAVELENGTHS_SPEC}

    def __init__(self, sample_path: str, **kwargs) -> None:
        super().__init__(sample_path=sample_path, **kwargs)
        self.sample_path = sample_path

    def forward(self, **_) -> dict[str, torch.Tensor]:
        cube = torch.load(self.sample_path, weights_only=True)
        return {"cube": cube, "wavelengths": np.array(WAVELENGTHS, dtype=np.int32)}


def _build_pipeline(tmp_path) -> tuple[CuvisPipeline, torch.Tensor]:
    """A normalizer+selector pipeline; the source serves one frame at a time, matching
    ExportedSpan's batch-1 contract (the exported graph is traced at batch size 1 and the
    node loops over a larger batch itself)."""
    torch.manual_seed(0)
    fit_cube = torch.rand(4, 4, 4, 5)  # several frames, fit only
    frame = torch.rand(1, 4, 4, 5)  # the single frame the source actually serves
    sample_path = tmp_path / "sample.pt"
    torch.save(frame, sample_path)
    source = _Source(sample_path=str(sample_path), name="source")
    normalizer = MinMaxNormalizer(use_running_stats=True, name="normalizer")
    normalizer.statistical_initialization(iter([{"data": fit_cube}]))
    selector = FixedWavelengthSelector(
        target_wavelengths=(650.0, 550.0, 450.0), normalize_output=False, name="selector"
    )

    pipeline = CuvisPipeline("toy")
    pipeline.connect(
        (source.outputs.cube, normalizer.inputs.data),
        (normalizer.outputs.normalized, selector.inputs.cube),
        (source.outputs.wavelengths, selector.inputs.wavelengths),
    )
    return pipeline, frame


def test_builtin_span_exports_and_deploy_folder_reproduces_the_pipeline(tmp_path) -> None:
    """Export the normalizer+selector span with wavelengths as a constant and reload it."""
    from cuvis_ai.node.exported_span import DEPLOY_DIR_ENV

    pipeline, cube = _build_pipeline(tmp_path)
    reference = pipeline.forward(batch={}, stage=ExecutionStage.INFERENCE)

    examples = [
        {
            "normalizer.data": cube,
            "selector.wavelengths": torch.tensor(WAVELENGTHS, dtype=torch.int32),
        }
    ]
    result = export.export_deployment(
        pipeline,
        tmp_path / "deploy",
        inputs=["normalizer.data", "selector.wavelengths"],
        outputs=["selector.rgb_image"],
        examples=examples,
        constants=["selector.wavelengths"],
    )

    os.environ[DEPLOY_DIR_ENV] = str(result.yaml_path.parent)
    try:
        deployed = CuvisPipeline.load_pipeline(
            result.yaml_path, weights_path=result.weights_path, device="cpu"
        )
        out = deployed.forward(batch={}, stage=ExecutionStage.INFERENCE)
    finally:
        os.environ.pop(DEPLOY_DIR_ENV, None)

    assert [n.name for n in deployed.nodes] == ["source", "exported_span"]
    torch.testing.assert_close(
        out[("exported_span", "rgb_image")],
        reference[("selector", "rgb_image")],
        atol=1e-5,
        rtol=1e-4,
    )
