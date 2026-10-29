"""Guards on the presets and trainruns CuvisNEXT starts a training run from.

CuvisNEXT saves a training run's pipeline yaml verbatim and its picker renders only the
hparam keys present in that yaml. For the operator to set the image gate (and a calibrated
pixel threshold) after training without editing files, the training preset itself must carry
``image_threshold: null`` and ``pixel_threshold: null`` on a ``TwoStageBinaryDecider``.

Two more settings are pinned here. The Dinomaly presets set no ``pixel_stride`` on their
``AnomalyDetectionMetrics`` node, so the evaluation's pixel precision and recall count the
pixels the decider calibration counted and a precision or recall target set in the app reads
the same on the results tab; their AUROC node keeps ``pixel_stride: 2`` (a speed choice, its
200-bin histogram holds no per-pixel state). Every CuvisNEXT trainrun opts into releasing the
CUDA cache around validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from cuvis_ai_schemas.pipeline.config import PipelineConfig

import cuvis_ai

pytestmark = pytest.mark.unit

CONFIGS = Path(cuvis_ai.__file__).resolve().parent / "configs"
TRAINRUNS = sorted((CONFIGS / "trainrun").glob("*_cuvisnext.yaml"))
DINOMALY_PRESETS = [
    CONFIGS / "pipeline" / "anomaly" / "dinomaly" / name
    for name in ("dinomaly_cir.yaml", "dinomaly_custom.yaml", "dinomaly_rgb.yaml")
]
# The AUROC node strides the pixel grid: 1000x1080 at stride 2 leaves 270k px per frame,
# deliberately above the 50k count below which torchmetrics switches to a vectorized path
# that allocates an [N, thresholds] matrix. The anomaly metrics node counts every pixel.
STRIDED_METRIC_NODE = "metrics_auroc"
UNSTRIDED_METRIC_NODE = "metrics_anomaly"
PIXEL_STRIDE = 2


def _pipeline_path(trainrun_yaml: Path) -> Path:
    trainrun = yaml.safe_load(trainrun_yaml.read_text(encoding="utf-8"))
    return (trainrun_yaml.parent / trainrun["pipeline"]).resolve()


def test_every_cuvisnext_trainrun_is_covered():
    """The wizard offers exactly one trainrun; a new ``*_cuvisnext`` file must be added here."""
    assert {p.name for p in TRAINRUNS} == {"dinomaly_custom_cuvisnext.yaml"}


@pytest.mark.parametrize("trainrun_yaml", TRAINRUNS, ids=lambda p: p.stem)
def test_training_preset_decider_carries_null_thresholds(trainrun_yaml: Path):
    pipeline = PipelineConfig.load_from_file(_pipeline_path(trainrun_yaml))
    deciders = [node for node in pipeline.nodes if node.name == "decider"]
    assert len(deciders) == 1, f"{trainrun_yaml.name}: expected one node named 'decider'"
    decider = deciders[0]
    assert decider.class_name.endswith(".TwoStageBinaryDecider"), decider.class_name
    hparams = decider.hparams
    assert "image_threshold" in hparams and hparams["image_threshold"] is None
    assert "pixel_threshold" in hparams and hparams["pixel_threshold"] is None
    assert hparams["quantile"] == 0.995
    assert hparams["top_k_fraction"] == 0.001


@pytest.mark.parametrize("pipeline_yaml", DINOMALY_PRESETS, ids=lambda p: p.stem)
def test_dinomaly_preset_anomaly_metrics_count_every_pixel(pipeline_yaml: Path):
    """The anomaly metrics node of a Dinomaly preset sets no ``pixel_stride``.

    The decider calibration counts every pixel of the validation frames, so the evaluation's
    pixel precision and recall must count the same grid for a target set in the app to read
    the same on the results tab.
    """
    pipeline = PipelineConfig.load_from_file(pipeline_yaml)
    hparams = {node.name: node.hparams for node in pipeline.nodes}
    assert UNSTRIDED_METRIC_NODE in hparams, (
        f"{pipeline_yaml.name}: no node named {UNSTRIDED_METRIC_NODE!r}"
    )
    assert "pixel_stride" not in (hparams[UNSTRIDED_METRIC_NODE] or {}), (
        f"{pipeline_yaml.name}: {UNSTRIDED_METRIC_NODE} must count every pixel"
    )


@pytest.mark.parametrize("pipeline_yaml", DINOMALY_PRESETS, ids=lambda p: p.stem)
def test_dinomaly_preset_auroc_node_strides_the_pixel_grid(pipeline_yaml: Path):
    """The AUROC node of a Dinomaly training preset carries ``pixel_stride: 2``."""
    pipeline = PipelineConfig.load_from_file(pipeline_yaml)
    hparams = {node.name: node.hparams for node in pipeline.nodes}
    assert STRIDED_METRIC_NODE in hparams, (
        f"{pipeline_yaml.name}: no node named {STRIDED_METRIC_NODE!r}"
    )
    assert hparams[STRIDED_METRIC_NODE].get("pixel_stride") == PIXEL_STRIDE, (
        f"{pipeline_yaml.name}: {STRIDED_METRIC_NODE} must set pixel_stride: {PIXEL_STRIDE}"
    )


@pytest.mark.parametrize("trainrun_yaml", TRAINRUNS, ids=lambda p: p.stem)
def test_cuvisnext_trainrun_releases_the_cuda_cache_on_validation(trainrun_yaml: Path):
    """The in-app trainruns opt into the (schema-default-off) validation cache release."""
    trainrun = yaml.safe_load(trainrun_yaml.read_text(encoding="utf-8"))
    training = trainrun.get("training") or {}
    assert training.get("release_cuda_cache_on_validation") is True, (
        f"{trainrun_yaml.name}: training.release_cuda_cache_on_validation must be true"
    )
