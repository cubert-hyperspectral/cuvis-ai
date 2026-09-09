"""Guards on the presets and trainruns CuvisNEXT starts a training run from.

CuvisNEXT saves a training run's pipeline yaml verbatim and its picker renders only the
hparam keys present in that yaml. For the operator to set the image gate (and a calibrated
pixel threshold) after training without editing files, the training preset itself must carry
``image_threshold: null`` and ``pixel_threshold: null`` on a ``TwoStageBinaryDecider``.

The in-app run is also the memory-tightest path, so two settings are pinned here: the
Dinomaly presets stride the pixel metrics (``pixel_stride: 2``), and every CuvisNEXT
trainrun opts into releasing the CUDA cache around validation.
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
# Both metric nodes of a Dinomaly preset score pixels; 1000x1080 at stride 2 leaves
# 270k px per frame, deliberately above the 50k count below which torchmetrics
# switches to a vectorized path that allocates an [N, thresholds] matrix.
STRIDED_METRIC_NODES = ("metrics_anomaly", "metrics_auroc")
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
def test_dinomaly_preset_metric_nodes_stride_the_pixel_grid(pipeline_yaml: Path):
    """Both metric nodes of a Dinomaly training preset carry ``pixel_stride: 2``."""
    pipeline = PipelineConfig.load_from_file(pipeline_yaml)
    hparams = {node.name: node.hparams for node in pipeline.nodes}
    for node_name in STRIDED_METRIC_NODES:
        assert node_name in hparams, f"{pipeline_yaml.name}: no node named {node_name!r}"
        assert hparams[node_name].get("pixel_stride") == PIXEL_STRIDE, (
            f"{pipeline_yaml.name}: {node_name} must set pixel_stride: {PIXEL_STRIDE}"
        )


@pytest.mark.parametrize("trainrun_yaml", TRAINRUNS, ids=lambda p: p.stem)
def test_cuvisnext_trainrun_releases_the_cuda_cache_on_validation(trainrun_yaml: Path):
    """The in-app trainruns opt into the (schema-default-off) validation cache release."""
    trainrun = yaml.safe_load(trainrun_yaml.read_text(encoding="utf-8"))
    training = trainrun.get("training") or {}
    assert training.get("release_cuda_cache_on_validation") is True, (
        f"{trainrun_yaml.name}: training.release_cuda_cache_on_validation must be true"
    )
