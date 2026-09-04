"""The CuvisNEXT training presets ship a two-stage decider with both thresholds unset.

CuvisNEXT saves a training run's pipeline yaml verbatim and its picker renders only the
hparam keys present in that yaml. For the operator to set the image gate (and a calibrated
pixel threshold) after training without editing files, the training preset itself must carry
``image_threshold: null`` and ``pixel_threshold: null`` on a ``TwoStageBinaryDecider``.
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


def _pipeline_path(trainrun_yaml: Path) -> Path:
    trainrun = yaml.safe_load(trainrun_yaml.read_text(encoding="utf-8"))
    return (trainrun_yaml.parent / trainrun["pipeline"]).resolve()


def test_every_cuvisnext_trainrun_is_covered():
    assert {p.name for p in TRAINRUNS} == {
        "adaclip_supervised_cir_cuvisnext.yaml",
        "dinomaly_cir_cuvisnext.yaml",
        "dinomaly_rgb_cuvisnext.yaml",
    }


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
