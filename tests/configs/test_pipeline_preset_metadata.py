"""Tags the CuvisNEXT pipeline pickers read off every shipped preset.

The picker derives a preset's category from the first word of ``PIPELINE_CATEGORIES`` found in
``metadata.tags`` and lists presets tagged ``cuvisnext`` (plus saved training runs) under its
default filter. cuvis-next mirrors the vocabulary as ``kPipelineCategories`` in
``libs/pilot_utility/include/pilot_utility/pipeline_catalog.hpp``; change both together.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from cuvis_ai_schemas.pipeline.config import PipelineConfig

import cuvis_ai

pytestmark = pytest.mark.unit

CONFIGS = Path(cuvis_ai.__file__).resolve().parent / "configs"
PIPELINES = CONFIGS / "pipeline"
PRESETS = sorted(PIPELINES.rglob("*.yaml"))
TRAINRUNS = sorted((CONFIGS / "trainrun").glob("*_cuvisnext.yaml"))

PIPELINE_CATEGORIES = ("anomaly", "segmentation", "tracking", "thickness", "medical")
CUVISNEXT_PRESETS = {
    "rtsam2/rtsam2_mask_propagation_view.yaml",
    "rtsam2/rtsam2_point_expansion_postprocess_view.yaml",
    "rtsam2/rtsam2_point_expansion_view.yaml",
    "sam3/sam3_mask_propagation_postprocess_view.yaml",
    "sam3/sam3_mask_propagation_view.yaml",
    "sam3/sam3_point_expansion_postprocess_view.yaml",
    "sam3/sam3_point_expansion_view.yaml",
    "wafer_thickness/wafer_thickness_pipeline_500nm_cuvisnext_cube.yaml",
    "wafer_thickness/wafer_thickness_pipeline_cuvisnext_cube.yaml",
}
# Kept beside ``anomaly``: core's discovery filters by exact tag, so removing it would empty
# existing ``filter_tag="anomaly_detection"`` queries.
LEGACY_ANOMALY_DETECTION = {
    "anomaly/adaclip/concrete_adaclip_gradient_two_stage.yaml",
    "anomaly/adaclip/drcnn_adaclip_gradient.yaml",
    "anomaly/deep_svdd/deep_svdd.yaml",
}


def _rel(path: Path) -> str:
    return path.relative_to(PIPELINES).as_posix()


def _tags(path: Path) -> list[str]:
    return PipelineConfig.load_from_file(path).metadata.tags


def test_presets_were_found():
    assert len(PRESETS) >= 42


@pytest.mark.parametrize("preset", PRESETS, ids=_rel)
def test_preset_carries_exactly_one_category_tag(preset: Path):
    tags = [tag.lower() for tag in _tags(preset)]
    found = [category for category in PIPELINE_CATEGORIES if category in tags]
    assert len(found) == 1, (
        f"{_rel(preset)}: expected exactly one of {PIPELINE_CATEGORIES} in metadata.tags, found {found}"
    )


def test_cuvisnext_set_is_pinned():
    tagged = {_rel(preset) for preset in PRESETS if "cuvisnext" in _tags(preset)}
    assert tagged == CUVISNEXT_PRESETS


@pytest.mark.parametrize("trainrun_yaml", TRAINRUNS, ids=lambda p: p.stem)
def test_cuvisnext_trainrun_source_carries_a_category(trainrun_yaml: Path):
    """A saved run inherits its source preset's tags, so the source must be categorised."""
    trainrun = yaml.safe_load(trainrun_yaml.read_text(encoding="utf-8"))
    pipeline = (trainrun_yaml.parent / trainrun["pipeline"]).resolve()
    tags = [tag.lower() for tag in _tags(pipeline)]
    assert any(category in tags for category in PIPELINE_CATEGORIES), _rel(pipeline)


@pytest.mark.parametrize("rel", sorted(LEGACY_ANOMALY_DETECTION))
def test_legacy_anomaly_detection_tag_is_kept(rel: str):
    assert "anomaly_detection" in _tags(PIPELINES / rel)
