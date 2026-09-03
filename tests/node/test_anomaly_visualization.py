"""Artifact visualizers: stage declarations, yaml-style overrides, and forward smoke tests."""

from __future__ import annotations

import pytest
import torch
from cuvis_ai_schemas.enums import ArtifactType, ExecutionStage, NodeTag
from cuvis_ai_schemas.execution import Context

from cuvis_ai.node.anomaly_visualization import (
    AnomalyMask,
    RGBAnomalyMask,
    ScoreHeatmapVisualizer,
)

pytestmark = pytest.mark.unit

VAL_TEST = frozenset({ExecutionStage.VAL, ExecutionStage.TEST})


def _context() -> Context:
    return Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0, global_step=0)


def _assert_image_artifacts(artifacts: list, expected: int) -> None:
    assert len(artifacts) == expected
    for artifact in artifacts:
        assert artifact.type == ArtifactType.IMAGE
        assert artifact.value.ndim == 3
        assert artifact.value.shape[-1] in (1, 3)
        assert artifact.value.dtype.kind == "u"


@pytest.mark.parametrize(
    "factory",
    [
        lambda **kw: ScoreHeatmapVisualizer(up_to=2, **kw),
        lambda **kw: AnomalyMask(channel=0, up_to=2, **kw),
        lambda **kw: RGBAnomalyMask(up_to=2, **kw),
    ],
    ids=["ScoreHeatmapVisualizer", "AnomalyMask", "RGBAnomalyMask"],
)
def test_visualizers_run_in_val_and_test_only(factory):
    node = factory()
    assert type(node).EXECUTION_STAGES == VAL_TEST
    assert NodeTag.EVALUATION in type(node).get_tags()
    assert node.should_execute(ExecutionStage.VAL)
    assert node.should_execute(ExecutionStage.TEST)
    assert not node.should_execute(ExecutionStage.INFERENCE)
    assert not node.should_execute(ExecutionStage.TRAIN)

    # A pipeline yaml can move the node with `hparams: {execution_stages: [inference]}`.
    moved = factory(execution_stages=["inference"])
    assert moved.should_execute(ExecutionStage.INFERENCE)
    assert "execution_stages" not in moved.hparams


def test_score_heatmap_forward_bounded_by_up_to():
    node = ScoreHeatmapVisualizer(up_to=2)
    out = node.forward(scores=torch.rand(3, 8, 8, 1), context=_context())
    _assert_image_artifacts(out["artifacts"], 2)


def test_anomaly_mask_forward_with_and_without_ground_truth():
    node = AnomalyMask(channel=1, up_to=2)
    decisions = torch.rand(2, 8, 8, 1) > 0.5
    cube = torch.rand(2, 8, 8, 4)
    with_gt = node.forward(
        decisions=decisions, cube=cube, context=_context(), mask=torch.rand(2, 8, 8, 1) > 0.5
    )
    without_gt = node.forward(decisions=decisions, cube=cube, context=_context(), mask=None)
    assert len(with_gt["artifacts"]) > 0
    assert len(without_gt["artifacts"]) > 0
    _assert_image_artifacts(with_gt["artifacts"], len(with_gt["artifacts"]))
    _assert_image_artifacts(without_gt["artifacts"], len(without_gt["artifacts"]))


def test_rgb_anomaly_mask_forward():
    node = RGBAnomalyMask(up_to=2)
    out = node.forward(
        decisions=torch.rand(3, 8, 8, 1) > 0.5,
        rgb_image=torch.rand(3, 8, 8, 3),
        context=_context(),
        mask=torch.rand(3, 8, 8, 1) > 0.5,
    )
    assert 0 < len(out["artifacts"])
    _assert_image_artifacts(out["artifacts"], len(out["artifacts"]))
