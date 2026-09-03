"""TensorBoardMonitorNode: stage declaration and lazy writer creation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from cuvis_ai_schemas.enums import ArtifactType, ExecutionStage, NodeTag
from cuvis_ai_schemas.execution import Artifact, Context, Metric

from cuvis_ai.node.monitor import TensorBoardMonitorNode

pytestmark = pytest.mark.unit


def _artifact() -> Artifact:
    return Artifact(
        name="img",
        value=np.zeros((8, 8, 3), dtype=np.uint8),
        el_id=0,
        desc="test image",
        type=ArtifactType.IMAGE,
        stage=ExecutionStage.VAL,
    )


def _metric() -> Metric:
    return Metric(name="loss", value=0.5, stage=ExecutionStage.VAL)


def _context(step: int = 1) -> Context:
    return Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0, global_step=step)


def _run_dirs(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir()) if root.exists() else []


class TestStages:
    def test_class_declares_training_stages_and_tags(self):
        assert TensorBoardMonitorNode.EXECUTION_STAGES == {
            ExecutionStage.TRAIN,
            ExecutionStage.VAL,
            ExecutionStage.TEST,
        }
        assert {NodeTag.TRAINING, NodeTag.EVALUATION} <= TensorBoardMonitorNode.get_tags()

    def test_default_instance_is_pruned_at_inference(self, tmp_path: Path):
        node = TensorBoardMonitorNode(output_dir=str(tmp_path / "tb"))
        assert node.should_execute(ExecutionStage.TRAIN)
        assert node.should_execute(ExecutionStage.VAL)
        assert not node.should_execute(ExecutionStage.INFERENCE)

    def test_yaml_style_override_opts_into_inference(self, tmp_path: Path):
        # What `hparams: {execution_stages: [inference]}` delivers through PipelineFactory.
        node = TensorBoardMonitorNode(
            output_dir=str(tmp_path / "tb"), execution_stages=["inference"]
        )
        assert node.should_execute(ExecutionStage.INFERENCE)
        assert not node.should_execute(ExecutionStage.TRAIN)
        assert "execution_stages" not in node.hparams


class TestLazyWriter:
    def test_construction_touches_no_disk(self, tmp_path: Path):
        root = tmp_path / "tb"
        node = TensorBoardMonitorNode(output_dir=str(root))
        assert not root.exists()
        assert node.log_dir is None
        assert node._writer is None

    def test_first_forward_creates_exactly_one_run_dir(self, tmp_path: Path):
        root = tmp_path / "tb"
        node = TensorBoardMonitorNode(output_dir=str(root))
        node.forward(artifacts=[_artifact()], metrics=[_metric()], context=_context(1))
        assert _run_dirs(root) == ["run_01"]
        assert node.log_dir == root / "run_01"
        node.forward(artifacts=[[_artifact()]], metrics=[[_metric()]], context=_context(2))
        assert _run_dirs(root) == ["run_01"]
        node.cleanup()

    def test_log_before_any_forward_creates_the_writer(self, tmp_path: Path):
        root = tmp_path / "tb"
        node = TensorBoardMonitorNode(output_dir=str(root))
        node.log("train/loss", 0.5, step=1)
        assert _run_dirs(root) == ["run_01"]
        node.cleanup()

    def test_cleanup_is_idempotent_and_survives_no_writer(self, tmp_path: Path):
        node = TensorBoardMonitorNode(output_dir=str(tmp_path / "tb"))
        node.cleanup()  # nothing was created: no-op
        node.forward(metrics=[_metric()], context=_context())
        node.cleanup()
        assert node._writer is None
        node.cleanup()

    def test_run_name_versioning_applies_on_first_use(self, tmp_path: Path):
        root = tmp_path / "tb"
        (root / "exp").mkdir(parents=True)
        node = TensorBoardMonitorNode(output_dir=str(root), run_name="exp")
        assert _run_dirs(root) == ["exp"]
        node.log("x", 1.0, step=0)
        assert _run_dirs(root) == ["exp", "exp_v2"]
        node.cleanup()
