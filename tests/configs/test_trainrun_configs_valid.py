"""The shipped trainrun/training configs validate against the flat schema.

Backstop for the ``TrainerConfig`` fold: every ``training:`` block in a
packaged trainrun (and every ``training/*.yaml`` group file) must parse as a
flat ``TrainingConfig`` with no nested ``trainer`` key. Catches an un-migrated
config that would otherwise fail ``extra="forbid"`` only at run time.

Every packaged trainrun also loads the way ``restore-trainrun`` reads it and
references a pipeline yaml that exists, so a trainrun whose script went away
cannot stay in the catalog unrunnable.
"""

from pathlib import Path

import pytest
import yaml
from cuvis_ai_schemas.training import TrainingConfig, TrainRunConfig

from cuvis_ai_core.utils.config_helpers import resolve_config_with_hydra

_CONFIGS = Path(__file__).resolve().parents[2] / "cuvis_ai" / "configs"
_TRAINRUNS = sorted((_CONFIGS / "trainrun").glob("*.yaml"))
_TRAINING_GROUPS = sorted((_CONFIGS / "training").glob("*.yaml"))


# AnomalyDetectionMetrics reports these only when its ``decisions`` input carries a value
# (cuvis_ai/node/metrics.py); average_precision and the AUROC metrics come from the scores.
_DECISION_METRICS = {
    "image_precision",
    "image_recall",
    "image_f1",
    "precision",
    "recall",
    "f1_score",
    "iou",
}


def _load_trainrun(path: Path) -> TrainRunConfig:
    """Load a trainrun the two ways ``restore-trainrun`` does.

    A trainrun with a ``defaults`` block is composed with Hydra against the packaged
    config tree, a flat one is read as it is.
    """
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "defaults" in doc:
        return TrainRunConfig.model_validate(
            resolve_config_with_hydra(
                config_type="trainrun",
                config_path=path.relative_to(_CONFIGS).with_suffix("").as_posix(),
                search_paths=[str(_CONFIGS)],
            )
        )
    return TrainRunConfig.load_from_file(path)


def _monitors(node: object) -> set[str]:
    """Every ``monitor`` value in a training block: scheduler, checkpoint, early stopping."""
    if isinstance(node, dict):
        found = {node["monitor"]} if isinstance(node.get("monitor"), str) else set()
        return found.union(*(_monitors(value) for value in node.values()))
    if isinstance(node, list):
        return set().union(*(_monitors(item) for item in node))
    return set()


def _undecided_metric_nodes(pipeline: dict) -> set[str]:
    """Metrics nodes whose ``decisions`` come from a TwoStageBinaryDecider that is uncalibrated.

    Such a decider (one of ``image_threshold`` / ``pixel_threshold`` unset) emits no
    decisions at TRAIN, VAL and TEST, so the metrics node logs no thresholded metric.
    """
    nodes = {node["name"]: node for node in pipeline.get("nodes", [])}
    undecided = set()
    for connection in pipeline.get("connections", []):
        target_node, _, target_port = connection["target"].partition(".inputs.")
        if target_port != "decisions":
            continue
        target = nodes.get(target_node, {})
        source = nodes.get(connection["source"].split(".outputs.")[0], {})
        hparams = source.get("hparams") or {}
        if (
            target.get("class_name", "").endswith("AnomalyDetectionMetrics")
            and source.get("class_name", "").endswith("TwoStageBinaryDecider")
            and (hparams.get("image_threshold") is None or hparams.get("pixel_threshold") is None)
        ):
            undecided.add(target_node)
    return undecided


@pytest.mark.parametrize("path", _TRAINRUNS, ids=lambda p: p.name)
def test_trainrun_loads_like_restore_trainrun(path: Path) -> None:
    """A trainrun validates as a TrainRunConfig and names an existing pipeline yaml."""
    config = _load_trainrun(path)
    assert config.pipeline, "restore-trainrun builds the pipeline the trainrun references"
    assert (path.parent / config.pipeline).is_file(), f"missing pipeline {config.pipeline}"


@pytest.mark.parametrize("path", _TRAINRUNS, ids=lambda p: p.name)
def test_trainrun_monitors_a_metric_the_pipeline_logs_while_training(path: Path) -> None:
    """No scheduler, checkpoint or early stopping waits on a metric an uncalibrated decider withholds.

    Lightning raises after the first validation when a monitored key was never logged.
    """
    config = _load_trainrun(path)
    if config.training is None:
        pytest.skip("statistical-only trainrun (no training block)")
    assert config.pipeline, "restore-trainrun builds the pipeline the trainrun references"
    pipeline = yaml.safe_load((path.parent / config.pipeline).read_text(encoding="utf-8"))
    withheld = {
        f"{node}/{metric}"
        for node in _undecided_metric_nodes(pipeline)
        for metric in _DECISION_METRICS
    }
    assert not _monitors(config.training.model_dump()) & withheld, (
        "monitors a thresholded metric the uncalibrated decider does not produce during training"
    )


@pytest.mark.parametrize("path", _TRAINRUNS, ids=lambda p: p.name)
def test_trainrun_training_block_is_flat(path: Path) -> None:
    """A trainrun's ``training`` block (if any) validates as a flat TrainingConfig."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    training = doc.get("training")
    if training is None:
        pytest.skip("statistical-only trainrun (no training block)")
    assert "trainer" not in training, "nested 'trainer' block must be flattened"
    TrainingConfig.model_validate(training)


@pytest.mark.parametrize("path", _TRAINING_GROUPS, ids=lambda p: p.name)
def test_training_group_file_is_flat(path: Path) -> None:
    """A ``training/*.yaml`` group file validates as a flat TrainingConfig."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    assert "trainer" not in doc, "nested 'trainer' block must be flattened"
    TrainingConfig.model_validate(doc)
