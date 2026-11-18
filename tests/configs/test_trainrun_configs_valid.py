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


@pytest.mark.parametrize("path", _TRAINRUNS, ids=lambda p: p.name)
def test_trainrun_loads_like_restore_trainrun(path: Path) -> None:
    """A trainrun validates as a TrainRunConfig and names an existing pipeline yaml.

    A trainrun with a ``defaults`` block is composed with Hydra against the packaged
    config tree, a flat one is read as it is, the two ways ``restore-trainrun`` loads one.
    """
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "defaults" in doc:
        config = TrainRunConfig.model_validate(
            resolve_config_with_hydra(
                config_type="trainrun",
                config_path=path.relative_to(_CONFIGS).with_suffix("").as_posix(),
                search_paths=[str(_CONFIGS)],
            )
        )
    else:
        config = TrainRunConfig.load_from_file(path)
    assert config.pipeline, "restore-trainrun builds the pipeline the trainrun references"
    assert (path.parent / config.pipeline).is_file(), f"missing pipeline {config.pipeline}"


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
