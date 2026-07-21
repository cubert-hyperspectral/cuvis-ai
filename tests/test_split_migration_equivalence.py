"""Migration-equivalence: the migrated selector configs resolve to the intended ids.

Proves the flat-id-list -> selector migration preserved per-stage membership: each
shipped config's selectors, resolved against a synthetic universe for its source, yield
exactly the measurement indices the old `train_ids/val_ids/test_ids` encoded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from cuvis_ai_schemas.training.data import DataSplitConfig, SampleRef

from cuvis_ai_core.data.selectors import resolve_selectors

_CONFIGS = Path(__file__).resolve().parents[1] / "cuvis_ai" / "configs"

# (config path, source path, expected {stage: sorted measurement ids})
_CASES = [
    (
        _CONFIGS / "data" / "lentils.yaml",
        "data/Lentils/Lentils_000.cu3s",
        {"train": [0, 2, 3], "val": [1], "test": [5]},
    ),
]


def _universe(source: str, n: int = 16) -> list[SampleRef]:
    return [SampleRef(source=source, index=i, label_id=i) for i in range(n)]


def _resolved_ids(selectors, universe) -> list[int]:
    return sorted(r.index for r in resolve_selectors(selectors, universe))


@pytest.mark.parametrize("config_path, source, expected", _CASES)
def test_config_selectors_resolve_to_expected_ids(config_path, source, expected):
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    splits = DataSplitConfig.model_validate(raw["splits"])
    universe = _universe(source)
    for stage, ids in expected.items():
        assert _resolved_ids(getattr(splits, stage), universe) == ids


def test_overlapping_config_declares_warn():
    """A migrated config with intentional train/val/test overlap opts out of the hard guard."""
    raw = yaml.safe_load(
        (_CONFIGS / "trainrun" / "adaclip_cir_false_color.yaml").read_text(encoding="utf-8")
    )
    splits = DataSplitConfig.model_validate(raw["data"]["splits"])
    assert splits.leakage_check == "warn"  # train/val share frame 2
    universe = _universe("data/Lentils/Lentils_000.cu3s")
    assert _resolved_ids(splits.train, universe) == [0, 2]
    assert _resolved_ids(splits.val, universe) == [2, 4]


_TRAINRUN_DIR = _CONFIGS / "trainrun"


def _string_pipeline_configs():
    """Trainrun configs whose ``pipeline`` is a path reference (the migrated form).

    Skips script-driven configs whose ``pipeline`` is a parameter-override mapping
    (not a real pipeline) and configs with no pipeline at all.
    """
    for path in sorted(_TRAINRUN_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("pipeline"), str):
            yield path, raw["pipeline"]


def test_trainrun_pipeline_references_resolve():
    """Every migrated trainrun references its pipeline by path (no inline nodes/connections)."""
    from cuvis_ai_schemas.pipeline.config import PipelineConfig

    checked = 0
    for trainrun_path, ref in _string_pipeline_configs():
        resolved = (trainrun_path.parent / ref).resolve()
        assert resolved.is_file(), f"{trainrun_path.name}: pipeline ref {ref!r} not found"
        pipeline = PipelineConfig.load_from_file(resolved)
        assert pipeline.nodes, f"{trainrun_path.name}: referenced pipeline has no nodes"
        checked += 1
    assert checked >= 12  # 12 migrated Hydra configs + 2 extracted snapshots
