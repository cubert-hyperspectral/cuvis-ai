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

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"

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
