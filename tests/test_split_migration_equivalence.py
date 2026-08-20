"""Migration-equivalence: the migrated selector configs resolve to the intended ids.

Proves the flat-id-list -> selector migration preserved per-stage membership: each
shipped config's selectors, resolved against a synthetic universe for its source, yield
exactly the measurement indices the old `train_ids/val_ids/test_ids` encoded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from cuvis_ai_schemas.training.data import (
    Constraint,
    ConstraintKind,
    ConstraintSeverity,
    DataSplitConfig,
    SampleRef,
)

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
    """A migrated config with intentional train/val/test overlap downgrades the hard guard."""
    raw = yaml.safe_load(
        (_CONFIGS / "trainrun" / "adaclip_cir_false_color.yaml").read_text(encoding="utf-8")
    )
    splits = DataSplitConfig.model_validate(raw["data"]["splits"])
    # train/val share frame 2, so no_split_overlap is declared at warn, not error.
    assert splits.constraints == [
        Constraint(kind=ConstraintKind.NO_SPLIT_OVERLAP, severity=ConstraintSeverity.WARN)
    ]
    universe = _universe("data/Lentils/Lentils_000.cu3s")
    assert _resolved_ids(splits.train, universe) == [0, 2]
    assert _resolved_ids(splits.val, universe) == [2, 4]


# Every shipped config with an inline splits block and its declared overlap severity:
# warn = intentional small-dataset reuse, error = the old schemas-0.8 default hard guard.
_CONSTRAINT_CASES = [
    ("trainrun/adaclip_baseline.yaml", ("data", "splits"), ConstraintSeverity.ERROR),
    ("trainrun/adaclip_cir_false_color.yaml", ("data", "splits"), ConstraintSeverity.WARN),
    (
        "trainrun/adaclip_cir_false_color_optimal_threshold.yaml",
        ("data", "splits"),
        ConstraintSeverity.WARN,
    ),
    ("trainrun/adaclip_cir_false_rg_color.yaml", ("data", "splits"), ConstraintSeverity.WARN),
    ("trainrun/adaclip_high_contrast.yaml", ("data", "splits"), ConstraintSeverity.WARN),
    ("trainrun/adaclip_supervised_cir.yaml", ("data", "splits"), ConstraintSeverity.WARN),
    ("trainrun/adaclip_supervised_full_spectrum.yaml", ("data", "splits"), ConstraintSeverity.WARN),
    (
        "trainrun/adaclip_supervised_windowed_false_rgb.yaml",
        ("data", "splits"),
        ConstraintSeverity.WARN,
    ),
    ("trainrun/deep_svdd.yaml", ("data", "splits"), ConstraintSeverity.ERROR),
    ("trainrun/drcnn_adaclip_trainrun.yaml", ("data", "splits"), ConstraintSeverity.WARN),
    ("data/lentils.yaml", ("splits",), ConstraintSeverity.ERROR),
    ("data/tracking_cap_and_car.yaml", ("splits",), ConstraintSeverity.ERROR),
]


@pytest.mark.parametrize("rel_path, keys, severity", _CONSTRAINT_CASES, ids=lambda c: str(c))
def test_inline_splits_declare_overlap_constraint(rel_path, keys, severity):
    """Every inline splits block declares no_split_overlap explicitly.

    schemas 0.9.0 removed the implicit default guard (``leakage_check`` -> typed
    ``constraints``, absent = no checks), so silence would silently disable it.
    splits_path-only configs are exempt: constraints are file-owned there — the
    datamodule takes them from the loaded splits.json and ignores inline ones.
    """
    raw = yaml.safe_load((_CONFIGS / rel_path).read_text(encoding="utf-8"))
    for key in keys:
        raw = raw[key]
    splits = DataSplitConfig.model_validate(raw)
    assert splits.constraints == [
        Constraint(kind=ConstraintKind.NO_SPLIT_OVERLAP, severity=severity)
    ]


def test_no_config_still_uses_leakage_check():
    """`leakage_check` was removed in schemas 0.9.0 (extra=forbid); nothing may still say it."""
    offenders = [
        str(p.relative_to(_CONFIGS))
        for p in sorted(_CONFIGS.rglob("*.yaml"))
        if "leakage_check" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"configs still using removed leakage_check: {offenders}"


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


def test_trainrun_node_references_exist_in_pipeline():
    """Trainrun node lists and the checkpoint monitor name real pipeline nodes.

    Guards against a pipeline edit orphaning its trainrun: dropping a loss/metric
    node from the pipeline (e.g. slimming a preset for inference) silently breaks
    RestoreTrainRun for every trainrun that still names it.
    """
    checked = 0
    for trainrun_path, ref in _string_pipeline_configs():
        raw = yaml.safe_load(trainrun_path.read_text(encoding="utf-8"))
        pipeline = yaml.safe_load(
            (trainrun_path.parent / ref).resolve().read_text(encoding="utf-8")
        )
        node_names = {node["name"] for node in pipeline.get("nodes", [])}
        for key in ("loss_nodes", "metric_nodes", "unfreeze_nodes"):
            missing = set(raw.get(key) or []) - node_names
            assert not missing, f"{trainrun_path.name}: {key} {sorted(missing)} not in {ref}"
        checkpoint = ((raw.get("training") or {}).get("callbacks") or {}).get("checkpoint") or {}
        monitor = checkpoint.get("monitor")
        if monitor and "/" in monitor:
            node = monitor.split("/", 1)[0]
            assert node in node_names, (
                f"{trainrun_path.name}: checkpoint monitor {monitor!r} names no node in {ref}"
            )
        checked += 1
    assert checked >= 12
