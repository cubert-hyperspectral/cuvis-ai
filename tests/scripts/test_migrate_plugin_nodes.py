"""migrate-plugin-nodes: class, hparam, connection and state rewrites, the reports, and that a
migrated pipeline builds from the built-ins and reproduces the former arithmetic."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from click.testing import CliRunner

from cuvis_ai.node.fusion import ScoreMapFusion
from cuvis_ai.node.normalization import PercentileNormalizer
from scripts.migrate_plugin_nodes import (
    PERCENTILE,
    RENAMED,
    SCORE_FUSION,
    SCORE_INTERSECTION,
    SCORE_MAP_FUSION,
    SCORE_RANGE,
    STRETCH,
    MigrationError,
    Report,
    main,
    migrate_doc,
    migrate_file,
)

pytestmark = pytest.mark.unit


def _doc() -> dict:
    """A pipeline with one node of every rewrite kind (the sources are plain built-ins)."""
    return {
        "plugins": ["patchcore", "steervit", "rfdetr_seg"],
        "nodes": [
            {"name": "src", "class_name": "cuvis_ai.node.data.CU3SDataNode", "hparams": {}},
            {
                "name": "sel",
                "class_name": "cuvis_ai.node.channel_selector.FixedWavelengthSelector",
                "hparams": {},
            },
            {
                "name": "stretch",
                "class_name": STRETCH,
                "hparams": {
                    "low": 2.0,
                    "high": 98.0,
                    "per_channel": False,
                    "quantize_levels": 255,
                    "eps": 1e-6,
                },
            },
            {
                "name": "norm",
                "class_name": SCORE_RANGE,
                "hparams": {
                    "n_channels": 1,
                    "low": 1.0,
                    "high": 99.0,
                    "floor": True,
                    "fit_subsample": 4,
                    "max_fit_values": 4000000,
                    "seed": 0,
                    "eps": 1e-9,
                    "invert": False,
                },
            },
            {
                "name": "fuse",
                "class_name": SCORE_FUSION,
                "hparams": {"mode": "mean", "weight": 0.5},
            },
            {"name": "tap", "class_name": SCORE_FUSION, "hparams": {"mode": "max", "weight": 0.5}},
            {"name": "inter", "class_name": SCORE_INTERSECTION, "hparams": {}},
            {
                "name": "gate",
                "class_name": "cuvis_ai_patchcore.node.gate.FrameScoreGate",
                "hparams": {"threshold": 1.0, "topk_frac": 0.001},
            },
            {
                "name": "pc",
                "class_name": "cuvis_ai_patchcore.node.patchcore.PatchCoreDetector",
                "hparams": {},
            },
            {
                "name": "persist",
                "class_name": "cuvis_ai_patchcore.node.temporal.MaskPersistence",
                "hparams": {"radius_px": 40},
            },
        ],
        "connections": [
            {"source": "src.outputs.cube", "target": "sel.inputs.cube"},
            {"source": "sel.outputs.rgb_image", "target": "stretch.inputs.data"},
            {"source": "pc.outputs.scores", "target": "norm.inputs.scores"},
            {"source": "norm.outputs.normalized", "target": "fuse.inputs.b"},
            {"source": "pc.outputs.scores", "target": "fuse.inputs.a"},
            {"source": "fuse.outputs.scores", "target": "tap.inputs.a"},
            {"source": "fuse.outputs.scores", "target": "tap.inputs.b"},
            {"source": "fuse.outputs.scores", "target": "inter.inputs.a"},
            {"source": "norm.outputs.normalized", "target": "inter.inputs.b"},
            {"source": "fuse.outputs.scores", "target": "gate.inputs.scores"},
        ],
    }


def _state() -> dict:
    return {
        "sel": {
            "running_min": torch.zeros(4),
            "running_max": torch.ones(4),
            "_norm_frame_count": torch.tensor(3),
        },
        "stretch": {},
        "norm": {"lo": torch.tensor([0.25]), "hi": torch.tensor([2.5])},
        "fuse": {},
        "pc": {"memory": torch.arange(6.0)},
    }


def _migrate():
    report = Report(yaml_path=Path("p.yaml"))
    doc, state = migrate_doc(_doc(), _state(), report)
    return doc, state, report, {n["name"]: n for n in doc["nodes"]}


# ----- classes and hparams ------------------------------------------------------------------


def test_renamed_classes_keep_their_hparams():
    _, _, report, nodes = _migrate()
    gate = nodes["gate"]
    assert gate["class_name"] == RENAMED["cuvis_ai_patchcore.node.gate.FrameScoreGate"]
    assert gate["hparams"] == {"threshold": 1.0, "topk_frac": 0.001}
    assert "gate" in report.renamed


def test_score_range_becomes_the_pooled_statistical_normalizer():
    _, _, _, nodes = _migrate()
    norm = nodes["norm"]
    assert norm["class_name"] == PERCENTILE
    assert norm["hparams"] == {
        "n_channels": 1,
        "norm_mode": "statistical",
        "statistical_fit": "pooled",
        "quantile_low": 0.01,
        "quantile_high": 0.99,
        "clamp": "floor",
        "eps": 1e-9,
        "fit_subsample": 4,
        "max_fit_values": 4000000,
        "seed": 0,
        "invert": False,
    }
    PercentileNormalizer(**norm["hparams"])  # valid hparams


def test_stretch_becomes_the_per_frame_normalizer_with_the_source_channels():
    _, _, report, nodes = _migrate()
    hp = nodes["stretch"]["hparams"]
    assert nodes["stretch"]["class_name"] == PERCENTILE
    assert hp["n_channels"] == 4  # from the selector's fitted bounds
    assert hp["norm_mode"] == "per_frame" and hp["per_frame_quantiles"] and hp["joint_channels"]
    assert (hp["quantile_low"], hp["quantile_high"], hp["quantize_levels"]) == (0.02, 0.98, 255)
    PercentileNormalizer(**hp)
    assert "4 channels" in report.rewritten["stretch"]


def test_fusion_nodes_become_score_map_fusion():
    _, _, _, nodes = _migrate()
    assert nodes["fuse"]["class_name"] == SCORE_MAP_FUSION
    assert nodes["fuse"]["hparams"] == {"mode": "mean"}
    assert nodes["inter"]["hparams"] == {"mode": "min"}


def test_classes_without_a_rewrite_stay_and_are_reported():
    _, _, report, nodes = _migrate()
    assert nodes["pc"]["class_name"].startswith("cuvis_ai_patchcore")
    assert nodes["persist"]["class_name"].startswith("cuvis_ai_patchcore")
    assert "MaskPersistence" in report.kept["persist"] and "PatchCoreDetector" in report.kept["pc"]


# ----- connections and plugins --------------------------------------------------------------


def test_connections_a_before_b_and_the_renamed_input():
    doc, _, report, _ = _migrate()
    targets = [(c["source"], c["target"]) for c in doc["connections"]]
    assert ("pc.outputs.scores", "norm.inputs.data") in targets
    fuse_in = [s for s, t in targets if t == "fuse.inputs.scores"]
    assert fuse_in == [
        "pc.outputs.scores",
        "norm.outputs.normalized",
    ]  # a first, though b was listed first
    assert [s for s, t in targets if t == "tap.inputs.scores"] == ["fuse.outputs.scores"]
    assert any("tap: a and b came from" in n for n in report.notes)
    assert not any(t.endswith((".inputs.a", ".inputs.b")) for _, t in targets)


def test_builtin_plugin_is_added():
    doc, _, _, _ = _migrate()
    assert doc["plugins"][0] == "cuvis_ai_builtin"


def test_plugins_without_remaining_classes_are_dropped(tmp_path):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    for name, classes in {
        "patchcore": ["cuvis_ai_patchcore.node.patchcore.PatchCoreDetector"],
        "steervit": ["cuvis_ai_steervit.node.stretch.JointPercentileStretch"],
    }.items():
        (plugins / f"{name}.yaml").write_text(
            yaml.safe_dump({"name": name, "capabilities": [{"class_name": c} for c in classes]})
        )
    src = tmp_path / "p.yaml"
    src.write_text(yaml.safe_dump(_doc()))
    report = migrate_file(src, tmp_path / "out", plugins)
    doc = yaml.safe_load((tmp_path / "out" / "p.yaml").read_text())
    assert "steervit" not in doc["plugins"] and "patchcore" in doc["plugins"]
    assert "rfdetr_seg" in doc["plugins"]  # no manifest given: kept
    assert any("steervit dropped" in n for n in report.notes)


# ----- state ----------------------------------------------------------------------------------


def test_state_moves_lo_hi_and_adds_the_normalizer_buffers():
    _, state, _, nodes = _migrate()
    assert torch.equal(state["norm"]["running_min"], torch.tensor([0.25]))
    assert torch.equal(state["norm"]["running_max"], torch.tensor([2.5]))
    assert set(state["stretch"]) == {"running_min", "running_max", "_norm_frame_count"}
    assert torch.isnan(state["stretch"]["running_min"]).all()
    assert torch.equal(state["pc"]["memory"], torch.arange(6.0))  # untouched
    for name in ("norm", "stretch"):  # loads strictly into the built-in
        PercentileNormalizer(**nodes[name]["hparams"]).load_state_dict(state[name], strict=True)


def test_inputs_are_not_modified():
    doc, state = _doc(), _state()
    migrate_doc(doc, state, Report(yaml_path=Path("p.yaml")))
    assert doc == _doc() and set(state["norm"]) == {"lo", "hi"}


# ----- the arithmetic carries over ------------------------------------------------------------


def test_migrated_score_range_reproduces_the_former_formula():
    _, state, _, nodes = _migrate()
    node = PercentileNormalizer(**nodes["norm"]["hparams"])
    node.load_state_dict(state["norm"])
    x = torch.rand(2, 9, 7, 1, generator=torch.Generator().manual_seed(0)) * 5 - 1
    lo, hi = torch.tensor([0.25]), torch.tensor([2.5])
    former = ((x - lo) / (hi - lo).clamp_min(1e-9)).clamp_min(0.0)  # ScoreRangeNormalizer.forward
    assert torch.equal(node(data=x)["normalized"], former)


@pytest.mark.parametrize(
    ("mode", "former"),
    [("mean", lambda a, b: 0.5 * (a + b)), ("max", torch.maximum), ("min", torch.minimum)],
)
def test_migrated_fusion_reproduces_the_two_input_node(mode, former):
    g = torch.Generator().manual_seed(1)
    a, b = torch.rand(2, 8, 8, 1, generator=g), torch.rand(2, 8, 8, 1, generator=g)
    assert torch.equal(ScoreMapFusion(mode=mode)(scores=[a, b])["scores"], former(a, b))
    assert torch.equal(ScoreMapFusion(mode=mode)(scores=[a])["scores"], a)  # the identity tap


# ----- refusals and the CLI -------------------------------------------------------------------


def test_inexact_fusion_mode_is_refused():
    doc = _doc()
    doc["nodes"][4]["hparams"]["mode"] = "gmean"
    with pytest.raises(MigrationError, match="gmean"):
        migrate_doc(doc, None, Report(yaml_path=Path("p.yaml")))


def test_fusion_without_both_inputs_is_refused():
    doc = _doc()
    doc["connections"] = [c for c in doc["connections"] if c["target"] != "fuse.inputs.b"]
    with pytest.raises(MigrationError, match="a and b"):
        migrate_doc(doc, None, Report(yaml_path=Path("p.yaml")))


def test_a_guessed_stretch_channel_count_is_reported():
    """Without fitted bounds on the stretch's source the channel count is a guess; say so."""
    report = Report(yaml_path=Path("p.yaml"))
    doc, _ = migrate_doc(_doc(), None, report)
    stretch = next(n for n in doc["nodes"] if n["name"] == "stretch")
    assert stretch["hparams"]["n_channels"] == 3
    assert any(note.startswith("stretch: n_channels=3 guessed") for note in report.notes)
    _, _, fitted, _ = _migrate()
    assert not any("guessed" in note for note in fitted.notes)


def _write(folder: Path, doc: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    src = folder / "p.yaml"
    src.write_text(yaml.safe_dump(doc))
    return src


def test_cli_writes_the_pair_and_reports(tmp_path):
    src = _write(tmp_path, _doc())
    torch.save({"state_dict": _state(), "metadata": {"name": "p"}}, tmp_path / "p.pt")
    result = CliRunner().invoke(main, [str(src), "--out-dir", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    saved = torch.load(tmp_path / "out" / "p.pt", weights_only=True)
    assert saved["metadata"] == {"name": "p"} and "running_min" in saved["state_dict"]["norm"]
    assert "rewritten: norm" in result.output
    bad = _doc()
    bad["nodes"][4]["hparams"]["mode"] = "gmean"
    src.write_text(yaml.safe_dump(bad))
    result = CliRunner().invoke(main, [str(src), "--out-dir", str(tmp_path / "out2")])
    assert result.exit_code == 1
    assert "NOT migrated" in result.output


def test_cli_keeps_same_named_pipelines_of_different_folders_apart(tmp_path):
    """The catalog repeats file names across folders; each output keeps its folder."""
    other = _doc()
    other["nodes"][4]["hparams"]["mode"] = "max"
    a = _write(tmp_path / "catalog" / "best", _doc())
    b = _write(tmp_path / "catalog" / "best_80ms", other)
    out = tmp_path / "out"
    result = CliRunner().invoke(main, [str(a), str(b), "--out-dir", str(out)])
    assert result.exit_code == 0, result.output
    modes = {
        folder: next(
            n["hparams"]["mode"]
            for n in yaml.safe_load((out / folder / "p.yaml").read_text())["nodes"]
            if n["name"] == "fuse"
        )
        for folder in ("best", "best_80ms")
    }
    assert modes == {"best": "mean", "best_80ms": "max"}


def test_migrated_yaml_keeps_the_key_order(tmp_path):
    """A migrated preset differs from its source only where it was rewritten."""
    doc = {"metadata": {"name": "p"}, **_doc()}
    gate = next(n for n in doc["nodes"] if n["name"] == "gate")
    gate["hparams"] = {"topk_frac": 0.001, "threshold": 1.0}
    src = tmp_path / "p.yaml"
    src.write_text(yaml.safe_dump(doc, sort_keys=False))
    migrate_file(src, tmp_path / "out")
    out = yaml.safe_load((tmp_path / "out" / "p.yaml").read_text())
    assert list(out) == ["metadata", "plugins", "nodes", "connections"]
    gate = next(n for n in out["nodes"] if n["name"] == "gate")
    assert list(gate) == ["name", "class_name", "hparams"]
    assert list(gate["hparams"]) == ["topk_frac", "threshold"]
