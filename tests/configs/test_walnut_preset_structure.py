"""Every walnut preset is a well-formed graph before it reaches a stand.

The checks need no plugin installs: each connection names nodes of the preset, a built-in node
(``cuvis_ai.node.*``) has the port a connection uses, and every node's class comes from a plugin
the preset lists (``cuvis_ai_builtin`` for the built-ins, else a manifest under
``cuvis_ai/configs/plugins`` whose capabilities name the class).
"""

from __future__ import annotations

import importlib
from functools import cache
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
PIPELINES = REPO / "cuvis_ai" / "configs" / "pipeline"
PLUGINS = REPO / "cuvis_ai" / "configs" / "plugins"
PRESETS = sorted(PIPELINES.glob("walnut_*/**/*.yaml"))


@cache
def _providers() -> dict[str, frozenset[str]]:
    found: dict[str, set[str]] = {}
    for manifest in PLUGINS.glob("*.yaml"):
        doc = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        for cap in doc.get("capabilities") or []:
            if "class_name" in cap:
                found.setdefault(cap["class_name"], set()).add(doc["name"])
    return {cls: frozenset(names) for cls, names in found.items()}


def _ports(node: dict) -> tuple[set[str], set[str]]:
    """Input and output port names of a built-in node, from the instance when it builds."""
    module, cls_name = node["class_name"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module), cls_name)
    try:
        obj = cls(name=node["name"], **(node.get("hparams") or {}))
    except Exception:  # noqa: BLE001 - a node that needs fitted state still has its class specs
        obj = cls
    return set(obj.INPUT_SPECS), set(obj.OUTPUT_SPECS)


def _problems(path: Path) -> list[str]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    nodes = {n["name"]: n for n in doc["nodes"]}
    listed = set(doc.get("plugins") or [])
    found = []
    for node in nodes.values():
        cls = node["class_name"]
        if cls.startswith("cuvis_ai.node."):
            if "cuvis_ai_builtin" not in listed:
                found.append(f"{node['name']} is built-in but cuvis_ai_builtin is not listed")
        elif not (_providers().get(cls, frozenset()) & listed):
            found.append(f"no listed plugin provides {cls}")
    for conn in doc.get("connections") or []:
        for end in (conn["source"], conn["target"]):
            name, direction, port = end.split(".", 2)
            if name not in nodes:
                found.append(f"{end} names a missing node")
                continue
            if nodes[name]["class_name"].startswith("cuvis_ai.node."):
                inputs, outputs = _ports(nodes[name])
                if port not in (outputs if direction == "outputs" else inputs):
                    found.append(f"{end} is not a port of {nodes[name]['class_name']}")
    return found


def test_the_walnut_catalog_is_there():
    assert len(PRESETS) >= 14


@pytest.mark.parametrize("path", PRESETS, ids=lambda p: p.relative_to(PIPELINES).as_posix())
def test_walnut_preset_is_a_well_formed_graph(path):
    assert _problems(path) == []


def test_a_dropped_node_is_reported(tmp_path):
    """The check is not vacuous: removing a node that a connection uses is caught."""
    source = next(p for p in PRESETS if "ShellFuse" in p.read_text(encoding="utf-8"))
    doc = yaml.safe_load(source.read_text(encoding="utf-8"))
    doc["nodes"] = [n for n in doc["nodes"] if n["name"] != "ShellFuse"]
    broken = tmp_path / source.name
    broken.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    assert any("ShellFuse" in p and "missing node" in p for p in _problems(broken))
