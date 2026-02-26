"""Fast plugin compliance checks based on AdaCLIP selective manifest loading."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from cuvis_ai_core.node.node import Node
from cuvis_ai_core.utils.node_registry import NodeRegistry

pytestmark = pytest.mark.unit

ADACLIP_MANIFEST_PATH = Path("configs/plugins/adaclip.yaml")
PLUGIN_NAME = "adaclip"


def test_plugin_manifest_loads() -> None:
    assert ADACLIP_MANIFEST_PATH.exists(), f"Missing plugin manifest: {ADACLIP_MANIFEST_PATH}"


def test_plugin_classes_resolve_and_match_node_contract() -> None:
    registry = NodeRegistry()
    registry.load_plugins(ADACLIP_MANIFEST_PATH)

    assert set(registry.list_plugins()) == {PLUGIN_NAME}

    for class_path in registry.plugin_configs[PLUGIN_NAME].provides:
        node_cls = registry.get(class_path)
        class_name = class_path.rsplit(".", 1)[1]

        assert inspect.isclass(node_cls), (
            f"{PLUGIN_NAME}: '{class_path}' did not resolve to a class."
        )
        assert issubclass(node_cls, Node), f"{PLUGIN_NAME}: '{class_path}' must inherit from Node."
        assert registry.get(class_name) is node_cls
        assert class_path in registry.plugin_class_map

        input_specs = getattr(node_cls, "INPUT_SPECS", None)
        output_specs = getattr(node_cls, "OUTPUT_SPECS", None)
        assert isinstance(input_specs, dict), (
            f"{PLUGIN_NAME}: '{class_path}' missing INPUT_SPECS dict."
        )
        assert isinstance(output_specs, dict), (
            f"{PLUGIN_NAME}: '{class_path}' missing OUTPUT_SPECS dict."
        )
        assert callable(getattr(node_cls, "forward", None)), (
            f"{PLUGIN_NAME}: '{class_path}' must define forward()."
        )
