"""Slow plugin runtime smoke checks using AdaCLIP selective manifest loading."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from cuvis_ai_core.node.node import Node
from cuvis_ai_core.utils.node_registry import NodeRegistry

pytestmark = [pytest.mark.integration, pytest.mark.slow]

ADACLIP_MANIFEST_PATH = Path("configs/plugins/adaclip.yaml")
PLUGIN_NAME = "adaclip"


def _instantiate_with_default_kwargs(node_cls: type[Node]) -> Node:
    """Instantiate a plugin node class using only defaulted constructor args."""
    signature = inspect.signature(node_cls.__init__)
    kwargs: dict[str, object] = {}

    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if parameter.default is inspect.Parameter.empty:
            raise AssertionError(
                f"{node_cls.__name__} has required constructor arg '{name}'. "
                "Plugin smoke tests need default-constructable plugin nodes."
            )
        kwargs[name] = parameter.default

    return node_cls(**kwargs)


def test_plugin_nodes_can_instantiate_and_move_to_cpu() -> None:
    registry = NodeRegistry()
    registry.load_plugins(ADACLIP_MANIFEST_PATH)
    assert set(registry.list_plugins()) == {PLUGIN_NAME}

    for class_path in registry.plugin_configs[PLUGIN_NAME].provides:
        node_cls = registry.get(class_path)
        node = _instantiate_with_default_kwargs(node_cls)
        assert isinstance(node, Node), f"{PLUGIN_NAME}: {class_path} is not a Node instance."

        moved = node.to("cpu")
        assert moved is node
