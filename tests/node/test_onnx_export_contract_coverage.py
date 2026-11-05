"""The ONNX export contract covers every ONNX-tagged node the package exports.

The contract test enumerates the registry, and ``NodeRegistry.auto_register_package`` scans one
package folder at a time, so a node in a subpackage (``pretreatments``, ``anomaly``,
``deciders``) could carry ``NodeTag.ONNX`` without the weekly sweep ever exporting it. This
check runs on every PR and refuses that.
"""

from __future__ import annotations

import inspect

import pytest
from cuvis_ai_schemas.enums import NodeTag

import cuvis_ai.node as node_package
from cuvis_ai_core.node import Node
from tests.node.test_onnx_export_contract import _ONNX_NODE_NAMES

pytestmark = pytest.mark.unit


def _onnx_tagged_exports() -> set[str]:
    """Every concrete ``Node`` subclass in ``cuvis_ai.node.__all__`` that carries the tag."""
    names: set[str] = set()
    for name in node_package.__all__:
        obj = getattr(node_package, name)
        if (
            inspect.isclass(obj)
            and issubclass(obj, Node)
            and not inspect.isabstract(obj)
            and NodeTag.ONNX in obj.get_tags()
        ):
            names.add(name)
    return names


def test_contract_covers_every_onnx_tagged_export() -> None:
    """A node cannot carry ``NodeTag.ONNX`` without the contract exporting it."""
    tagged = _onnx_tagged_exports()
    assert tagged, "no ONNX-tagged node exported by cuvis_ai.node"
    missing = tagged - set(_ONNX_NODE_NAMES)
    assert not missing, f"ONNX-tagged nodes the contract never exports: {sorted(missing)}"
