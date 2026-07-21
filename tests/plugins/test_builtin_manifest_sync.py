"""Guard the built-in manifest's data-node port sets against the live classes.

``cuvis_ai_builtin.yaml`` is hand-maintained, and the docs catalog skips it in
favour of live introspection, so nothing else checks it against the classes it
mirrors. This asserts the anomaly data nodes' declared ports match the live
``INPUT_SPECS`` / ``OUTPUT_SPECS`` key sets, catching a manifest that drifts
behind a class change (e.g. a newly added port).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from cuvis_ai_schemas.plugin import load_plugin_manifest

pytestmark = pytest.mark.unit

BUILTIN_MANIFEST_PATH = Path("cuvis_ai/configs/plugins/cuvis_ai_builtin.yaml")


def _manifest_capability(class_name: str):
    """Return the built-in manifest capability entry for ``class_name``."""
    manifest = load_plugin_manifest(BUILTIN_MANIFEST_PATH)
    for cap in manifest.capabilities:
        if cap.class_name == class_name:
            return cap
    raise AssertionError(f"{class_name} not found in {BUILTIN_MANIFEST_PATH}")


@pytest.mark.parametrize(
    "class_name",
    [
        "cuvis_ai.node.data.AnomalyDataNode",
        "cuvis_ai.node.data.LentilsAnomalyDataNode",
    ],
)
def test_anomaly_data_node_ports_match_live_class(class_name: str) -> None:
    """The manifest's input/output port names match the live class introspection."""
    from cuvis_ai.node import data as data_module

    cls = getattr(data_module, class_name.rsplit(".", 1)[1])
    cap = _manifest_capability(class_name)

    assert set(cap.input_specs) == set(cls.INPUT_SPECS), (
        f"{class_name}: manifest input ports out of sync with the live class"
    )
    assert set(cap.output_specs) == set(cls.OUTPUT_SPECS), (
        f"{class_name}: manifest output ports out of sync with the live class"
    )
    # class_mask is the port this guard was added for; assert it explicitly.
    assert "class_mask" in cap.input_specs and "class_mask" in cap.output_specs
