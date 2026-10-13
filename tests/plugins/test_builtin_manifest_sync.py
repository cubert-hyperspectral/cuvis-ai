"""Guard the built-in manifest's port declarations against the live classes.

``cuvis_ai_builtin.yaml`` is hand-maintained, and the docs catalog skips it in
favour of live introspection, so nothing else checks it against the classes it
mirrors. This asserts the anomaly data nodes' declared ports match the live
``INPUT_SPECS`` / ``OUTPUT_SPECS`` key sets, catching a manifest that drifts
behind a class change (e.g. a newly added port), and that every channel
selector's port names and descriptions equal the live ``PortSpec`` entries,
catching a description that changes on a base class without the manifest. The
other direction is guarded too: every public ``Node`` subclass defined in
``cuvis_ai.node.channel_selector`` has a manifest entry, catching a new selector
that ships unlisted.
"""

from __future__ import annotations

import inspect
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


def _channel_selector_class_names() -> list[str]:
    manifest = load_plugin_manifest(BUILTIN_MANIFEST_PATH)
    prefix = "cuvis_ai.node.channel_selector."
    return [cap.class_name for cap in manifest.capabilities if cap.class_name.startswith(prefix)]


def _public_node_classes() -> set[str]:
    """Every public ``Node`` subclass defined in ``cuvis_ai.node.channel_selector``."""
    from cuvis_ai.node import channel_selector
    from cuvis_ai_core.node import Node

    return {
        name
        for name, obj in vars(channel_selector).items()
        if inspect.isclass(obj)
        and issubclass(obj, Node)
        and obj.__module__ == channel_selector.__name__
        and not name.startswith("_")
    }


def _classes_without_manifest_entry() -> set[str]:
    listed = {name.rsplit(".", 1)[1] for name in _channel_selector_class_names()}
    return _public_node_classes() - listed


@pytest.mark.parametrize(
    "class_name", _channel_selector_class_names(), ids=lambda name: name.rsplit(".", 1)[1]
)
def test_channel_selector_ports_match_live_class(class_name: str) -> None:
    """Port names and descriptions of every channel selector equal the live PortSpecs.

    The selectors inherit most ports from ``ChannelSelectorBase``; a description that
    changes on the base reaches every subclass that does not override it, so the manifest
    text of each entry is checked, not only the base's.
    """
    from cuvis_ai.node import channel_selector

    cls = getattr(channel_selector, class_name.rsplit(".", 1)[1])
    cap = _manifest_capability(class_name)

    for side, live in (("input_specs", cls.INPUT_SPECS), ("output_specs", cls.OUTPUT_SPECS)):
        declared = getattr(cap, side)
        assert set(declared) == set(live), f"{class_name}: manifest {side} port names out of sync"
        for port, spec in declared.items():
            assert spec.description == live[port].description, (
                f"{class_name}.{port}: manifest description out of sync with the live PortSpec"
            )


def test_every_public_channel_selector_class_has_a_manifest_entry() -> None:
    """A new public selector class cannot ship without its manifest entry."""
    assert _classes_without_manifest_entry() == set()


def test_the_guard_reports_a_public_class_without_an_entry(monkeypatch) -> None:
    from cuvis_ai.node import channel_selector

    extra = type(
        "ExtraSelector",
        (channel_selector.ChannelSelectorBase,),
        {"__module__": channel_selector.__name__},
    )
    monkeypatch.setattr(channel_selector, "ExtraSelector", extra, raising=False)
    assert _classes_without_manifest_entry() == {"ExtraSelector"}
