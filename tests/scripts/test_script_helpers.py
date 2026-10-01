"""The small helpers of the CLIs and the gRPC workflow module, pinned before they were reshaped."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cuvis_ai.utils import grpc_workflow
from scripts.bump_plugin_pins import _extract_class_names, _manifest_node_set
from scripts.generate_node_catalog import _first_doc_line
from scripts.generate_node_port_stubs import _gather_nodes, _is_concrete_node, _module_all

pytestmark = pytest.mark.unit


# --- grpc_workflow.config_search_paths ---------------------------------------------------------


def test_config_search_paths_keeps_order_drops_duplicates_and_missing_dirs(tmp_path: Path) -> None:
    extra = tmp_path / "extra"
    extra.mkdir()
    paths = grpc_workflow.config_search_paths(
        [extra, tmp_path / "missing", str(extra), grpc_workflow.CONFIG_ROOT]
    )
    seeds = [str(p.resolve()) for p in [grpc_workflow.CONFIG_ROOT] if p.is_dir()]
    assert paths[0] == seeds[0]
    assert paths.count(str(extra.resolve())) == 1
    assert paths[-1] == str(extra.resolve())
    assert str((tmp_path / "missing").resolve()) not in paths
    assert len(paths) == len(set(paths))


# --- calibrate_thresholds: the device probe ---------------------------------------------------


def test_pipeline_device_takes_the_first_parameter_device_or_cpu() -> None:
    from scripts.calibrate_thresholds import _pipeline_device

    on_meta = SimpleNamespace(torch_layers=[nn.Identity(), nn.Linear(2, 2, device="meta")])
    assert _pipeline_device(on_meta) == torch.device("meta")
    assert _pipeline_device(SimpleNamespace(torch_layers=[nn.Identity()])) == torch.device("cpu")
    assert _pipeline_device(SimpleNamespace(torch_layers=[])) == torch.device("cpu")


# --- generate_node_catalog._first_doc_line ----------------------------------------------------


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        (None, ""),
        ("", ""),
        ("   \n\n  ", ""),
        ("Summary line.\n\nDetails.", "Summary line."),
        ("\n\n   Indented summary.  \nMore.", "Indented summary."),
    ],
)
def test_first_doc_line(doc: str | None, expected: str) -> None:
    assert _first_doc_line(doc) == expected


# --- generate_node_port_stubs ------------------------------------------------------------------


def test_is_concrete_node_filters_base_abstract_and_foreign_classes() -> None:
    from cuvis_ai.node.normalization import IdentityNormalizer, _NormalizerBase
    from cuvis_ai_core.node import Node

    module = "cuvis_ai.node.normalization"
    assert _is_concrete_node(IdentityNormalizer, module)
    assert _is_concrete_node(_NormalizerBase, module)  # not abstract, so it is listed
    assert not _is_concrete_node(Node, module)
    assert not _is_concrete_node(IdentityNormalizer, "cuvis_ai.node.other")
    assert not _is_concrete_node(int, module)


def test_gather_nodes_is_sorted_by_class_name() -> None:
    nodes = _gather_nodes("cuvis_ai.node.normalization")
    names = [cls.__name__ for cls in nodes]
    assert names == sorted(names)
    assert "IdentityNormalizer" in names and "SigmoidTransform" in names
    assert all(cls.__module__ == "cuvis_ai.node.normalization" for cls in nodes)


def test_module_all_prefers_the_module_all_and_sorts_it() -> None:
    import sys
    from types import ModuleType

    def _listed(block: str) -> list[str]:
        return [line.strip().strip('",') for line in block.splitlines() if line.startswith("    ")]

    with_all = ModuleType("stub_probe_with_all")
    with_all.__all__ = ["Zed", "Alpha", "Zed"]
    without_all = ModuleType("stub_probe_without_all")
    sys.modules["stub_probe_with_all"] = with_all
    sys.modules["stub_probe_without_all"] = without_all
    try:
        assert _listed(_module_all("stub_probe_with_all", ["Other"])) == ["Alpha", "Zed"]
        assert _listed(_module_all("stub_probe_without_all", ["B", "A", "B"])) == ["A", "B"]
    finally:
        del sys.modules["stub_probe_with_all"], sys.modules["stub_probe_without_all"]


# --- bump_plugin_pins: manifest node sets ----------------------------------------------------


def test_manifest_node_set_reads_capabilities_only() -> None:
    doc = {"capabilities": [{"class_name": "A"}, "B", {"kind": "data_module"}]}
    assert _manifest_node_set(doc) == {"A", "B"}
    assert _manifest_node_set({"name": "x"}) is None
    assert _manifest_node_set({"capabilities": []}) is None
    assert _extract_class_names("not a list") == set()
