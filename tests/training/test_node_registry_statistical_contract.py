"""NodeRegistry-wide statistical initialization contract checks."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from cuvis_ai_core.node.node import Node
from cuvis_ai_core.utils.node_registry import NodeRegistry

pytestmark = pytest.mark.unit

EMPTY_INIT_ERRORS = (RuntimeError, ValueError, StopIteration)
REGISTRY_PACKAGES = ("cuvis_ai.node", "cuvis_ai.anomaly", "cuvis_ai.deciders")

REQUIRED_ARG_DEFAULTS: dict[str, object] = {
    "num_channels": 5,
    "in_channels": 5,
    "input_channels": 5,
    "output_channels": 3,
    "n_components": 3,
    "rep_dim": 4,
    "hidden": 8,
    "num_spectral_bands": 5,
    "n_select": 3,
    "k": 3,
    "channel": 0,
    "normal_class_ids": [0],
    "min_wavelength_nm": 450.0,
}

SUPERVISED_SELECTOR_CLASSES = {
    "SupervisedCIRSelector",
    "SupervisedWindowedSelector",
    "SupervisedFullSpectrumSelector",
}


@contextmanager
def _isolated_builtin_registry() -> Iterator[None]:
    """Temporarily rebuild builtin NodeRegistry entries for this test only."""
    snapshot = dict(NodeRegistry._builtin_registry)  # noqa: SLF001
    try:
        NodeRegistry.clear()
        for package_name in REGISTRY_PACKAGES:
            NodeRegistry.auto_register_package(package_name)
        yield
    finally:
        NodeRegistry.clear()
        NodeRegistry._builtin_registry.update(snapshot)  # noqa: SLF001


def _minimal_constructor_kwargs(node_cls: type[Node]) -> dict[str, object]:
    """Construct deterministic minimal kwargs for registry node instantiation."""
    kwargs: dict[str, object] = {}
    signature = inspect.signature(node_cls.__init__)

    for param_name, parameter in signature.parameters.items():
        if param_name == "self":
            continue
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if parameter.default is not inspect.Parameter.empty:
            continue

        if param_name in REQUIRED_ARG_DEFAULTS:
            kwargs[param_name] = REQUIRED_ARG_DEFAULTS[param_name]
            continue
        if "channels" in param_name:
            kwargs[param_name] = 5
            continue
        if param_name.startswith("n_"):
            kwargs[param_name] = 3
            continue

        raise AssertionError(
            f"Missing constructor heuristic for {node_cls.__name__}.{param_name}. "
            "Add a value in REQUIRED_ARG_DEFAULTS."
        )

    class_name = node_cls.__name__
    if class_name in SUPERVISED_SELECTOR_CLASSES:
        kwargs.setdefault("num_spectral_bands", 5)
    if class_name == "LearnableChannelMixer":
        kwargs["init_method"] = "pca"
    if class_name == "SoftChannelSelector":
        kwargs["init_method"] = "variance"
    if class_name == "MinMaxNormalizer":
        kwargs["use_running_stats"] = True

    return kwargs


def test_registry_requires_initial_fit_nodes_reject_empty_stream() -> None:
    """All constructable requires_initial_fit registry nodes must reject empty init."""
    requires_fit_node_names: list[str] = []

    with _isolated_builtin_registry():
        for class_name in NodeRegistry.list_builtin_nodes():
            node_cls = NodeRegistry.get_builtin_class(class_name)
            if inspect.isabstract(node_cls):
                continue

            node = node_cls(**_minimal_constructor_kwargs(node_cls))
            if not node.requires_initial_fit:
                continue

            requires_fit_node_names.append(class_name)
            with pytest.raises(EMPTY_INIT_ERRORS):
                node.statistical_initialization(iter(()))
            assert node._statistically_initialized is False, (
                f"{class_name} accepted empty initialization but must stay uninitialized."
            )

    assert requires_fit_node_names, "Expected at least one requires_initial_fit node in registry."
