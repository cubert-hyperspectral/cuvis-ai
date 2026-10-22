"""Every optional input port is omitted, not passed as None, when nothing feeds it.

The pipeline executor skips optional ports without a producer and calls ``forward`` with the
remaining inputs only, so a ``forward`` parameter behind an optional port needs a default or the
call raises ``TypeError`` at the first batch. The registry-wide check keeps every builtin node
honest about the ports it declares optional.
"""

import inspect
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from cuvis_ai_core.utils.node_registry import NodeRegistry

REGISTRY_PACKAGES = (
    "cuvis_ai.node",
    "cuvis_ai.node.anomaly",
    "cuvis_ai.node.deciders",
    "cuvis_ai.node.pretreatments",
    "cuvis_ai.node.unmixing",
    "cuvis_ai.node.clustering",
    "cuvis_ai.node.segmentation",
)


@contextmanager
def _isolated_builtin_registry() -> Iterator[None]:
    snapshot = dict(NodeRegistry._builtin_registry)  # noqa: SLF001
    try:
        NodeRegistry.clear()
        for package_name in REGISTRY_PACKAGES:
            NodeRegistry.auto_register_package(package_name)
        yield
    finally:
        NodeRegistry.clear()
        NodeRegistry._builtin_registry.update(snapshot)  # noqa: SLF001


def _optional_ports_without_default() -> list[str]:
    offenders: list[str] = []
    with _isolated_builtin_registry():
        for name in NodeRegistry.list_builtin_nodes():
            node_class = NodeRegistry.get_builtin_class(name)
            specs = getattr(node_class, "INPUT_SPECS", {}) or {}
            optional = [port for port, spec in specs.items() if getattr(spec, "optional", False)]
            if not optional:
                continue
            parameters = inspect.signature(node_class.forward).parameters
            accepts_var_kwargs = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
            )
            for port in optional:
                parameter = parameters.get(port)
                if parameter is None:
                    if not accepts_var_kwargs:
                        offenders.append(
                            f"{name}.forward has no parameter for optional port {port}"
                        )
                    continue
                if parameter.default is inspect.Parameter.empty:
                    offenders.append(f"{name}.forward({port}) has no default")
    return offenders


@pytest.mark.parametrize("offender", _optional_ports_without_default() or [None])
def test_optional_input_ports_have_defaults(offender):
    assert offender is None, offender
