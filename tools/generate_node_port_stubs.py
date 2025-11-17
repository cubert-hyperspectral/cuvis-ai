"""Generate type stubs that expose node port attributes to IDEs/type checkers.

Currently targeted modules are enumerated explicitly via TARGET_NODE_MODULES.
The script inspects each module for concrete Node subclasses and writes a
matching .pyi stub where every declared port appears as a typed attribute.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path

from cuvis_ai.node import Node

# For now we explicitly list node modules that should receive generated stubs.
# Extend this list whenever new modules under cuvis_ai.node define Node subclasses.
TARGET_NODE_MODULES = [
    "cuvis_ai.node.data",
    "cuvis_ai.node.labels",
    "cuvis_ai.node.losses",
    "cuvis_ai.node.metrics",
    "cuvis_ai.node.monitor",
    "cuvis_ai.node.normalization",
    "cuvis_ai.node.pca",
    "cuvis_ai.node.selector",
    "cuvis_ai.node.visualizations",
]

STUB_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "cuvis_ai" / "_stubs"


def _ensure_package_inits(target: Path) -> None:
    """Ensure every parent between stub root and file has an __init__.py."""
    current = target.parent
    while current.is_relative_to(STUB_PACKAGE_ROOT):
        current.mkdir(parents=True, exist_ok=True)
        init_file = current / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Auto-generated stub package."""\n')
        if current == STUB_PACKAGE_ROOT:
            break
        current = current.parent


STUB_HEADER = """\
from __future__ import annotations

from cuvis_ai.node.node import Node
from cuvis_ai.pipeline.ports import InputPort, OutputPort
"""


def _module_to_stub_path(module_name: str) -> Path:
    parts = module_name.split(".")
    if parts[0] != "cuvis_ai":
        raise ValueError(f"Unexpected module outside package: {module_name}")
    return STUB_PACKAGE_ROOT.joinpath(*parts[1:]).with_suffix(".pyi")


def _is_concrete_node(cls: type[Node], module_name: str) -> bool:
    if not inspect.isclass(cls):
        return False
    if not issubclass(cls, Node) or cls is Node:
        return False
    if inspect.isabstract(cls):
        return False
    return cls.__module__ == module_name


def _gather_nodes(module_name: str) -> list[type[Node]]:
    module = importlib.import_module(module_name)
    nodes: list[type[Node]] = []
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if _is_concrete_node(cls, module_name):
            nodes.append(cls)
    nodes.sort(key=lambda c: c.__name__)
    return nodes


@dataclass
class PortSets:
    inputs: list[str]
    outputs: list[str]
    both: list[str]


def _partition_ports(cls: type[Node]) -> PortSets:
    input_specs = list((getattr(cls, "INPUT_SPECS", {}) or {}).keys())
    output_specs = list((getattr(cls, "OUTPUT_SPECS", {}) or {}).keys())
    both = sorted(set(input_specs) & set(output_specs))
    unique_inputs = [name for name in input_specs if name not in both]
    unique_outputs = [name for name in output_specs if name not in both]
    return PortSets(unique_inputs, unique_outputs, both)


def _format_ports_class(class_name: str, port_names: list[str], port_type: str) -> str:
    lines = [f"class {class_name}:"]
    if port_names:
        lines.extend(f"    {name}: {port_type}" for name in port_names)
    else:
        lines.append("    ...")
    return "\n".join(lines) + "\n"


def _format_class_block(cls: type[Node]) -> str:
    port_sets = _partition_ports(cls)
    inputs_class = f"_{cls.__name__}Inputs"
    outputs_class = f"_{cls.__name__}Outputs"

    sections = [
        _format_ports_class(inputs_class, port_sets.inputs + port_sets.both, "InputPort"),
        _format_ports_class(outputs_class, port_sets.outputs + port_sets.both, "OutputPort"),
    ]

    class_lines = [f"class {cls.__name__}(Node):"]
    class_lines.append(f"    inputs: {inputs_class}")
    class_lines.append(f"    outputs: {outputs_class}")

    for name in port_sets.inputs:
        class_lines.append(f"    {name}: InputPort")
    for name in port_sets.outputs:
        class_lines.append(f"    {name}: OutputPort")

    if len(class_lines) == 3:  # only inputs/outputs lines present
        class_lines.append("    ...")

    sections.append("\n".join(class_lines) + "\n")
    return "\n".join(sections)


def _module_all(module_name: str, class_names: list[str]) -> str:
    module = importlib.import_module(module_name)
    exported = getattr(module, "__all__", None)
    if exported is None:
        exported = class_names
    # Ensure deterministic ordering
    exported = sorted(set(exported))
    inner = ",\n".join(f'    "{name}"' for name in exported)
    return f"__all__ = [\n{inner},\n]\n\n"


def generate_stub_for_module(module_name: str) -> None:
    stub_path = _module_to_stub_path(module_name)
    stub_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_package_inits(stub_path)

    nodes = _gather_nodes(module_name)
    class_blocks = "\n".join(_format_class_block(cls) for cls in nodes)
    all_block = _module_all(module_name, [cls.__name__ for cls in nodes])

    contents = (
        '"""Auto-generated by tools/generate_node_port_stubs.py - do not edit."""\n'
        + STUB_HEADER
        + "\n"
        + all_block
        + class_blocks
    )
    stub_path.write_text(contents, encoding="utf-8")
    print(f"Wrote stub for {module_name} -> {stub_path}")


def main() -> None:
    _ensure_package_inits(STUB_PACKAGE_ROOT / "__init__.py")
    for module_name in TARGET_NODE_MODULES:
        generate_stub_for_module(module_name)


if __name__ == "__main__":
    main()
