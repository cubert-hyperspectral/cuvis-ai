from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from loguru import logger

from cuvis_ai.node import Node
from cuvis_ai.pipeline.pipeline import CuvisPipeline
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.utils.types import ExecutionStage

NodeTypeResolver = Callable[[Node], str]

DEFAULT_STAGE_LABELS: dict[str, str] = {
    ExecutionStage.TRAIN.value: "Train",
    ExecutionStage.VAL.value: "Validation",
    ExecutionStage.TEST.value: "Test",
    ExecutionStage.INFERENCE.value: "Inference",
    ExecutionStage.ALWAYS.value: "All Stages",
}


class PipelineVisualizer:
    """Generate Graphviz and Mermaid representations of a CuvisPipeline."""

    def __init__(self, pipeline: CuvisPipeline) -> None:
        self.pipeline = pipeline
        self._graph = pipeline._graph

    def to_graphviz(
        self,
        *,
        graph_name: str | None = None,
        rankdir: str = "LR",
        node_shape: str = "box",
        include_node_class: bool = True,
        node_type_resolver: NodeTypeResolver | None = None,
        node_colors: Mapping[str, str] | None = None,
        default_node_color: str | None = "#f8f9fb",
        group_by_stage: bool = False,
        stage_labels: Mapping[str, str] | None = None,
        show_port_types: bool = False,
        show_execution_stage: bool = False,
        graph_attributes: Mapping[str, Any] | None = None,
        node_attributes: Mapping[str, Any] | None = None,
        edge_attributes: Mapping[str, Any] | None = None,
    ) -> str:
        """Return a DOT string describing the pipeline graph."""

        title = self._sanitize_identifier(graph_name or self.pipeline.name or "CuvisPipeline")
        lines: list[str] = [f"digraph {title} {{"]

        # Global graph defaults
        lines.append(f"    rankdir={rankdir};")
        lines.append(f"    node [shape={node_shape}];")

        if graph_attributes:
            lines.extend(self._format_graphviz_attributes(graph_attributes))
        if edge_attributes:
            attr_line = self._compose_attribute_list(edge_attributes)
            lines.append(f"    edge [{attr_line}];")

        node_entries: dict[Node, str] = {}
        node_type_resolver = node_type_resolver or (lambda node: node.__class__.__name__)
        node_colors = node_colors or {}
        stage_labels = {**DEFAULT_STAGE_LABELS, **(stage_labels or {})}
        node_type_lookup: dict[Node, str] = {}

        for node in self._graph.nodes:
            identifier = self._dot_identifier(node)
            stage_text = (
                self._format_execution_stage_text(node, stage_labels)
                if show_execution_stage
                else None
            )
            label = self._escape_label(
                self._format_node_label(
                    node,
                    include_node_class,
                    stage_text=stage_text,
                )
            )
            node_type = node_type_resolver(node)
            node_type_lookup[node] = node_type

            attrs: list[str] = [f'label="{label}"']
            fill = node_colors.get(node_type, default_node_color)
            if fill:
                attrs.append('style="filled"')
                attrs.append(f'fillcolor="{self._escape_label(str(fill))}"')

            if node_attributes:
                attrs.extend(self._format_inline_attributes(node_attributes))

            node_entries[node] = f'"{identifier}" [{", ".join(attrs)}];'

        if group_by_stage:
            stage_groups = self._group_nodes_by_stage()
            for stage_key, nodes in stage_groups.items():
                cluster_name = self._sanitize_identifier(stage_key, allow_dash=False)
                label = self._escape_label(stage_labels.get(stage_key, stage_key.title()))
                lines.append(f'    subgraph "cluster_{cluster_name}" {{')
                lines.append(f'        label="{label}";')
                for node in nodes:
                    lines.append(f"        {node_entries[node]}")
                lines.append("    }")
        else:
            for entry in node_entries.values():
                lines.append(f"    {entry}")

        for source, target, edge_data in self._graph.edges(data=True):
            edge_label = self._format_edge_label(
                source,
                target,
                edge_data,
                include_port_types=show_port_types,
            )
            src = self._dot_identifier(source)
            dst = self._dot_identifier(target)
            if edge_label:
                label = self._escape_label(edge_label)
                lines.append(f'    "{src}" -> "{dst}" [label="{label}"];')
            else:
                lines.append(f'    "{src}" -> "{dst}";')

        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(
        self,
        *,
        direction: str = "LR",
        include_node_class: bool = True,
        node_type_resolver: NodeTypeResolver | None = None,
        node_colors: Mapping[str, str] | None = None,
        group_by_stage: bool = False,
        stage_labels: Mapping[str, str] | None = None,
        show_port_types: bool = False,
        show_execution_stage: bool = False,
    ) -> str:
        """Return Mermaid flowchart syntax describing the pipeline graph."""

        lines = [f"flowchart {direction}"]
        node_type_resolver = node_type_resolver or (lambda node: node.__class__.__name__)
        node_colors = node_colors or {}
        stage_labels = {**DEFAULT_STAGE_LABELS, **(stage_labels or {})}
        node_classes: dict[str, str] = {}
        class_styles: dict[str, str] = {}

        node_defs: dict[Node, str] = {}
        for node in self._graph.nodes:
            identifier = self._mermaid_identifier(node)
            stage_text = (
                self._format_execution_stage_text(node, stage_labels)
                if show_execution_stage
                else None
            )
            label = self._escape_mermaid_label(
                self._format_node_label(
                    node,
                    include_node_class,
                    mermaid=True,
                    stage_text=stage_text,
                )
            )
            node_defs[node] = f"{identifier}[{label}]"

            node_type = node_type_resolver(node)
            class_name = self._sanitize_identifier(node_type, allow_dash=False) or "type"
            node_classes[identifier] = class_name

            color = node_colors.get(node_type)
            if color:
                class_styles[class_name] = color

        if group_by_stage:
            stage_groups = self._group_nodes_by_stage()
            for stage_key, nodes in stage_groups.items():
                label = stage_labels.get(stage_key, stage_key.title())
                lines.append(
                    f"    subgraph {self._sanitize_identifier(stage_key, allow_dash=False)}[{label}]"
                )
                lines.append(f"        direction {direction}")
                for node in nodes:
                    lines.append(f"        {node_defs[node]}")
                lines.append("    end")
        else:
            for definition in node_defs.values():
                lines.append(f"    {definition}")

        for source, target, edge_data in self._graph.edges(data=True):
            src = self._mermaid_identifier(source)
            dst = self._mermaid_identifier(target)
            edge_label = self._format_edge_label(
                source,
                target,
                edge_data,
                include_port_types=show_port_types,
                mermaid=True,
            )
            if edge_label:
                sanitized = self._sanitize_mermaid_pipe(edge_label)
                lines.append(f"    {src} -->|{sanitized}| {dst}")
            else:
                lines.append(f"    {src} --> {dst}")

        if class_styles:
            lines.append("    %% Node styling")
            for class_name, color in class_styles.items():
                lines.append(
                    f"    classDef {class_name} fill:{color},stroke:#333,stroke-width:1px;"
                )
            for node_id, class_name in node_classes.items():
                if class_name in class_styles:
                    lines.append(f"    class {node_id} {class_name};")

        return "\n".join(lines)

    def render_graphviz(
        self,
        output_path: str | Path,
        *,
        format: str = "png",
        rankdir: str = "LR",
        node_shape: str = "box",
        include_node_class: bool = True,
        engine: str = "dot",
        **graphviz_kwargs: Any,
    ) -> Path:
        """Render the Graphviz output to an image file."""

        try:
            from graphviz import Source
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "graphviz package is required for rendering. "
                "Install it with `pip install graphviz` and ensure Graphviz binaries are available."
            ) from exc

        dot_source = self.to_graphviz(
            rankdir=rankdir,
            node_shape=node_shape,
            include_node_class=include_node_class,
            **graphviz_kwargs,
        )

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        source = Source(
            dot_source,
            filename=path.stem,
            directory=str(path.parent),
            engine=engine,
        )
        rendered = Path(source.render(format=format, cleanup=True))
        logger.success("Graphviz visualization saved to {}", rendered)
        return rendered

    def render_mermaid(
        self,
        output_path: str | Path,
        *,
        direction: str = "LR",
        include_node_class: bool = True,
        wrap_markdown: bool = True,
        **mermaid_kwargs: Any,
    ) -> Path:
        """Write the Mermaid diagram to disk (defaults to markdown fenced block)."""

        mermaid_source = self.to_mermaid(
            direction=direction,
            include_node_class=include_node_class,
            **mermaid_kwargs,
        )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if wrap_markdown:
            content = f"```mermaid\n{mermaid_source}\n```"
        else:
            content = mermaid_source

        path.write_text(content, encoding="utf-8")
        logger.success("Mermaid visualization saved to {}", path)
        return path

    def _dot_identifier(self, node: Node) -> str:
        return self._sanitize_identifier(node.name)

    def _mermaid_identifier(self, node: Node) -> str:
        return self._sanitize_identifier(node.name, allow_dash=False)

    def _format_node_label(
        self,
        node: Node,
        include_node_class: bool,
        mermaid: bool = False,
        stage_text: str | None = None,
    ) -> str:
        parts: list[str] = [node.name]
        if include_node_class and node.__class__.__name__ != node.name:
            parts.append(node.__class__.__name__)
        if stage_text:
            parts.append(f"Stage: {stage_text}")
        separator = "<br/>" if mermaid else "\n"
        return separator.join(parts)

    def _format_edge_label(
        self,
        source: Node,
        target: Node,
        edge_data: dict,
        *,
        include_port_types: bool,
        mermaid: bool = False,
    ) -> str:
        from_port = edge_data.get("from_port") or ""
        to_port = edge_data.get("to_port") or ""

        from_spec = (
            self._resolve_port_spec(source, from_port, is_output=True)
            if include_port_types
            else None
        )
        to_spec = (
            self._resolve_port_spec(target, to_port, is_output=False)
            if include_port_types
            else None
        )

        left = self._format_port_segment(from_port, from_spec)
        right = self._format_port_segment(to_port, to_spec)

        if not (left or right):
            return ""

        connector = "-->" if mermaid else "->"
        return f"{left} {connector} {right}".strip()

    def _format_port_segment(self, port_name: str, spec: PortSpec | None) -> str:
        if not port_name:
            return ""
        if not spec:
            return port_name
        detail = self._format_port_spec(spec)
        return f"{port_name} ({detail})" if detail else port_name

    def _resolve_port_spec(
        self,
        node: Node,
        port_name: str | None,
        *,
        is_output: bool,
    ) -> PortSpec | None:
        if not port_name:
            return None
        ports = getattr(node, "_output_ports" if is_output else "_input_ports", None)
        if not ports:
            return None
        port = ports.get(port_name)
        if port is None:
            return None
        return getattr(port, "spec", None)

    def _format_port_spec(self, spec: PortSpec | None) -> str:
        if spec is None:
            return ""
        dtype = self._format_dtype(spec.dtype)
        shape = ", ".join(str(dim) for dim in spec.shape)
        if dtype and shape:
            return f"{dtype} [{shape}]"
        return dtype or shape

    @staticmethod
    def _format_dtype(dtype: Any) -> str:
        if hasattr(dtype, "__name__"):
            return dtype.__name__
        return str(dtype)

    @staticmethod
    def _sanitize_identifier(value: str, *, allow_dash: bool = True) -> str:
        safe = []
        for char in value:
            if char.isalnum() or char == "_":
                safe.append(char)
            elif allow_dash and char == "-":
                safe.append(char)
            else:
                safe.append("_")
        return "".join(safe) or "node"

    def _escape_label(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _escape_mermaid_label(self, value: str) -> str:
        return value.replace("]", ")")

    def _sanitize_mermaid_pipe(self, value: str) -> str:
        return value.replace("|", "/")

    def _group_nodes_by_stage(self) -> dict[str, list[Node]]:
        groups: dict[str, list[Node]] = defaultdict(list)
        for node in self._graph.nodes:
            groups[self._stage_bucket(node)].append(node)
        return groups

    def _stage_bucket(self, node: Node) -> str:
        stages = getattr(node, "execution_stages", None)
        if not stages:
            return ExecutionStage.ALWAYS.value
        normalized = {self._normalize_stage(stage) for stage in stages}
        normalized.discard(ExecutionStage.ALWAYS.value)
        if not normalized:
            return ExecutionStage.ALWAYS.value
        if len(normalized) > 1:
            return ExecutionStage.ALWAYS.value
        return next(iter(normalized))

    def _normalize_stage(self, stage: ExecutionStage | str) -> str:
        if isinstance(stage, ExecutionStage):
            return stage.value
        stage_value = str(stage).lower()
        if stage_value == ExecutionStage.VALIDATE.value:
            return ExecutionStage.VAL.value
        return stage_value

    def _format_graphviz_attributes(self, attrs: Mapping[str, Any]) -> list[str]:
        return [f"    {key}={self._quote_graphviz_value(value)};" for key, value in attrs.items()]

    def _compose_attribute_list(self, attrs: Mapping[str, Any]) -> str:
        return ", ".join(
            f"{key}={self._quote_graphviz_value(value)}" for key, value in attrs.items()
        )

    def _format_inline_attributes(self, attrs: Mapping[str, Any]) -> list[str]:
        return [f"{key}={self._quote_graphviz_value(value)}" for key, value in attrs.items()]

    def _quote_graphviz_value(self, value: Any) -> str:
        if isinstance(value, (int, float)):
            return str(value)
        if value is None:
            return '""'
        return f'"{self._escape_label(str(value))}"'

    def _format_execution_stage_text(
        self, node: Node, stage_labels: Mapping[str, str]
    ) -> str | None:
        stages = self._node_stage_values(node)
        if not stages:
            return None
        labels = sorted({stage_labels.get(stage, stage.title()) for stage in stages})
        if not labels:
            return None
        if labels == [stage_labels.get(ExecutionStage.ALWAYS.value, "All Stages")]:
            return labels[0]
        return " / ".join(labels)

    def _node_stage_values(self, node: Node) -> set[str]:
        raw_stages = getattr(node, "execution_stages", None)
        if not raw_stages:
            return {ExecutionStage.ALWAYS.value}
        normalized = {self._normalize_stage(stage) for stage in raw_stages if stage}
        normalized.discard(None)
        if not normalized or ExecutionStage.ALWAYS.value in normalized:
            return {ExecutionStage.ALWAYS.value}
        return normalized


def visualize_pipeline(
    pipeline: CuvisPipeline,
    *,
    format: str = "graphviz",
    output_path: str | Path | None = None,
    **kwargs,
) -> str | Path:
    """Convenience helper to generate or render pipeline visualizations."""

    visualizer = PipelineVisualizer(pipeline)
    format_key = format.lower()

    if format_key in {"graphviz", "dot", "dot_string"}:
        return visualizer.to_graphviz(**kwargs)
    if format_key in {"mermaid", "mermaid_string"}:
        return visualizer.to_mermaid(**kwargs)
    if format_key in {"render", "render_graphviz"}:
        if output_path is None:
            raise ValueError("output_path is required for render_graphviz formats.")
        return visualizer.render_graphviz(output_path, **kwargs)
    if format_key == "render_mermaid":
        if output_path is None:
            raise ValueError("output_path is required for render_mermaid format.")
        return visualizer.render_mermaid(output_path, **kwargs)

    raise ValueError(f"Unsupported visualization format: {format}")
