from __future__ import annotations

import torch

from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.pipeline.ports import PortSpec
from cuvis_ai_core.pipeline.visualizer import PipelineVisualizer
from cuvis_ai_core.utils.types import ExecutionStage


class ProducerNode(Node):
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {
        "data": PortSpec(torch.Tensor, (-1, 3), "feature tensor"),
    }

    def forward(self, **inputs):
        return {"data": torch.zeros(1, 3)}


class ConsumerNode(Node):
    INPUT_SPECS = {
        "data": PortSpec(torch.Tensor, (-1, 3)),
    }
    OUTPUT_SPECS: dict[str, PortSpec] = {}

    def forward(self, **inputs):
        return {}


def _build_pipeline() -> CuvisPipeline:
    pipeline = CuvisPipeline("phase3")
    source = ProducerNode(name="source", execution_stages={ExecutionStage.ALWAYS})
    sink = ConsumerNode(name="sink", execution_stages={ExecutionStage.TRAIN})
    pipeline.connect(source.outputs.data, sink.inputs.data)
    return pipeline, source, sink


def test_graphviz_supports_phase3_styling():
    pipeline, source, sink = _build_pipeline()
    visualizer = PipelineVisualizer(pipeline)

    dot = visualizer.to_graphviz(
        node_type_resolver=lambda node: "producer"
        if isinstance(node, ProducerNode)
        else "consumer",
        node_colors={"producer": "#cfe2ff", "consumer": "#fde2e1"},
        group_by_stage=True,
        stage_labels={ExecutionStage.TRAIN.value: "Training"},
        show_port_types=True,
        graph_attributes={"bgcolor": "transparent"},
        node_attributes={"fontname": "Helvetica"},
        show_execution_stage=True,
    )

    assert 'fillcolor="#cfe2ff"' in dot
    assert 'fillcolor="#fde2e1"' in dot
    assert 'subgraph "cluster_train"' in dot
    assert 'label="Training"' in dot
    assert "data (Tensor [-1, 3]) -> data (Tensor [-1, 3])" in dot
    assert 'bgcolor="transparent"' in dot
    assert 'fontname="Helvetica"' in dot
    assert "Stage: All Stages" in dot
    assert "Stage: Training" in dot


def test_mermaid_supports_phase3_features():
    pipeline, source, sink = _build_pipeline()
    visualizer = PipelineVisualizer(pipeline)

    mermaid = visualizer.to_mermaid(
        direction="TB",
        node_type_resolver=lambda node: "producer"
        if isinstance(node, ProducerNode)
        else "consumer",
        node_colors={"producer": "#cfe2ff"},
        group_by_stage=True,
        stage_labels={ExecutionStage.TRAIN.value: "Training"},
        show_port_types=True,
        show_execution_stage=True,
    )

    assert "subgraph train[Training]" in mermaid
    assert "direction TB" in mermaid
    assert "classDef producer fill:#cfe2ff" in mermaid
    assert "class source" in mermaid
    assert "data (Tensor [-1, 3]) -->" in mermaid
    assert "Stage: All Stages" in mermaid
    assert "Stage: Training" in mermaid


def test_pipeline_visualize_method_proxy():
    pipeline, *_ = _build_pipeline()

    dot = pipeline.visualize(format="graphviz")
    mermaid = pipeline.visualize(format="mermaid")

    assert dot.startswith("digraph")
    assert mermaid.startswith("flowchart")


def test_pipeline_visualize_can_render_to_file(tmp_path):
    pipeline, *_ = _build_pipeline()
    output_svg = tmp_path / "viz"
    output_md = tmp_path / "viz.md"

    # Rendering requires graphviz dependency; skip gracefully if unavailable
    try:
        path = pipeline.visualize(
            format="render_graphviz",
            output_path=output_svg,
            show_execution_stage=True,
        )
        assert path.is_file()
    except RuntimeError as exc:
        if "graphviz package is required" not in str(exc):
            raise

    path_md = pipeline.visualize(
        format="render_mermaid", output_path=output_md, wrap_markdown=False
    )
    assert path_md.is_file()
    assert path_md.read_text().startswith("flowchart")
