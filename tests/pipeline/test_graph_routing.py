"""
Test suite for CuvisPipeline.forward() port-based routing.

Converted from test_executor_routing.py to use pipeline.forward() instead of
deprecated MemoryExecutor. Tests ensure proper data routing through typed
ports, gradient preservation, and multiple entry inputs.
"""

from __future__ import annotations

import torch

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.pipeline.ports import PortSpec
from cuvis_ai_core.utils.types import ExecutionStage


class TestGraphBasicRouting:
    """Test basic data routing through ports using pipeline.forward()."""

    def test_simple_linear_pipeline(self) -> None:
        """Data should propagate through a simple A -> B pipeline."""

        class DoubleNode(Node):
            INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, x, **kwargs):
                return {"y": x * 2}

        class AddOneNode(Node):
            INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, x, **kwargs):
                return {"y": x + 1}

        pipeline = CuvisPipeline("linear")
        n1 = DoubleNode()
        n2 = AddOneNode()

        pipeline.connect(n1.outputs.y, n2.x)

        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        batch = {"x": input_tensor}
        outputs = pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)

        expected = (input_tensor * 2) + 1
        torch.testing.assert_close(outputs[(n2.name, "y")], expected)
        # Intermediate outputs should also be exposed
        assert (n1.name, "y") in outputs

    def test_graph_returns_all_outputs(self) -> None:
        """Graph should expose every produced port value."""

        class SourceNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {
                "out": PortSpec(dtype=torch.float32, shape=()),
                "meta": PortSpec(dtype=torch.float32, shape=()),
            }

            def forward(self, **kwargs):
                return {"out": torch.tensor(1.0), "meta": torch.tensor(5.0)}

            def load(self, params, serial_dir) -> None:
                return None

        class ConsumerNode(Node):
            INPUT_SPECS = {
                "inp": PortSpec(dtype=torch.float32, shape=()),
                "meta": PortSpec(dtype=torch.float32, shape=()),
            }
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=())}

            def forward(self, inp, meta, **kwargs):
                return {"out": inp + meta}

        pipeline = CuvisPipeline("fanout")
        src = SourceNode()
        dst = ConsumerNode()

        pipeline.connect((src.outputs.out, dst.inp), (src.outputs.meta, dst.meta))

        outputs = pipeline.forward(batch={}, stage=ExecutionStage.INFERENCE)

        assert outputs[(src.name, "out")].item() == 1.0
        assert outputs[(src.name, "meta")].item() == 5.0
        assert outputs[(dst.name, "out")].item() == 6.0


class TestGraphGradientFlow:
    """Ensure gradients are preserved through the graph."""

    def test_gradients_flow_through_pipeline(self) -> None:
        """Gradients should backpropagate across multiple nodes."""

        class ScaleNode(Node):
            INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(2.0))

            def forward(self, x, **kwargs):
                return {"y": x * self.scale}

            def load(self, params, serial_dir) -> None:
                return None

        class BiasNode(Node):
            INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}

            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, x, **kwargs):
                return {"y": x + self.bias}

        pipeline = CuvisPipeline("grad")
        scale = ScaleNode()
        bias = BiasNode()

        pipeline.connect(scale.outputs.y, bias.x)

        entry = torch.tensor([1.5, -0.5], requires_grad=True)
        batch = {"x": entry}
        outputs = pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)
        final = outputs[(bias.name, "y")].sum()
        final.backward()

        assert entry.grad is not None
        assert scale.scale.grad is not None
        assert bias.bias.grad is not None


class TestGraphMultipleConnections:
    """Test routing with branching connection patterns."""

    def test_fan_out_and_fan_in(self) -> None:
        """Graph should handle branching graphs."""

        class SplitNode(Node):
            INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {
                "left": PortSpec(dtype=torch.float32, shape=(-1,)),
                "right": PortSpec(dtype=torch.float32, shape=(-1,)),
            }

            def forward(self, x, **kwargs):
                return {"left": x, "right": -x}

        class LeftNode(Node):
            INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, x, **kwargs):
                return {"out": x * 2}

        class RightNode(Node):
            INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, x, **kwargs):
                return {"out": x + 3}

        class MergeNode(Node):
            INPUT_SPECS = {
                "left": PortSpec(dtype=torch.float32, shape=(-1,)),
                "right": PortSpec(dtype=torch.float32, shape=(-1,)),
            }
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, left, right, **kwargs):
                return {"out": left + right}

        pipeline = CuvisPipeline("branch")
        split = SplitNode()
        left = LeftNode()
        right = RightNode()
        merge = MergeNode()

        pipeline.connect(
            (split.left, left.x),
            (split.right, right.x),
            (left.outputs.out, merge.left),
            (right.outputs.out, merge.right),
        )

        inp = torch.tensor([1.0, 2.0])
        batch = {"x": inp}
        outputs = pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)

        expected_left = inp * 2
        expected_right = (-inp) + 3
        expected_merge = expected_left + expected_right

        torch.testing.assert_close(outputs[(left.name, "out")], expected_left)
        torch.testing.assert_close(outputs[(right.name, "out")], expected_right)
        torch.testing.assert_close(outputs[(merge.name, "out")], expected_merge)


class TestGraphEntryInputs:
    """Test entry input specification handling."""

    def test_multiple_entry_nodes(self) -> None:
        """Graph should support multiple entry points from batch."""

        class SourceA(Node):
            INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, x, **kwargs):
                return {"out": x}

        class SourceB(Node):
            INPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, y, **kwargs):
                return {"out": y * 3}

        class Combiner(Node):
            INPUT_SPECS = {
                "a": PortSpec(dtype=torch.float32, shape=(-1,)),
                "b": PortSpec(dtype=torch.float32, shape=(-1,)),
            }
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, a, b, **kwargs):
                return {"out": a + b}

            def load(self, params, serial_dir) -> None:
                return None

        pipeline = CuvisPipeline("multi-entry")
        a = SourceA()
        b = SourceB()
        c = Combiner()

        pipeline.connect((a.outputs.out, c.a), (b.outputs.out, c.b))

        batch = {
            "x": torch.tensor([1.0]),
            "y": torch.tensor([2.0]),
        }

        outputs = pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)
        torch.testing.assert_close(outputs[(c.name, "out")], torch.tensor([1.0 + (2.0 * 3.0)]))


class TestGraphPartialExecution:
    """Test partial graph execution with upto_node."""

    def test_upto_node_stops_execution(self) -> None:
        """Execution should stop before upto_node."""

        class Node1(Node):
            INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, x, **kwargs):
                return {"y": x * 2}

        class Node2(Node):
            INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, x, **kwargs):
                return {"y": x + 1}

        class Node3(Node):
            INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, x, **kwargs):
                return {"y": x * 3}

        pipeline = CuvisPipeline("partial")
        n1 = Node1()
        n2 = Node2()
        n3 = Node3()

        pipeline.connect((n1.outputs.y, n2.x), (n2.outputs.y, n3.x))

        batch = {"x": torch.tensor([1.0, 2.0])}
        outputs = pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE, upto_node=n2)

        # Only n1 should execute (ancestors of n2)
        assert (n1.name, "y") in outputs
        # n2 should NOT execute (upto_node is exclusive)
        assert (n2.name, "y") not in outputs
        # n3 should definitely NOT execute
        assert (n3.name, "y") not in outputs


class TestGraphStageFiltering:
    """Test stage-aware node execution filtering."""

    def test_inference_stage_filters_correctly(self) -> None:
        """Nodes should execute or skip based on stage."""

        class AlwaysNode(Node):
            INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, x, **kwargs):
                return {"y": x * 2}

        class TrainOnlyNode(Node):
            INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}

            def __init__(self):
                super().__init__(
                    execution_stages={ExecutionStage.TRAIN}
                )  # Only executes in train stage

            def forward(self, x, **kwargs):
                return {"y": x + 1}

        pipeline = CuvisPipeline("stage_test")
        always = AlwaysNode()
        train_only = TrainOnlyNode()

        pipeline.connect(always.outputs.y, train_only.x)

        batch = {"x": torch.tensor([1.0, 2.0])}

        # In inference stage, train_only should not execute
        outputs_inf = pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)
        assert (always.name, "y") in outputs_inf
        assert (train_only.name, "y") not in outputs_inf

        # In train stage, both should execute
        outputs_train = pipeline.forward(batch=batch, stage=ExecutionStage.TRAIN)
        assert (always.name, "y") in outputs_train
        assert (train_only.name, "y") in outputs_train
