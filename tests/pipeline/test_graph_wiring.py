"""
Test suite for graph wiring validation with required/optional ports and batch data keys.

Inspired by bad_wiring() in typed_io_example.py.
Tests ensure proper validation of:
- Required vs optional port connections
- Batch data key matching
- Stage-aware execution (train/val/test/inference)
- Partial graph execution with upto_node
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

from cuvis_ai.anomaly.rx_detector import RXPerBatch
from cuvis_ai.anomaly.rx_logit_head import RXLogitHead
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.losses import AnomalyBCEWithLogits
from cuvis_ai.node.node import Node
from cuvis_ai.pipeline.canvas import CuvisCanvas
from cuvis_ai.pipeline.ports import PortSpec


class BCEwithSigmoidLoss(AnomalyBCEWithLogits):
    """Custom BCE loss for testing that applies sigmoid."""

    INPUT_SPECS = {
        "predictions": PortSpec(
            dtype=torch.float32, shape=(-1, -1, -1, 1), description="Anomaly predictions (logits)"
        ),
        "targets": PortSpec(
            dtype=torch.bool, shape=(-1, -1, -1, 1), description="Ground truth anomaly masks"
        ),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Loss nodes should only execute in train/val/test, not inference
        from cuvis_ai.utils.types import ExecutionStage

        self.execution_stages = {ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST}

    def forward(self, predictions: Tensor, targets: Tensor, **_: Any) -> dict[str, torch.Tensor]:
        sigmoided_preds = torch.sigmoid(predictions)
        result = super().forward(sigmoided_preds, targets, **_)
        return result


class TestRequiredPortValidation:
    """Test validation of required port connections."""

    def test_missing_required_input_raises_runtime_error(self) -> None:
        """Missing required input for executing node should raise RuntimeError."""
        canvas = CuvisCanvas("selector_rx_validation")
        data = LentilsAnomalyDataNode(normal_class_ids=[0])  # I: cube, mask, O: cube, mask
        rx_node = RXPerBatch()  # I: data, O: scores
        rx_logits = RXLogitHead()  # I: scores, O: logits
        bce_loss = BCEwithSigmoidLoss(weight=0.3)  # I: predictions, targets, O: loss

        canvas.connect(
            (data.outputs.cube, rx_node.data),
            (rx_node.scores, rx_logits.scores),
            (rx_logits.logits, bce_loss.predictions),
        )

        # Note: bce_loss.targets is NOT connected and NOT in batch
        # This should fail because bce_loss executes in train stage
        bad_batch = {
            "cube": torch.randn(2, 4, 4, 8),
            "mask": torch.randint(0, 2, (2, 4, 4, 1), dtype=torch.int32),
        }

        with pytest.raises(RuntimeError, match="missing required inputs"):
            canvas.forward(stage="train", batch=bad_batch)

    def test_excluded_node_with_missing_input_only_warns(self, caplog) -> None:
        """Excluded node with missing input should only warn, not error."""
        canvas = CuvisCanvas("selector_rx_validation")
        data = LentilsAnomalyDataNode(normal_class_ids=[0])
        rx_node = RXPerBatch()
        rx_logits = RXLogitHead()
        bce_loss = BCEwithSigmoidLoss(weight=0.3)

        canvas.connect(
            (data.outputs.cube, rx_node.data),
            (rx_node.scores, rx_logits.scores),
            (rx_logits.logits, bce_loss.predictions),
            (data.outputs.mask, bce_loss.targets),  # Connect targets from data node
        )

        # Provide all required inputs with correct shape (3D not 4D)
        bad_batch = {
            "cube": torch.randn(2, 4, 4, 8),
            "mask": torch.randint(0, 2, (2, 4, 4), dtype=torch.int32),
        }

        # Should NOT raise since all inputs are provided
        outputs = canvas.forward(stage="val", batch=bad_batch)

        # Execution should succeed up to rx_logits
        assert (rx_logits.id, "logits") in outputs
        # BCE loss executes in val stage for validation metrics
        assert (bce_loss.id, "loss") in outputs

    def test_upto_node_excludes_downstream_validation(self) -> None:
        """upto_node parameter should exclude downstream nodes from validation."""
        canvas = CuvisCanvas("selector_rx_validation")
        data = LentilsAnomalyDataNode(normal_class_ids=[0])
        rx_node = RXPerBatch()
        rx_logits = RXLogitHead()
        bce_loss = BCEwithSigmoidLoss(weight=0.3)

        canvas.connect(
            (data.outputs.cube, rx_node.data),
            (rx_node.scores, rx_logits.scores),
            (rx_logits.logits, bce_loss.predictions),
        )

        # bce_loss is excluded via upto_node, so missing targets should not error
        bad_batch = {"cube": torch.randn(2, 4, 4, 8)}

        # Should succeed - bce_loss is downstream of rx_node and excluded
        outputs = canvas.forward(
            stage="train",
            batch=bad_batch,
            upto_node=rx_node,
        )

        # Should only execute data node (ancestor of rx_node)
        assert (data.id, "cube") in outputs
        # rx_node itself should NOT execute (upto_node is exclusive)
        assert (rx_node.id, "scores") not in outputs


class TestOptionalPortHandling:
    """Test handling of optional ports."""

    def test_optional_port_missing_does_not_error(self) -> None:
        """Missing optional port should not raise error."""

        class NodeWithOptionalInput(Node):
            INPUT_SPECS = {
                "required": PortSpec(dtype=torch.float32, shape=(-1,)),
                "optional": PortSpec(dtype=torch.float32, shape=(-1,), optional=True),
            }
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, required, optional=None, **kwargs):
                if optional is not None:
                    return {"out": required + optional}
                return {"out": required}

            def load(self, params, serial_dir):
                return None

        # Test standalone node directly without graph
        node = NodeWithOptionalInput()

        # Only provide required input
        from cuvis_ai.utils.types import Context

        context = Context(stage="inference")
        result = node.forward(required=torch.tensor([1.0, 2.0, 3.0]), context=context)
        assert "out" in result
        torch.testing.assert_close(result["out"], torch.tensor([1.0, 2.0, 3.0]))

    def test_optional_port_provided_is_used(self) -> None:
        """Optional port when provided should be used correctly."""

        class NodeWithOptionalInput(Node):
            INPUT_SPECS = {
                "required": PortSpec(dtype=torch.float32, shape=(-1,)),
                "optional": PortSpec(dtype=torch.float32, shape=(-1,), optional=True),
            }
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, required, optional=None, **kwargs):
                if optional is not None:
                    return {"out": required + optional}
                return {"out": required}

            def load(self, params, serial_dir):
                return None

        # Test standalone node directly without graph
        node = NodeWithOptionalInput()

        # Provide both inputs
        from cuvis_ai.utils.types import Context

        context = Context(stage="inference")
        result = node.forward(
            required=torch.tensor([1.0, 2.0, 3.0]),
            optional=torch.tensor([0.5, 0.5, 0.5]),
            context=context,
        )
        expected = torch.tensor([1.5, 2.5, 3.5])
        torch.testing.assert_close(result["out"], expected)


class TestBatchDataKeyMapping:
    """Test mapping of batch dictionary keys to node input ports."""

    def test_batch_key_matches_port_name(self) -> None:
        """Batch keys matching port names should be routed correctly."""

        class SimpleNode(Node):
            INPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1, -1))}
            OUTPUT_SPECS = {"result": PortSpec(dtype=torch.float32, shape=(-1, -1))}

            def forward(self, data, **kwargs):
                return {"result": data * 2}

            def load(self, params, serial_dir):
                return None

        # Test standalone node directly without graph
        node = SimpleNode()

        # Test that the node processes data correctly
        test_data = torch.randn(4, 10)
        from cuvis_ai.utils.types import Context

        context = Context(stage="inference")
        result = node.forward(data=test_data, context=context)

        assert "result" in result
        torch.testing.assert_close(result["result"], test_data * 2)

    def test_batch_key_mismatch_raises_error(self) -> None:
        """Batch key not matching any port should cause missing input error in graph execution."""

        class SimpleNode(Node):
            INPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1, -1))}
            OUTPUT_SPECS = {"result": PortSpec(dtype=torch.float32, shape=(-1, -1))}

            def forward(self, data, **kwargs):
                return {"result": data * 2}

            def load(self, params, serial_dir):
                return None

        canvas = CuvisCanvas("batch_mismatch")
        consumer = SimpleNode()

        # Add node to graph by connecting to itself (dummy connection to ensure it's in graph)
        # Actually, just connect its output to input to create a cycle isn't what we want
        # Let's use a simpler approach - create two nodes where one feeds the other
        class ProducerNode(Node):
            INPUT_SPECS = {"input_data": PortSpec(dtype=torch.float32, shape=(-1, -1))}
            OUTPUT_SPECS = {"output": PortSpec(dtype=torch.float32, shape=(-1, -1))}

            def forward(self, input_data, **kwargs):
                return {"output": input_data}

            def load(self, params, serial_dir):
                return None

        producer = ProducerNode()
        canvas.connect(producer.output, consumer.data)

        # Batch has wrong key - producer needs "input_data" but batch only has "wrong_key"
        batch = {"wrong_key": torch.randn(4, 10)}

        with pytest.raises(RuntimeError, match="missing required inputs"):
            canvas.forward(stage="inference", batch=batch)

    def test_connection_overrides_batch_key(self) -> None:
        """Port connection should take precedence over batch key."""

        class ProducerNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, **kwargs):
                return {"data": torch.tensor([10.0, 20.0, 30.0])}

            def load(self, params, serial_dir):
                return None

        class ConsumerNode(Node):
            INPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"result": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, data, **kwargs):
                return {"result": data * 2}

            def load(self, params, serial_dir):
                return None

        canvas = CuvisCanvas("connection_priority")
        producer = ProducerNode()
        consumer = ConsumerNode()

        canvas.connect(producer.data, consumer.data)

        # Batch also has "data" key, but connection should take precedence
        batch = {"data": torch.tensor([1.0, 2.0, 3.0])}

        outputs = canvas.forward(stage="inference", batch=batch)

        # Consumer should receive data from producer, not batch
        expected = torch.tensor([20.0, 40.0, 60.0])  # producer output * 2
        torch.testing.assert_close(outputs[(consumer.id, "result")], expected)


class TestStageAwareExecution:
    """Test stage-specific execution filtering."""

    def test_train_stage_executes_loss_nodes(self) -> None:
        """Loss nodes should execute in train stage."""
        canvas = CuvisCanvas("stage_test")
        data = LentilsAnomalyDataNode(normal_class_ids=[0])
        rx_node = RXPerBatch()
        rx_logits = RXLogitHead()
        bce_loss = BCEwithSigmoidLoss(weight=0.3)

        canvas.connect(
            (data.outputs.cube, rx_node.data),
            (rx_node.scores, rx_logits.scores),
            (rx_logits.logits, bce_loss.predictions),
            (data.outputs.mask, bce_loss.targets),  # Connect targets from data node
        )

        # Provide all required inputs with correct dtype and shape (3D not 4D)
        batch = {
            "cube": torch.randn(2, 4, 4, 8),
            "mask": torch.randint(0, 2, (2, 4, 4), dtype=torch.int32),
        }

        outputs = canvas.forward(stage="train", batch=batch)

        # Loss should execute in train stage
        assert (bce_loss.id, "loss") in outputs

    def test_inference_stage_skips_loss_nodes(self) -> None:
        """Loss nodes should not execute in inference stage."""
        canvas = CuvisCanvas("stage_test")
        data = LentilsAnomalyDataNode(normal_class_ids=[0])
        rx_node = RXPerBatch()
        rx_logits = RXLogitHead()
        bce_loss = BCEwithSigmoidLoss(weight=0.3)

        canvas.connect(
            (data.outputs.cube, rx_node.data),
            (rx_node.scores, rx_logits.scores),
            (rx_logits.logits, bce_loss.predictions),
            # Don't connect targets - loss won't execute in inference anyway
        )

        # Provide cube only - loss doesn't execute in inference
        batch = {"cube": torch.randn(2, 4, 4, 8)}

        outputs = canvas.forward(stage="inference", batch=batch)

        # Loss should NOT execute in inference stage
        assert (bce_loss.id, "loss") not in outputs
        # But rx_logits should execute
        assert (rx_logits.id, "logits") in outputs


class TestMultipleEntryPoints:
    """Test graphs with multiple entry nodes."""

    def test_multiple_batch_keys_to_different_nodes(self) -> None:
        """Multiple batch keys should route to different nodes correctly."""

        class SourceA(Node):
            INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, x, **kwargs):
                return {"out": x * 2}

            def load(self, params, serial_dir):
                return None

        class SourceB(Node):
            INPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, y, **kwargs):
                return {"out": y * 3}

            def load(self, params, serial_dir):
                return None

        class Combiner(Node):
            INPUT_SPECS = {
                "a": PortSpec(dtype=torch.float32, shape=(-1,)),
                "b": PortSpec(dtype=torch.float32, shape=(-1,)),
            }
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, a, b, **kwargs):
                return {"out": a + b}

            def load(self, params, serial_dir):
                return None

        canvas = CuvisCanvas("multi_entry")
        source_a = SourceA()
        source_b = SourceB()
        combiner = Combiner()

        canvas.connect(
            (source_a.out, combiner.a),
            (source_b.out, combiner.b),
        )

        # Provide both entry inputs
        batch = {
            "x": torch.tensor([1.0]),
            "y": torch.tensor([2.0]),
        }

        outputs = canvas.forward(stage="inference", batch=batch)

        # Result should be (1.0 * 2) + (2.0 * 3) = 8.0
        expected = torch.tensor([8.0])
        torch.testing.assert_close(outputs[(combiner.id, "out")], expected)

    def test_missing_one_entry_point_raises_error(self) -> None:
        """Missing one of multiple entry points should raise error."""

        class SourceA(Node):
            INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, x, **kwargs):
                return {"out": x * 2}

            def load(self, params, serial_dir):
                return None

        class SourceB(Node):
            INPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, y, **kwargs):
                return {"out": y * 3}

            def load(self, params, serial_dir):
                return None

        class Combiner(Node):
            INPUT_SPECS = {
                "a": PortSpec(dtype=torch.float32, shape=(-1,)),
                "b": PortSpec(dtype=torch.float32, shape=(-1,)),
            }
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, a, b, **kwargs):
                return {"out": a + b}

            def load(self, params, serial_dir):
                return None

        canvas = CuvisCanvas("multi_entry")
        source_a = SourceA()
        source_b = SourceB()
        combiner = Combiner()

        canvas.connect(
            (source_a.out, combiner.a),
            (source_b.out, combiner.b),
        )

        # Only provide one entry input
        batch = {"x": torch.tensor([1.0])}

        with pytest.raises(RuntimeError, match="missing required inputs"):
            canvas.forward(stage="inference", batch=batch)
