"""
Test suite for runtime I/O validation in CuvisCanvas.

Tests that nodes' actual inputs and outputs are validated against their
INPUT_SPECS and OUTPUT_SPECS at runtime.
"""

import pytest
import torch

from cuvis_ai.node import Node
from cuvis_ai.pipeline.canvas import CuvisCanvas
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.utils.types import ExecutionStage


class ValidNode(Node):
    """Node that returns correct output types."""

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 10),
            description="Input data",
        )
    }

    OUTPUT_SPECS = {
        "result": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 5),
            description="Output result",
        )
    }

    def forward(self, data, **kwargs):
        # Correctly returns float32 with shape matching OUTPUT_SPECS
        B, H, W, C = data.shape
        result = torch.randn(B, H, W, 5, dtype=torch.float32)
        return {"result": result}

    def serialize(self, directory: str):
        return {}

    def load(self, params: dict, filepath: str):
        pass


class WrongDtypeOutputNode(Node):
    """Node that returns wrong output dtype."""

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 10),
            description="Input data",
        )
    }

    OUTPUT_SPECS = {
        "result": PortSpec(
            dtype=torch.bool,  # Declares bool
            shape=(-1, -1, -1, 1),
            description="Output result",
        )
    }

    def forward(self, data, **kwargs):
        # Returns float32 instead of bool (mimics BinaryDecider bug)
        B, H, W, C = data.shape
        result = torch.randn(B, H, W, 1, dtype=torch.float32)
        return {"result": result}

    def serialize(self, directory: str):
        return {}

    def load(self, params: dict, filepath: str):
        pass


class WrongShapeOutputNode(Node):
    """Node that returns wrong output shape."""

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 10),
            description="Input data",
        )
    }

    OUTPUT_SPECS = {
        "result": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 5),  # Declares shape with 5 channels
            description="Output result",
        )
    }

    def forward(self, data, **kwargs):
        # Returns 3 channels instead of 5
        B, H, W, C = data.shape
        result = torch.randn(B, H, W, 3, dtype=torch.float32)
        return {"result": result}

    def serialize(self, directory: str):
        return {}

    def load(self, params: dict, filepath: str):
        pass


class MissingOutputNode(Node):
    """Node that doesn't return a required output."""

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 10),
            description="Input data",
        )
    }

    OUTPUT_SPECS = {
        "result": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 5),
            description="Output result",
        )
    }

    def forward(self, data, **kwargs):
        # Returns empty dict instead of required output
        return {}

    def serialize(self, directory: str):
        return {}

    def load(self, params: dict, filepath: str):
        pass


class DataSourceNode(Node):
    """Node that provides data for testing."""

    INPUT_SPECS = {}

    OUTPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 10),
            description="Test data",
        )
    }

    def forward(self, **kwargs):
        # Generate test data
        data = torch.randn(2, 4, 4, 10, dtype=torch.float32)
        return {"data": data}

    def serialize(self, directory: str):
        return {}

    def load(self, params: dict, filepath: str):
        pass


class TestRuntimeOutputValidation:
    """Test runtime output validation against OUTPUT_SPECS."""

    def test_valid_output_passes(self):
        """Test that correct output types pass validation."""
        canvas = CuvisCanvas("test", strict_runtime_io_validation=True)

        source = DataSourceNode()
        node = ValidNode()

        canvas.connect(source.outputs.data, node.data)

        # Should not raise
        outputs = canvas.forward(stage=ExecutionStage.INFERENCE)
        assert (node.id, "result") in outputs

    def test_wrong_dtype_output_fails(self):
        """Test that wrong output dtype is caught."""
        from cuvis_ai.pipeline.ports import PortCompatibilityError

        canvas = CuvisCanvas("test", strict_runtime_io_validation=True)

        source = DataSourceNode()
        node = WrongDtypeOutputNode()

        canvas.connect(source.outputs.data, node.data)

        # Should raise PortCompatibilityError due to dtype mismatch
        with pytest.raises(PortCompatibilityError, match="[Dd]type"):
            canvas.forward(stage=ExecutionStage.INFERENCE)

    def test_wrong_shape_output_fails(self):
        """Test that wrong output shape is caught."""
        from cuvis_ai.pipeline.ports import PortCompatibilityError

        canvas = CuvisCanvas("test", strict_runtime_io_validation=True)

        source = DataSourceNode()
        node = WrongShapeOutputNode()

        canvas.connect(source.outputs.data, node.data)

        # Should raise PortCompatibilityError due to shape mismatch
        with pytest.raises(PortCompatibilityError, match="[Dd]imension"):
            canvas.forward(stage=ExecutionStage.INFERENCE)

    def test_missing_output_fails(self):
        """Test that missing required output is caught."""
        canvas = CuvisCanvas("test", strict_runtime_io_validation=True)

        source = DataSourceNode()
        node = MissingOutputNode()

        canvas.connect(source.outputs.data, node.data)

        # Should raise RuntimeError for missing output
        with pytest.raises(RuntimeError, match="did not produce required output"):
            canvas.forward(stage=ExecutionStage.INFERENCE)

    def test_validation_can_be_disabled(self):
        """Test that validation can be disabled."""
        canvas = CuvisCanvas("test", strict_runtime_io_validation=False)

        source = DataSourceNode()
        node = WrongDtypeOutputNode()

        canvas.connect(source.outputs.data, node.data)

        # Should not raise even with wrong dtype
        outputs = canvas.forward(stage=ExecutionStage.INFERENCE)
        assert (node.id, "result") in outputs


class WrongDtypeSourceNode(Node):
    """Source node that returns wrong dtype."""

    INPUT_SPECS = {}

    OUTPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 10),
            description="Test data",
        )
    }

    def forward(self, **kwargs):
        # Returns int32 instead of float32
        data = torch.randint(0, 10, (2, 4, 4, 10), dtype=torch.int32)
        return {"data": data}

    def serialize(self, directory: str):
        return {}

    def load(self, params: dict, filepath: str):
        pass


class WrongShapeSourceNode(Node):
    """Source node that returns wrong shape."""

    INPUT_SPECS = {}

    OUTPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 10),
            description="Test data",
        )
    }

    def forward(self, **kwargs):
        # Returns 15 channels instead of 10
        data = torch.randn(2, 4, 4, 15, dtype=torch.float32)
        return {"data": data}

    def serialize(self, directory: str):
        return {}

    def load(self, params: dict, filepath: str):
        pass


class TestRuntimeInputValidation:
    """Test runtime input validation against INPUT_SPECS."""

    def test_wrong_dtype_input_fails(self):
        """Test that wrong input dtype from source node is caught."""
        from cuvis_ai.pipeline.ports import PortCompatibilityError

        canvas = CuvisCanvas("test", strict_runtime_io_validation=True)

        source = WrongDtypeSourceNode()
        node = ValidNode()
        canvas.connect(source.outputs.data, node.data)

        # Should raise PortCompatibilityError when ValidNode receives int32 instead of float32
        with pytest.raises(PortCompatibilityError, match="[Dd]type"):
            canvas.forward(stage=ExecutionStage.INFERENCE)

    def test_wrong_shape_input_fails(self):
        """Test that wrong input shape from source node is caught."""
        from cuvis_ai.pipeline.ports import PortCompatibilityError

        canvas = CuvisCanvas("test", strict_runtime_io_validation=True)

        source = WrongShapeSourceNode()
        node = ValidNode()
        canvas.connect(source.outputs.data, node.data)

        # Should raise PortCompatibilityError when ValidNode receives 15 channels instead of 10
        with pytest.raises(PortCompatibilityError, match="[Dd]imension"):
            canvas.forward(stage=ExecutionStage.INFERENCE)

    def test_flexible_dimensions_work(self):
        """Test that flexible dimensions (-1) accept any size."""
        canvas = CuvisCanvas("test", strict_runtime_io_validation=True)

        source = DataSourceNode()
        node = ValidNode()

        canvas.connect(source.outputs.data, node.data)

        # Different batch sizes should work (first 3 dims are flexible)
        outputs = canvas.forward(stage=ExecutionStage.INFERENCE)
        assert (node.id, "result") in outputs


class TestBinaryDeciderBug:
    """Integration test that catches the BinaryDecider dtype bug."""

    def test_binary_decider_returns_bool(self):
        """Test that BinaryDecider now returns bool as specified."""
        from cuvis_ai.deciders.binary_decider import BinaryDecider

        CuvisCanvas("test", strict_runtime_io_validation=True)

        DataSourceNode()
        decider = BinaryDecider(threshold=0.5)

        # Note: BinaryDecider expects 4D input but source provides 10 channels
        # We need to adjust or it will fail shape validation
        # For this test, let's just test the decider directly

        logits = torch.randn(2, 4, 4, 1, dtype=torch.float32)
        outputs = decider.forward(logits=logits)

        # Should return bool dtype
        assert outputs["decisions"].dtype == torch.bool

    def test_binary_decider_in_canvas_with_validation(self):
        """Test that BinaryDecider works in canvas with validation enabled."""
        from cuvis_ai.deciders.binary_decider import BinaryDecider

        # Create a simple source that outputs 1-channel data
        # (matching BinaryDecider's expected input)
        class SingleChannelSource(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {
                "scores": PortSpec(
                    dtype=torch.float32, shape=(-1, -1, -1, 1), description="Normalized scores"
                )
            }

            def forward(self, **kwargs):
                return {"scores": torch.randn(2, 4, 4, 1, dtype=torch.float32)}

            def serialize(self, directory: str):
                return {}

            def load(self, params: dict, filepath: str):
                pass

        canvas = CuvisCanvas("test", strict_runtime_io_validation=True)

        source = SingleChannelSource()
        decider = BinaryDecider(threshold=0.5)

        canvas.connect(source.scores, decider.logits)

        # Should not raise - BinaryDecider now returns bool correctly
        outputs = canvas.forward(stage=ExecutionStage.INFERENCE)

        # Verify output is bool with correct shape
        decisions = outputs[(decider.id, "decisions")]
        assert decisions.dtype == torch.bool
        assert decisions.shape == (2, 4, 4, 1)
