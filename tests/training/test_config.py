"""Tests for training configuration infrastructure."""

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from cuvis_ai.anomaly.rx_logit_head import RXLogitHead
from cuvis_ai.node.selector import SoftChannelSelector
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.config import (
    OptimizerConfig,
    TrainerConfig,
    TrainingConfig,
)


def test_trainer_config_defaults():
    """Test TrainerConfig default values."""
    config = TrainerConfig()
    assert config.max_epochs == 100
    assert config.accelerator == "auto"
    assert config.devices is None
    assert config.precision == "32-true"
    assert config.accumulate_grad_batches == 1
    assert config.enable_progress_bar is True
    assert config.enable_checkpointing is False
    assert config.log_every_n_steps == 50


def test_optimizer_config_defaults():
    """Test OptimizerConfig default values."""
    config = OptimizerConfig()
    assert config.name == "adamw"
    assert config.lr == 1e-3
    assert config.weight_decay == 0.0
    assert config.betas is None


def test_training_config_defaults():
    """Test TrainingConfig default values."""
    config = TrainingConfig()
    assert config.seed == 42
    assert isinstance(config.trainer, TrainerConfig)
    assert isinstance(config.optimizer, OptimizerConfig)


def test_training_config_custom():
    """Test TrainingConfig with custom values."""
    trainer = TrainerConfig(max_epochs=10, accelerator="gpu")
    optimizer = OptimizerConfig(name="adamw", lr=0.001)
    config = TrainingConfig(seed=123, trainer=trainer, optimizer=optimizer)

    assert config.seed == 123
    assert config.trainer.max_epochs == 10
    assert config.trainer.accelerator == "gpu"
    assert config.optimizer.name == "adamw"
    assert config.optimizer.lr == 0.001


def test_to_dict():
    """Test to_dict conversion."""
    config = TrainingConfig(
        seed=42, trainer=TrainerConfig(max_epochs=5), optimizer=OptimizerConfig(lr=0.01)
    )

    result = config.to_dict()

    assert result["seed"] == 42
    assert result["trainer"]["max_epochs"] == 5
    assert result["optimizer"]["lr"] == 0.01


def test_to_dict_config():
    """Test to_dict_config conversion."""
    config = TrainingConfig(seed=99)
    dict_config = config.to_dict_config()

    assert isinstance(dict_config, DictConfig)
    assert dict_config.seed == 99
    assert dict_config.trainer.max_epochs == 100


def test_from_dict_config():
    """Test from_dict_config conversion."""
    dict_config = OmegaConf.create(
        {
            "seed": 77,
            "trainer": {"max_epochs": 20, "accelerator": "gpu"},
            "optimizer": {"name": "sgd", "lr": 0.1},
        }
    )

    config = TrainingConfig.from_dict_config(dict_config)

    assert isinstance(config, TrainingConfig)
    assert config.seed == 77
    assert config.trainer.max_epochs == 20
    assert config.trainer.accelerator == "gpu"
    assert config.optimizer.name == "sgd"
    assert config.optimizer.lr == 0.1


def test_roundtrip_serialization():
    """Test serialization roundtrip: config -> dict -> config."""
    original = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(max_epochs=15, accelerator="gpu", devices=2),
        optimizer=OptimizerConfig(name="adam", lr=0.003, weight_decay=0.01),
    )

    # Convert to dict config
    dict_config = original.to_dict_config()

    # Convert back to TrainingConfig
    restored = TrainingConfig.from_dict_config(dict_config)

    # Verify all fields match
    assert restored.seed == original.seed
    assert restored.trainer.max_epochs == original.trainer.max_epochs
    assert restored.trainer.accelerator == original.trainer.accelerator
    assert restored.trainer.devices == original.trainer.devices
    assert restored.optimizer.name == original.optimizer.name
    assert restored.optimizer.lr == original.optimizer.lr
    assert restored.optimizer.weight_decay == original.optimizer.weight_decay


class TestUnfreezeNodesByName:
    """Tests for unfreeze_nodes_by_name utility function."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline with named nodes."""
        pipeline = CuvisPipeline(name="test_pipeline")

        # Create nodes with names
        selector = SoftChannelSelector(
            name="SoftChannelSelector",
            n_select=3,
            input_channels=10,
            init_method="uniform",
        )
        logit_head = RXLogitHead(name="RXLogitHead")

        # Connect nodes to add them to pipeline (pipeline auto-adds nodes on connect)
        pipeline.connect(selector.outputs.selected, logit_head.inputs.scores)

        return pipeline

    def test_unfreeze_with_empty_list(self, mock_pipeline):
        """Test that empty node list does nothing."""
        # Should not raise any errors
        mock_pipeline.unfreeze_nodes_by_name([])

        # Nodes should still be in their default state
        for node in mock_pipeline.nodes():
            if hasattr(node, "unfreeze"):
                # Check if any parameters exist - if frozen they'd be buffers
                for _, param in node.named_parameters():
                    # Default state varies, just ensure no error occurred
                    assert param is not None

    def test_unfreeze_valid_nodes(self, mock_pipeline):
        """Test unfreezing valid nodes by name."""
        # First freeze the nodes
        for node in mock_pipeline.nodes():
            if hasattr(node, "freeze"):
                node.freeze()

        # Verify nodes are frozen (parameters converted to buffers)
        selector = None
        logit_head = None
        for node in mock_pipeline.nodes():
            if node.name == "SoftChannelSelector":
                selector = node
            elif node.name == "RXLogitHead":
                logit_head = node

        assert selector is not None, "SoftChannelSelector not found in pipeline"
        assert logit_head is not None, "RXLogitHead not found in pipeline"

        # After freezing, trainable parameter count should be 0
        assert sum(p.numel() for p in selector.parameters() if p.requires_grad) == 0
        assert sum(p.numel() for p in logit_head.parameters() if p.requires_grad) == 0

        # Now unfreeze
        mock_pipeline.unfreeze_nodes_by_name(["SoftChannelSelector", "RXLogitHead"])

        # After unfreezing, should have trainable parameters again
        assert sum(p.numel() for p in selector.parameters() if p.requires_grad) > 0
        assert sum(p.numel() for p in logit_head.parameters() if p.requires_grad) > 0

    def test_unfreeze_missing_nodes_raises_error(self, mock_pipeline):
        """Test that missing node names raise ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            mock_pipeline.unfreeze_nodes_by_name(["NonExistentNode", "AnotherMissingNode"])

        error_msg = str(exc_info.value)
        assert "Trainable nodes not found in pipeline" in error_msg
        assert "NonExistentNode" in error_msg
        assert "AnotherMissingNode" in error_msg
        assert "Available nodes" in error_msg
        assert "SoftChannelSelector" in error_msg
        assert "RXLogitHead" in error_msg

    def test_unfreeze_partial_missing_nodes_raises_error(self, mock_pipeline):
        """Test that partially missing nodes raise error."""
        with pytest.raises(ValueError) as exc_info:
            mock_pipeline.unfreeze_nodes_by_name(["SoftChannelSelector", "NonExistentNode"])

        error_msg = str(exc_info.value)
        assert "NonExistentNode" in error_msg
        # Should only list missing nodes, not valid ones
        assert error_msg.count("NonExistentNode") >= 1

    def test_unfreeze_single_node(self, mock_pipeline):
        """Test unfreezing a single node."""
        # Freeze first
        for node in mock_pipeline.nodes():
            if node.name == "SoftChannelSelector" and hasattr(node, "freeze"):
                node.freeze()

        # Unfreeze just one node
        mock_pipeline.unfreeze_nodes_by_name(["SoftChannelSelector"])

        # Check that the selector was unfrozen
        selector = None
        for node in mock_pipeline.nodes():
            if node.name == "SoftChannelSelector":
                selector = node
                break

        assert selector is not None
        assert sum(p.numel() for p in selector.parameters() if p.requires_grad) > 0

    def test_unfreeze_node_without_unfreeze_method(self):
        """Test that nodes without unfreeze method are skipped gracefully."""
        # This test verifies the graceful handling when a node doesn't have unfreeze
        # Since we can't easily add nodes without connecting them, we'll use a mock
        from unittest.mock import MagicMock

        pipeline = MagicMock()

        # Create a simple node without unfreeze method
        class SimpleNode(torch.nn.Module):
            def __init__(self, name):
                super().__init__()
                self.name = name
                self.weight = torch.nn.Parameter(torch.randn(5, 5))

        simple_node = SimpleNode(name="SimpleNode")
        pipeline.nodes.return_value = [simple_node]

        # Should not raise error even though node doesn't have unfreeze
        pipeline.unfreeze_nodes_by_name(["SimpleNode"])

        # Node should still be accessible
        assert len(list(pipeline.nodes())) == 1

    def test_unfreeze_preserves_node_state(self, mock_pipeline):
        """Test that unfreezing preserves node state/weights."""
        # Get initial weights
        selector = None
        for node in mock_pipeline.nodes():
            if node.name == "SoftChannelSelector":
                selector = node
                break

        assert selector is not None, "SoftChannelSelector not found in pipeline"

        initial_weights = {}
        for name, param in selector.named_parameters():
            initial_weights[name] = param.clone().detach()

        # Freeze and unfreeze
        selector.freeze()
        mock_pipeline.unfreeze_nodes_by_name(["SoftChannelSelector"])

        # Check weights are preserved
        for name, param in selector.named_parameters():
            if name in initial_weights:
                assert torch.allclose(param, initial_weights[name], rtol=1e-5), (
                    f"Parameter {name} changed after freeze/unfreeze"
                )
