"""Tests for external trainer orchestration."""

import pytest
import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.trainers import StatisticalTrainer
from cuvis_ai_schemas.enums import ExecutionStage

from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.normalization import MinMaxNormalizer

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_graph_train_statistical_only(synthetic_anomaly_datamodule):
    """Test StatisticalTrainer with statistical initialization only (no gradient training)."""
    # Create graph with data node to adapt inputs
    pipeline = CuvisPipeline("test_statistical")
    data_node = LentilsAnomalyDataNode(normal_class_ids=[0])
    rx = RXGlobal(num_channels=5)
    normalizer = MinMaxNormalizer(use_running_stats=True)

    # Connect nodes (automatically adds them to graph)
    pipeline.connect((data_node.outputs.cube, rx.data), (rx.scores, normalizer.data))

    # Create datamodule
    datamodule = synthetic_anomaly_datamodule(
        batch_size=2,
        num_samples=10,
        height=10,
        width=10,
        channels=5,
        include_labels=False,
        dtype=torch.uint16,
    )

    # Use StatisticalTrainer for initialization
    stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    stat_trainer.fit()

    # Verify statistical parameters were initialized
    assert rx.mu is not None
    assert rx.cov is not None
    assert rx.mu.shape == torch.Size([5])
    assert rx.cov.shape == torch.Size([5, 5])

    assert normalizer.running_min is not None
    assert normalizer.running_max is not None

    # Verify nodes are initialized
    assert rx._statistically_initialized is True
    assert normalizer._statistically_initialized is True


def test_graph_train_with_gradient_training(synthetic_anomaly_datamodule):
    """Test StatisticalTrainer with PCA initialization."""
    from cuvis_ai.node.dimensionality_reduction import TrainablePCA

    # Create graph with trainable PCA and data node
    pipeline = CuvisPipeline("test_with_training")
    data_node = LentilsAnomalyDataNode(normal_class_ids=[0])
    pca = TrainablePCA(num_channels=5, n_components=3, trainable=True)

    # Connect PCA
    pipeline.connect(data_node.outputs.cube, pca.data)

    # Create datamodule
    datamodule = synthetic_anomaly_datamodule(
        batch_size=2,
        num_samples=10,
        height=10,
        width=10,
        channels=5,
        include_labels=False,
        dtype=torch.uint16,
    )

    # Statistical initialization
    stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    stat_trainer.fit()

    # Verify PCA was initialized (using underscore-prefixed attributes)
    assert pca._mean is not None
    assert pca._components is not None
    assert pca._explained_variance is not None

    # Verify components shape
    assert pca._components.shape == torch.Size([3, 5])  # n_components x n_features


def test_graph_train_seed_reproducibility(synthetic_anomaly_datamodule):
    """Test that same seed produces reproducible results."""

    def create_and_train(run_seed):
        torch.manual_seed(run_seed)

        pipeline = CuvisPipeline(f"test_seed_{run_seed}")
        data_node = LentilsAnomalyDataNode(normal_class_ids=[0])
        rx = RXGlobal(num_channels=5)
        normalizer = MinMaxNormalizer(use_running_stats=True)

        # Connect with data node
        pipeline.connect((data_node.outputs.cube, rx.data), (rx.scores, normalizer.data))

        # Use same seed for deterministic data generation
        datamodule = synthetic_anomaly_datamodule(
            batch_size=2,
            num_samples=10,
            height=10,
            width=10,
            channels=5,
            include_labels=False,
            dtype=torch.uint16,
            seed=12345,
        )

        # Use StatisticalTrainer
        stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
        stat_trainer.fit()

        return rx.mu, rx.cov

    # Train twice with same seed
    mu1, cov1 = create_and_train(42)
    mu2, cov2 = create_and_train(42)

    # Should be identical
    assert torch.allclose(mu1, mu2, atol=1e-6)
    assert torch.allclose(cov1, cov2, atol=1e-6)


def test_graph_forward_after_training(synthetic_anomaly_datamodule):
    """Test that graph can perform forward pass after training."""
    # Create and train graph with data node
    pipeline = CuvisPipeline("test_forward")
    data_node = LentilsAnomalyDataNode(normal_class_ids=[0])
    rx = RXGlobal(num_channels=5)
    normalizer = MinMaxNormalizer(use_running_stats=True)

    # Connect nodes (automatically adds them to graph)
    pipeline.connect((data_node.outputs.cube, rx.data), (rx.scores, normalizer.data))

    datamodule = synthetic_anomaly_datamodule(
        batch_size=2,
        num_samples=10,
        height=10,
        width=10,
        channels=5,
        include_labels=False,
        dtype=torch.uint16,
    )

    # Use StatisticalTrainer
    stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    stat_trainer.fit()

    # Perform forward pass - use uint16 matching real sensor data
    test_input = torch.randint(0, 65535, (1, 10, 10, 5), dtype=torch.uint16)
    wavelengths = torch.arange(5, dtype=torch.int32).unsqueeze(0)  # 2D [1, 5]
    batch = {
        "cube": test_input,
        "wavelengths": wavelengths,
    }
    outputs = pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)

    # Extract normalized output using (node_name, port_name) tuple
    normalized = outputs[(normalizer.name, "normalized")]

    # Verify output shape
    assert normalized.shape == torch.Size([1, 10, 10, 1])

    # Verify output is normalized (between 0 and 1, with small tolerance for numerical precision)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.01  # Small tolerance for numerical precision
