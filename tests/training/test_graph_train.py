"""Tests for external trainer orchestration."""

import torch
from torch.utils.data import DataLoader

from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.datamodule import CuvisDataModule
from cuvis_ai_core.training.trainers import StatisticalTrainer
from cuvis_ai_core.utils.types import ExecutionStage


class MockDataModule(CuvisDataModule):
    """Mock datamodule for testing."""

    def __init__(self, batch_size=2, num_samples=10, dtype=torch.uint16):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples
        # Create data with specified dtype (default uint16)
        self.data = torch.randint(0, 65535, (num_samples, 10, 10, 5), dtype=dtype)  # N, H, W, C
        # Create wavelengths for 5 channels
        self.wavelengths = torch.arange(5, dtype=torch.int32)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        # Return DataLoader that yields dicts directly
        class DictDataset:
            def __init__(self, data, wavelengths):
                self.data = data
                self.wavelengths = wavelengths

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return {
                    "cube": self.data[idx],
                    "wavelengths": self.wavelengths,  # 1D, DataLoader will stack to 2D
                }

        dataset = DictDataset(self.data, self.wavelengths)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        # Return DataLoader that yields dicts directly
        class DictDataset:
            def __init__(self, data, wavelengths):
                self.data = data
                self.wavelengths = wavelengths

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return {
                    "cube": self.data[idx],
                    "wavelengths": self.wavelengths,  # 1D, DataLoader will stack to 2D
                }

        dataset = DictDataset(self.data[:4], self.wavelengths)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


def test_graph_train_statistical_only():
    """Test StatisticalTrainer with statistical initialization only (no gradient training)."""
    # Create graph with data node to adapt inputs
    pipeline = CuvisPipeline("test_statistical")
    data_node = LentilsAnomalyDataNode(normal_class_ids=[0])
    rx = RXGlobal(num_channels=5)
    normalizer = MinMaxNormalizer(use_running_stats=True)

    # Connect nodes (automatically adds them to graph)
    pipeline.connect((data_node.outputs.cube, rx.data), (rx.scores, normalizer.data))

    # Create datamodule
    datamodule = MockDataModule(batch_size=2, num_samples=10)

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


def test_graph_train_with_gradient_training():
    """Test StatisticalTrainer with PCA initialization."""
    from cuvis_ai.node.pca import TrainablePCA

    # Create graph with trainable PCA and data node
    pipeline = CuvisPipeline("test_with_training")
    data_node = LentilsAnomalyDataNode(normal_class_ids=[0])
    pca = TrainablePCA(n_components=3, trainable=True)

    # Connect PCA
    pipeline.connect(data_node.outputs.cube, pca.data)

    # Create datamodule
    datamodule = MockDataModule(batch_size=2, num_samples=10)

    # Statistical initialization
    stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    stat_trainer.fit()

    # Verify PCA was initialized (using underscore-prefixed attributes)
    assert pca._mean is not None
    assert pca._components is not None
    assert pca._explained_variance is not None

    # Verify components shape
    assert pca._components.shape == torch.Size([3, 5])  # n_components x n_features


def test_graph_train_seed_reproducibility():
    """Test that same seed produces reproducible results."""
    # Create fixed data (to isolate seed effects on initialization)
    torch.manual_seed(12345)  # Fixed seed for data generation
    fixed_data = torch.randint(0, 65535, (10, 10, 10, 5), dtype=torch.uint16)

    def create_and_train(seed):
        # Reset seed before each training run
        torch.manual_seed(seed)

        pipeline = CuvisPipeline(f"test_seed_{seed}")
        data_node = LentilsAnomalyDataNode(normal_class_ids=[0])
        rx = RXGlobal(num_channels=5)
        normalizer = MinMaxNormalizer(use_running_stats=True)

        # Connect with data node
        pipeline.connect((data_node.outputs.cube, rx.data), (rx.scores, normalizer.data))

        # Use same fixed data for all runs
        datamodule = MockDataModule(batch_size=2, num_samples=10)
        datamodule.data = fixed_data.clone()

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


def test_graph_forward_after_training():
    """Test that graph can perform forward pass after training."""
    # Create and train graph with data node
    pipeline = CuvisPipeline("test_forward")
    data_node = LentilsAnomalyDataNode(normal_class_ids=[0])
    rx = RXGlobal(num_channels=5)
    normalizer = MinMaxNormalizer(use_running_stats=True)

    # Connect nodes (automatically adds them to graph)
    pipeline.connect((data_node.outputs.cube, rx.data), (rx.scores, normalizer.data))

    datamodule = MockDataModule(batch_size=2, num_samples=10)

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
