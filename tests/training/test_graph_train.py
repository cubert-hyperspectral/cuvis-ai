"""Tests for graph.train() orchestration."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.normalization.normalization import MinMaxNormalizer
from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.training.config import TrainingConfig, TrainerConfig
from cuvis_ai.training.datamodule import GraphDataModule


class MockDataModule(GraphDataModule):
    """Mock datamodule for testing."""
    
    def __init__(self, batch_size=2, num_samples=10):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, 10, 10, 5)  # N, H, W, C
    
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        pass
    
    def train_dataloader(self):
        # Return DataLoader that yields dicts directly
        class DictDataset:
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return {"cube": self.data[idx]}
        
        dataset = DictDataset(self.data)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
    
    def val_dataloader(self):
        # Return DataLoader that yields dicts directly
        class DictDataset:
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return {"cube": self.data[idx]}
        
        dataset = DictDataset(self.data[:4])
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
        )


def test_graph_train_statistical_only():
    """Test graph.train() with statistical initialization only (no gradient training)."""
    # Create graph
    graph = Graph("test_statistical")
    rx = RXGlobal(trainable_stats=False)
    normalizer = MinMaxNormalizer(use_running_stats=True)
    
    graph.add_node(rx)
    graph.add_node(normalizer, parent=rx)
    
    # Create datamodule
    datamodule = MockDataModule(batch_size=2, num_samples=10)
    
    # Create config for statistical-only training
    config = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(max_epochs=0)  # No gradient training
    )
    
    # Train (should only do statistical initialization)
    trainer = graph.train(datamodule=datamodule, training_config=config)
    
    # Verify statistical parameters were initialized
    assert rx.mu is not None
    assert rx.cov is not None
    assert rx.mu.shape == torch.Size([5])
    assert rx.cov.shape == torch.Size([5, 5])
    
    assert normalizer.running_min is not None
    assert normalizer.running_max is not None
    
    # Verify nodes are frozen
    assert rx.freezed is True
    assert normalizer.freezed is True
    
    # Verify no trainer was created (max_epochs=0)
    assert trainer is None


def test_graph_train_with_gradient_training():
    """Test graph.train() with both statistical init and gradient training."""
    from cuvis_ai.node.pca import TrainablePCA
    from cuvis_ai.training.losses import OrthogonalityLoss
    
    # Create graph with trainable PCA
    graph = Graph("test_with_training")
    pca = TrainablePCA(n_components=3, trainable=True)
    
    graph.add_node(pca)
    
    # Add orthogonality loss leaf
    orth_loss = OrthogonalityLoss(weight=1.0)
    graph.add_leaf_node(orth_loss, parent=pca)
    
    # Create datamodule
    datamodule = MockDataModule(batch_size=2, num_samples=10)
    
    # Create config with gradient training
    config = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(
            max_epochs=2,  # Two epochs of training
            enable_progress_bar=False,
            enable_checkpointing=False
        )
    )
    
    # Train
    trainer = graph.train(datamodule=datamodule, training_config=config)
    
    # Verify PCA was initialized
    assert pca.mean is not None
    assert pca.components is not None
    assert pca.explained_variance is not None
    
    # Verify trainer was created
    assert trainer is not None
    
    # Verify PCA components are trainable parameters
    assert isinstance(pca.components, torch.nn.Parameter)
    assert pca.components.requires_grad is True
    
    # Verify PCA is not frozen (it's trainable)
    assert pca.freezed is False


def test_graph_train_seed_reproducibility():
    """Test that same seed produces reproducible results."""
    # Create fixed data (to isolate seed effects on initialization)
    torch.manual_seed(12345)  # Fixed seed for data generation
    fixed_data = torch.randn(10, 10, 10, 5)
    
    def create_and_train(seed):
        # Reset seed before each training run
        torch.manual_seed(seed)
        
        graph = Graph(f"test_seed_{seed}")
        rx = RXGlobal()
        graph.add_node(rx)
        
        # Use same fixed data for all runs
        datamodule = MockDataModule(batch_size=2, num_samples=10)
        datamodule.data = fixed_data.clone()
        
        config = TrainingConfig(
            seed=seed,
            trainer=TrainerConfig(max_epochs=0)
        )
        
        graph.train(datamodule=datamodule, training_config=config)
        return rx.mu.clone(), rx.cov.clone()
    
    # Train twice with same seed
    mu1, cov1 = create_and_train(42)
    mu2, cov2 = create_and_train(42)
    
    # Should be identical
    assert torch.allclose(mu1, mu2, atol=1e-6)
    assert torch.allclose(cov1, cov2, atol=1e-6)


def test_graph_forward_after_training():
    """Test that graph can perform forward pass after training."""
    # Create and train graph
    graph = Graph("test_forward")
    rx = RXGlobal()
    normalizer = MinMaxNormalizer(use_running_stats=True)
    
    graph.add_node(rx)
    graph.add_node(normalizer, parent=rx)
    
    datamodule = MockDataModule(batch_size=2, num_samples=10)
    config = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(max_epochs=0)
    )
    
    graph.train(datamodule=datamodule, training_config=config)
    
    # Perform forward pass
    test_input = torch.randn(1, 10, 10, 5)
    output, _, _ = graph.forward(test_input)
    
    # Verify output shape
    assert output.shape == torch.Size([1, 10, 10, 1])
    
    # Verify output is normalized (between 0 and 1)
    assert output.min() >= 0.0
    assert output.max() <= 1.0
