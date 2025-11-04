"""Tests for loss leaf nodes."""

import pytest
import torch
import torch.nn as nn

from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.training.leaf_nodes import LossNode
from cuvis_ai.training.losses import (
    AnomalyBCEWithLogits,
    MSEReconstructionLoss,
    OrthogonalityLoss,
    WeightedMultiLoss,
)


@pytest.fixture
def trainable_pca():
    """Create a TrainablePCA node for testing."""
    pca = TrainablePCA(n_components=3, trainable=True)
    
    # Initialize with dummy data
    data_iterator = [
        (torch.randn(2, 10, 10, 5), None, None)
        for _ in range(3)
    ]
    pca.initialize_from_data(data_iterator)
    pca.prepare_for_train()
    
    return pca


@pytest.fixture
def dummy_parent():
    """Create a dummy parent node."""
    class DummyNode(nn.Module):
        def forward(self, x, y=None, m=None):
            return x, y, m
    
    return DummyNode()


class TestOrthogonalityLoss:
    """Tests for OrthogonalityLoss."""
    
    def test_initialization(self):
        """Test OrthogonalityLoss initialization."""
        loss_node = OrthogonalityLoss(weight=2.0)
        assert loss_node.weight == 2.0
        assert isinstance(loss_node, LossNode)
    
    def test_compatible_parent_types(self):
        """Test that OrthogonalityLoss requires TrainablePCA parent."""
        assert TrainablePCA in OrthogonalityLoss.compatible_parent_types
    
    def test_compute_loss(self, trainable_pca):
        """Test orthogonality loss computation."""
        loss_node = OrthogonalityLoss(weight=1.0)
        loss_node.parent = trainable_pca
        
        # Create dummy parent output
        dummy_output = (torch.randn(2, 10, 10, 3), None, None)
        
        # Compute loss
        loss, info = loss_node.compute_loss(dummy_output)
        
        # Check loss is scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.requires_grad
        
        # Check info dict
        assert isinstance(info, dict)
        assert "orthogonality_loss" in info
        assert "weight" in info
        assert info["weight"] == 1.0
    
    def test_loss_weighting(self, trainable_pca):
        """Test that weight parameter scales loss correctly."""
        loss_node_1x = OrthogonalityLoss(weight=1.0)
        loss_node_1x.parent = trainable_pca
        
        loss_node_2x = OrthogonalityLoss(weight=2.0)
        loss_node_2x.parent = trainable_pca
        
        dummy_output = (torch.randn(2, 10, 10, 3), None, None)
        
        loss_1x, _ = loss_node_1x.compute_loss(dummy_output)
        loss_2x, _ = loss_node_2x.compute_loss(dummy_output)
        
        # 2x weight should give approximately 2x loss
        assert torch.allclose(loss_2x, loss_1x * 2.0, rtol=1e-5)
    
    def test_loss_decreases_with_training(self, trainable_pca):
        """Test that loss can decrease with gradient updates."""
        loss_node = OrthogonalityLoss(weight=1.0)
        loss_node.parent = trainable_pca
        
        # Degrade orthogonality so there's something to optimize
        trainable_pca.components.data += 0.1 * torch.randn_like(trainable_pca.components)
        
        optimizer = torch.optim.SGD(trainable_pca.parameters(), lr=0.01)
        
        dummy_output = (torch.randn(2, 10, 10, 3), None, None)
        
        # Initial loss (should be non-trivial now)
        loss_initial, _ = loss_node.compute_loss(dummy_output)
        initial_value = loss_initial.item()
        
        # Loss should be significantly above zero
        assert initial_value > 1e-6
        
        # Train for a few steps
        for _ in range(10):
            optimizer.zero_grad()
            loss, _ = loss_node.compute_loss(dummy_output)
            loss.backward()
            optimizer.step()
        
        # Final loss
        loss_final, _ = loss_node.compute_loss(dummy_output)
        final_value = loss_final.item()
        
        # Loss should decrease (components becoming more orthogonal)
        assert final_value < initial_value


class TestAnomalyBCEWithLogits:
    """Tests for AnomalyBCEWithLogits loss."""
    
    def test_initialization(self):
        """Test AnomalyBCEWithLogits initialization."""
        loss_node = AnomalyBCEWithLogits(pos_weight=2.0, reduction="mean")
        assert loss_node.pos_weight == 2.0
        assert loss_node.reduction == "mean"
        assert isinstance(loss_node, LossNode)
    
    def test_compute_loss_with_logits(self, dummy_parent):
        """Test BCE loss computation with logits."""
        loss_node = AnomalyBCEWithLogits(pos_weight=1.0)
        loss_node.parent = dummy_parent
        
        # Create dummy scores (logits) and labels
        B, H, W = 2, 10, 10
        scores = torch.randn(B, H, W, requires_grad=True)  # Logits with gradient tracking
        labels = torch.randint(0, 2, (B, H, W)).float()
        
        parent_output = (scores, labels, None)
        
        # Compute loss
        loss, info = loss_node.compute_loss(parent_output, labels=labels)
        
        # Check loss
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.requires_grad  # Now should be True since inputs require grad
        
        # Check info dict
        assert "bce_loss" in info
        assert "positive_rate" in info
        assert "pred_positive_rate" in info
        assert "accuracy" in info
    
    def test_compute_loss_with_4d_tensors(self, dummy_parent):
        """Test BCE loss with 4D tensors [B, H, W, 1]."""
        loss_node = AnomalyBCEWithLogits()
        loss_node.parent = dummy_parent
        
        # Create 4D tensors
        B, H, W = 2, 10, 10
        scores = torch.randn(B, H, W, 1)
        labels = torch.randint(0, 2, (B, H, W, 1)).float()
        
        parent_output = (scores, labels, None)
        
        # Should handle 4D tensors correctly
        loss, info = loss_node.compute_loss(parent_output, labels=labels)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
    
    def test_pos_weight_effect(self, dummy_parent):
        """Test that pos_weight affects loss calculation."""
        # Create imbalanced data (more negatives)
        B, H, W = 2, 10, 10
        scores = torch.randn(B, H, W)
        labels = torch.zeros(B, H, W)
        labels[0, 0, 0] = 1  # Only one positive example
        
        parent_output = (scores, labels, None)
        
        # Loss with pos_weight=1
        loss_node_1 = AnomalyBCEWithLogits(pos_weight=1.0)
        loss_node_1.parent = dummy_parent
        loss_1, _ = loss_node_1.compute_loss(parent_output, labels=labels)
        
        # Loss with pos_weight=10 (emphasize rare positives)
        loss_node_10 = AnomalyBCEWithLogits(pos_weight=10.0)
        loss_node_10.parent = dummy_parent
        loss_10, _ = loss_node_10.compute_loss(parent_output, labels=labels)
        
        # Higher pos_weight should give different loss
        assert not torch.allclose(loss_1, loss_10)
    
    def test_no_labels_raises_error(self, dummy_parent):
        """Test that missing labels raises error."""
        loss_node = AnomalyBCEWithLogits()
        loss_node.parent = dummy_parent
        
        scores = torch.randn(2, 10, 10)
        parent_output = (scores, None, None)
        
        with pytest.raises(ValueError, match="Labels are required"):
            loss_node.compute_loss(parent_output, labels=None)


class TestMSEReconstructionLoss:
    """Tests for MSEReconstructionLoss."""
    
    def test_initialization(self):
        """Test MSEReconstructionLoss initialization."""
        loss_node = MSEReconstructionLoss(reduction="mean")
        assert loss_node.reduction == "mean"
        assert isinstance(loss_node, LossNode)
    
    def test_compute_loss_with_labels(self, dummy_parent):
        """Test MSE loss with target in labels."""
        loss_node = MSEReconstructionLoss()
        loss_node.parent = dummy_parent
        
        # Create reconstruction and target
        reconstruction = torch.randn(2, 10, 10, 5, requires_grad=True)  # Enable gradient tracking
        target = torch.randn(2, 10, 10, 5)
        
        parent_output = (reconstruction, None, None)
        
        # Compute loss
        loss, info = loss_node.compute_loss(parent_output, labels=target)
        
        # Check loss
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.requires_grad  # Now should be True since reconstruction requires grad
        
        # Check info
        assert "mse_loss" in info
        assert "snr_db" in info
    
    def test_compute_loss_with_metadata(self, dummy_parent):
        """Test MSE loss with target in metadata."""
        loss_node = MSEReconstructionLoss()
        loss_node.parent = dummy_parent
        
        reconstruction = torch.randn(2, 10, 10, 5)
        target = torch.randn(2, 10, 10, 5)
        metadata = {"original": target}
        
        parent_output = (reconstruction, None, metadata)
        
        # Compute loss
        loss, info = loss_node.compute_loss(parent_output, metadata=metadata)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
    
    def test_no_target_raises_error(self, dummy_parent):
        """Test that missing target raises error."""
        loss_node = MSEReconstructionLoss()
        loss_node.parent = dummy_parent
        
        reconstruction = torch.randn(2, 10, 10, 5)
        parent_output = (reconstruction, None, None)
        
        with pytest.raises(ValueError, match="Target required"):
            loss_node.compute_loss(parent_output, labels=None, metadata=None)
    
    def test_shape_mismatch_raises_error(self, dummy_parent):
        """Test that shape mismatch raises error."""
        loss_node = MSEReconstructionLoss()
        loss_node.parent = dummy_parent
        
        reconstruction = torch.randn(2, 10, 10, 5)
        target = torch.randn(2, 10, 10, 3)  # Wrong shape
        
        parent_output = (reconstruction, None, None)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_node.compute_loss(parent_output, labels=target)
    
    def test_perfect_reconstruction_zero_loss(self, dummy_parent):
        """Test that perfect reconstruction gives near-zero loss."""
        loss_node = MSEReconstructionLoss()
        loss_node.parent = dummy_parent
        
        target = torch.randn(2, 10, 10, 5)
        reconstruction = target.clone()
        
        parent_output = (reconstruction, None, None)
        
        loss, info = loss_node.compute_loss(parent_output, labels=target)
        
        # Loss should be very small
        assert loss.item() < 1e-10


class TestWeightedMultiLoss:
    """Tests for WeightedMultiLoss."""
    
    def test_initialization(self):
        """Test WeightedMultiLoss initialization."""
        loss_node = WeightedMultiLoss(loss_weights={"loss1": 1.0, "loss2": 0.5})
        assert loss_node.loss_weights["loss1"] == 1.0
        assert loss_node.loss_weights["loss2"] == 0.5
    
    def test_add_loss(self, dummy_parent):
        """Test adding child loss nodes."""
        multi_loss = WeightedMultiLoss()
        multi_loss.parent = dummy_parent
        
        # Create child losses
        loss1 = MSEReconstructionLoss()
        loss2 = MSEReconstructionLoss()
        
        multi_loss.add_loss("mse1", loss1, weight=1.0)
        multi_loss.add_loss("mse2", loss2, weight=0.5)
        
        assert "mse1" in multi_loss.loss_nodes
        assert "mse2" in multi_loss.loss_nodes
        assert multi_loss.loss_weights["mse1"] == 1.0
        assert multi_loss.loss_weights["mse2"] == 0.5
    
    def test_compute_weighted_sum(self, dummy_parent):
        """Test that weighted sum is computed correctly."""
        multi_loss = WeightedMultiLoss()
        multi_loss.parent = dummy_parent
        
        # Create child losses with known behavior
        reconstruction = torch.randn(2, 10, 10, 5)
        target = torch.randn(2, 10, 10, 5)
        
        # Add two MSE losses with different weights
        loss1 = MSEReconstructionLoss()
        loss1.parent = dummy_parent
        loss2 = MSEReconstructionLoss()
        loss2.parent = dummy_parent
        
        multi_loss.add_loss("loss1", loss1, weight=1.0)
        multi_loss.add_loss("loss2", loss2, weight=2.0)
        
        parent_output = (reconstruction, None, None)
        
        # Compute multi-loss
        total_loss, info = multi_loss.compute_loss(parent_output, labels=target)
        
        # Check structure
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.ndim == 0
        
        # Check info contains all child losses
        assert "loss1/mse_loss" in info
        assert "loss2/mse_loss" in info
        assert "loss1/weight" in info
        assert "loss2/weight" in info
        assert "total_loss" in info
    
    def test_empty_losses_returns_zero(self, dummy_parent):
        """Test that empty multi-loss returns zero."""
        multi_loss = WeightedMultiLoss()
        multi_loss.parent = dummy_parent
        
        reconstruction = torch.randn(2, 10, 10, 5)
        parent_output = (reconstruction, None, None)
        
        total_loss, info = multi_loss.compute_loss(parent_output)
        
        # Should return zero
        assert total_loss.item() == 0.0


class TestLossNodeProtocol:
    """Tests for loss node protocol compliance."""
    
    def test_all_losses_are_loss_nodes(self):
        """Test that all loss classes inherit from LossNode."""
        loss_classes = [
            OrthogonalityLoss,
            AnomalyBCEWithLogits,
            MSEReconstructionLoss,
            WeightedMultiLoss,
        ]
        
        for loss_class in loss_classes:
            assert issubclass(loss_class, LossNode)
    
    def test_all_losses_have_compute_loss(self):
        """Test that all losses implement compute_loss."""
        loss_classes = [
            OrthogonalityLoss(weight=1.0),
            AnomalyBCEWithLogits(),
            MSEReconstructionLoss(),
            WeightedMultiLoss(),
        ]
        
        for loss_node in loss_classes:
            assert hasattr(loss_node, 'compute_loss')
            assert callable(loss_node.compute_loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
