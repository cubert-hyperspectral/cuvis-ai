"""Tests for loss leaf nodes."""

import pytest
import torch
import torch.nn.functional as F
from cuvis_ai_core.node.node import Node

from cuvis_ai.node.losses import (
    AnomalyBCEWithLogits,
    DistinctnessLoss,
    MSEReconstructionLoss,
    OrthogonalityLoss,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def distinctness_loss_node():
    """Create a DistinctnessLoss node for testing."""
    return DistinctnessLoss(weight=1.0, eps=1e-6)


@pytest.fixture
def selection_weights_3x5():
    """Create 3x5 selection weights for testing."""
    return torch.randn(3, 5, requires_grad=True)


@pytest.fixture
def selection_weights_various():
    """Create selection weights with various channel counts."""
    return {
        "2_channels": torch.randn(2, 10, requires_grad=True),
        "3_channels": torch.randn(3, 15, requires_grad=True),
        "5_channels": torch.randn(5, 20, requires_grad=True),
        "10_channels": torch.randn(10, 30, requires_grad=True),
    }


class TestOrthogonalityLoss:
    """Tests for OrthogonalityLoss."""

    def test_initialization(self):
        """Test OrthogonalityLoss initialization."""
        loss_node = OrthogonalityLoss(weight=2.0)
        assert loss_node.weight == 2.0
        assert isinstance(loss_node, Node)

    def test_compute_loss(self, trainable_pca):
        """Test orthogonality loss computation."""
        loss_node = OrthogonalityLoss(weight=1.0)

        # Get components from the PCA node (private attribute, not port)
        components = trainable_pca._components

        # Compute loss using forward()
        outputs = loss_node.forward(components=components)

        # Check loss is scalar tensor
        loss = outputs["loss"]
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_loss_weighting(self, trainable_pca):
        """Test that weight parameter scales loss correctly."""
        loss_node_1x = OrthogonalityLoss(weight=1.0)
        loss_node_2x = OrthogonalityLoss(weight=2.0)

        components = trainable_pca._components

        outputs_1x = loss_node_1x.forward(components=components)
        outputs_2x = loss_node_2x.forward(components=components)

        loss_1x = outputs_1x["loss"]
        loss_2x = outputs_2x["loss"]

        # 2x weight should give approximately 2x loss
        assert torch.allclose(loss_2x, loss_1x * 2.0, rtol=1e-5)

    def test_loss_decreases_with_training(self, trainable_pca):
        """Test that loss can decrease with gradient updates."""
        import copy

        pca = copy.deepcopy(trainable_pca)
        loss_node = OrthogonalityLoss(weight=1.0)

        # Degrade orthogonality so there's something to optimize
        pca._components.data += 0.1 * torch.randn_like(pca._components)

        optimizer = torch.optim.SGD(pca.parameters(), lr=0.01)

        # Initial loss (should be non-trivial now)
        outputs_initial = loss_node.forward(components=pca._components)
        initial_value = outputs_initial["loss"].item()

        # Loss should be significantly above zero
        assert initial_value > 1e-6

        # Train for a few steps
        for _ in range(10):
            optimizer.zero_grad()
            outputs = loss_node.forward(components=pca._components)
            outputs["loss"].backward()
            optimizer.step()

        # Final loss
        outputs_final = loss_node.forward(components=pca._components)
        final_value = outputs_final["loss"].item()

        # Loss should decrease (components becoming more orthogonal)
        assert final_value < initial_value


class TestAnomalyBCEWithLogits:
    """Tests for AnomalyBCEWithLogits loss."""

    def test_initialization(self):
        """Test AnomalyBCEWithLogits initialization."""
        loss_node = AnomalyBCEWithLogits(pos_weight=2.0, reduction="mean")
        assert loss_node.pos_weight == 2.0
        assert loss_node.reduction == "mean"
        assert isinstance(loss_node, Node)

    def test_compute_loss_with_logits(self):
        """Test BCE loss computation with logits."""
        loss_node = AnomalyBCEWithLogits(pos_weight=1.0)

        # Create dummy scores (logits) and labels
        B, H, W = 2, 10, 10
        predictions = torch.randn(B, H, W, 1, requires_grad=True)  # Logits with gradient tracking
        targets = torch.randint(0, 2, (B, H, W, 1)).bool()

        # Compute loss using forward()
        outputs = loss_node.forward(predictions=predictions, targets=targets)

        # Check loss
        loss = outputs["loss"]
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.requires_grad  # Now should be True since inputs require grad

    def test_compute_loss_with_4d_tensors(self):
        """Test BCE loss with 4D tensors [B, H, W, 1]."""
        loss_node = AnomalyBCEWithLogits()

        # Create 4D tensors
        B, H, W = 2, 10, 10
        predictions = torch.randn(B, H, W, 1)
        targets = torch.randint(0, 2, (B, H, W, 1)).bool()

        # Should handle 4D tensors correctly
        outputs = loss_node.forward(predictions=predictions, targets=targets)

        loss = outputs["loss"]
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_pos_weight_effect(self):
        """Test that pos_weight affects loss calculation."""
        # Create imbalanced data (more negatives)
        B, H, W = 2, 10, 10
        predictions = torch.randn(B, H, W, 1)
        targets = torch.zeros(B, H, W, 1, dtype=torch.bool)
        targets[0, 0, 0, 0] = True  # Only one positive example

        # Loss with pos_weight=1
        loss_node_1 = AnomalyBCEWithLogits(pos_weight=1.0)
        outputs_1 = loss_node_1.forward(predictions=predictions, targets=targets)
        loss_1 = outputs_1["loss"]

        # Loss with pos_weight=10 (emphasize rare positives)
        loss_node_10 = AnomalyBCEWithLogits(pos_weight=10.0)
        outputs_10 = loss_node_10.forward(predictions=predictions, targets=targets)
        loss_10 = outputs_10["loss"]

        # Higher pos_weight should give different loss
        assert not torch.allclose(loss_1, loss_10)

    def test_no_labels_raises_error(self):
        """Test that missing labels raises error."""
        loss_node = AnomalyBCEWithLogits()

        predictions = torch.randn(2, 10, 10, 1)

        # forward() with missing required input should raise TypeError
        with pytest.raises(TypeError):
            loss_node.forward(predictions=predictions)


class TestMSEReconstructionLoss:
    """Tests for MSEReconstructionLoss."""

    def test_initialization(self):
        """Test MSEReconstructionLoss initialization."""
        loss_node = MSEReconstructionLoss(reduction="mean")
        assert loss_node.reduction == "mean"
        assert isinstance(loss_node, Node)

    def test_compute_loss_with_labels(self):
        """Test MSE loss with target."""
        loss_node = MSEReconstructionLoss()

        # Create reconstruction and target
        reconstruction = torch.randn(2, 10, 10, 5, requires_grad=True)  # Enable gradient tracking
        target = torch.randn(2, 10, 10, 5)

        # Compute loss using forward()
        outputs = loss_node.forward(reconstruction=reconstruction, target=target)

        # Check loss
        loss = outputs["loss"]
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.requires_grad  # Now should be True since reconstruction requires grad

    def test_compute_loss_with_target(self):
        """Test MSE loss with explicit target parameter."""
        loss_node = MSEReconstructionLoss()

        reconstruction = torch.randn(2, 10, 10, 5)
        target = torch.randn(2, 10, 10, 5)

        # Compute loss using forward()
        outputs = loss_node.forward(reconstruction=reconstruction, target=target)

        loss = outputs["loss"]
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_no_target_raises_error(self):
        """Test that missing target raises error."""
        loss_node = MSEReconstructionLoss()

        reconstruction = torch.randn(2, 10, 10, 5)

        # forward() with missing required input should raise TypeError
        with pytest.raises(TypeError):
            loss_node.forward(reconstruction=reconstruction)

    def test_shape_mismatch_raises_error(self):
        """Test that shape mismatch raises error."""
        loss_node = MSEReconstructionLoss()

        reconstruction = torch.randn(2, 10, 10, 5)
        target = torch.randn(2, 10, 10, 3)  # Wrong shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_node.forward(reconstruction=reconstruction, target=target)

    def test_perfect_reconstruction_zero_loss(self):
        """Test that perfect reconstruction gives near-zero loss."""
        loss_node = MSEReconstructionLoss()

        target = torch.randn(2, 10, 10, 5)
        reconstruction = target.clone()

        outputs = loss_node.forward(reconstruction=reconstruction, target=target)
        loss = outputs["loss"]

        # Loss should be very small
        assert loss.item() < 1e-10


class TestDistinctnessLoss:
    """Tests for DistinctnessLoss with matrix vs loop comparison."""

    def test_matrix_vs_loop_computation(self, distinctness_loss_node, selection_weights_various):
        """Test that matrix computation produces identical results to loop computation."""
        for name, weights in selection_weights_various.items():
            # Compute loss using the optimized node (matrix-based)
            node_output = distinctness_loss_node.forward(selection_weights=weights)
            matrix_loss = node_output["loss"]

            # Compute loss using loop-based approach for comparison
            w_norm = F.normalize(weights, p=2, dim=-1, eps=1e-6)
            num_channels = w_norm.shape[0]

            if num_channels < 2:
                expected_loss = torch.zeros((), device=weights.device, dtype=weights.dtype)
            else:
                cos_sum = torch.zeros((), device=w_norm.device, dtype=w_norm.dtype)
                num_pairs = 0

                for i in range(num_channels):
                    for j in range(i + 1, num_channels):
                        cos_ij = F.cosine_similarity(w_norm[i], w_norm[j], dim=0, eps=1e-6)
                        cos_sum = cos_sum + cos_ij
                        num_pairs += 1

                expected_loss = (
                    cos_sum / float(num_pairs)
                    if num_pairs > 0
                    else torch.zeros((), device=w_norm.device, dtype=w_norm.dtype)
                )

            # Compare results
            assert torch.allclose(matrix_loss, expected_loss, rtol=1e-5, atol=1e-7), (
                f"Failed for {name}"
            )

    def test_edge_cases(self, distinctness_loss_node):
        """Test edge cases like single channel."""
        # Single channel - should return zero loss
        single_channel = torch.randn(1, 10, requires_grad=True)
        output = distinctness_loss_node.forward(selection_weights=single_channel)
        assert output["loss"].item() == 0.0

        # Zero channels - should return zero loss
        zero_channels = torch.randn(0, 10, requires_grad=True)
        output = distinctness_loss_node.forward(selection_weights=zero_channels)
        assert output["loss"].item() == 0.0

    def test_gradient_flow(self, distinctness_loss_node, selection_weights_3x5):
        """Test that gradients flow correctly through the loss."""
        weights = selection_weights_3x5
        output = distinctness_loss_node.forward(selection_weights=weights)
        loss = output["loss"]

        # Should be able to compute gradients
        loss.backward()
        assert weights.grad is not None
        assert torch.any(weights.grad != 0)

    def test_weight_parameter(self, distinctness_loss_node, selection_weights_3x5):
        """Test that weight parameter scales loss correctly."""
        weights = selection_weights_3x5

        # Test with weight=1.0
        output_1x = distinctness_loss_node.forward(selection_weights=weights)
        loss_1x = output_1x["loss"]

        # Test with weight=2.0
        loss_node_2x = DistinctnessLoss(weight=2.0, eps=1e-6)
        output_2x = loss_node_2x.forward(selection_weights=weights)
        loss_2x = output_2x["loss"]

        # 2x weight should give exactly 2x loss
        assert torch.allclose(loss_2x, loss_1x * 2.0, rtol=1e-6)


class TestLossNodeProtocol:
    """Tests for loss node protocol compliance."""

    def test_all_losses_are_loss_nodes(self):
        """Test that all loss classes inherit from LossNode."""
        loss_classes = [
            OrthogonalityLoss,
            AnomalyBCEWithLogits,
            MSEReconstructionLoss,
            DistinctnessLoss,
        ]

        for loss_class in loss_classes:
            assert issubclass(loss_class, Node)

    def test_all_losses_have_forward(self):
        """Test that all losses implement forward."""
        loss_classes = [
            OrthogonalityLoss(weight=1.0),
            AnomalyBCEWithLogits(),
            MSEReconstructionLoss(),
            DistinctnessLoss(),
        ]

        for loss_node in loss_classes:
            assert hasattr(loss_node, "forward")
            assert callable(loss_node.forward)
            assert "loss" in loss_node.OUTPUT_SPECS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
