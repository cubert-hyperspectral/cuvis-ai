"""Mock nodes for testing serialization patterns."""

import pytest
import torch
import torch.nn as nn

from cuvis_ai.node import Node
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.utils.types import InputStream


class MockStatisticalTrainableNode(Node):
    """Mock node with both buffers and parameters for testing serialization."""

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input data [B, H, W, C]",
        )
    }

    OUTPUT_SPECS = {
        "output": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Processed output [B, H, W, C]",
        )
    }

    def __init__(self, input_dim: int = 10, hidden_dim: int = 5, **kwargs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)

        # Statistical state (buffers) - fitted during statistical training
        # Initialize as zero tensors so load_state_dict works properly
        self.register_buffer("fitted_mean", torch.zeros(input_dim))
        self.register_buffer("fitted_std", torch.ones(input_dim))
        self.register_buffer("fitted_transform", torch.zeros(hidden_dim, input_dim))

        # Trainable parameters (start as buffers, become parameters after unfreeze)
        # Note: These operate on hidden_dim outputs from fitted_transform
        initial_weights = torch.randn(hidden_dim, hidden_dim) * 0.01
        initial_bias = torch.zeros(hidden_dim)
        self.register_buffer("linear_weight", initial_weights)
        self.register_buffer("linear_bias", initial_bias)

        # Non-persistent buffer (won't be saved)
        self.register_buffer("temp_cache", torch.zeros(10), persistent=False)

        self._statistically_initialized = False

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Statistical initialization - compute mean, std, and PCA-like transform."""
        all_data = []

        for batch in input_stream:
            data = batch.get("data")
            if data is not None:
                # Flatten spatial dimensions: [B, H, W, C] -> [B*H*W, C]
                flat_data = data.reshape(-1, data.shape[-1])
                all_data.append(flat_data)

        if not all_data:
            raise RuntimeError("No data provided for fitting")

        # Concatenate all data
        X = torch.cat(all_data, dim=0)  # [N, C]

        # Compute statistics
        self.fitted_mean = X.mean(dim=0)  # [C]
        self.fitted_std = X.std(dim=0) + 1e-6  # [C]

        # Compute simple PCA-like transform (just random projection for testing)
        U, _, _ = torch.svd(X.T @ X)
        self.fitted_transform = U[: self.hidden_dim, :].clone()  # [hidden_dim, input_dim]

        self._statistically_initialized = True

    def unfreeze(self) -> None:
        """Convert trainable buffers to nn.Parameters."""
        if self.linear_weight is not None and self.linear_bias is not None:
            # Convert buffers to parameters
            self.linear_weight = nn.Parameter(self.linear_weight.clone())
            self.linear_bias = nn.Parameter(self.linear_bias.clone())
        super().unfreeze()

    def forward(self, data: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        """Forward pass using both statistical and trainable components."""
        if not self._statistically_initialized:
            raise RuntimeError("Node must be initialized before forward pass")

        B, H, W, C = data.shape
        flat_data = data.reshape(-1, C)  # [B*H*W, C]

        # Apply statistical normalization
        normalized = (flat_data - self.fitted_mean) / self.fitted_std

        # Apply statistical transform
        transformed = normalized @ self.fitted_transform.T  # [B*H*W, hidden_dim]

        # Apply trainable linear layer
        output = transformed @ self.linear_weight.T + self.linear_bias  # [B*H*W, hidden_dim]

        # Reshape back
        output = output.reshape(B, H, W, self.hidden_dim)

        return {"output": output}


@pytest.fixture
def mock_statistical_trainable_node():
    """Factory fixture for creating MockStatisticalTrainableNode instances."""

    def _create_node(input_dim: int = 10, hidden_dim: int = 5, **kwargs):
        return MockStatisticalTrainableNode(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)

    return _create_node
