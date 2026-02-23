"""Unit tests for RXPerBatch stateless anomaly detector."""

import pytest
import torch

from cuvis_ai.anomaly.rx_detector import RXPerBatch

pytestmark = pytest.mark.unit


class TestRXPerBatch:
    """Tests for RXPerBatch per-image Mahalanobis distance detector."""

    def test_output_shape(self):
        """Output should be (B, H, W, 1)."""
        rx = RXPerBatch(eps=1e-6)
        data = torch.randn(2, 8, 8, 5)
        result = rx.forward(data=data)
        assert result["scores"].shape == (2, 8, 8, 1)

    def test_stateless_no_initialization_needed(self):
        """RXPerBatch should work without statistical_initialization."""
        rx = RXPerBatch(eps=1e-6)
        data = torch.randn(1, 4, 4, 3)
        result = rx.forward(data=data)
        assert "scores" in result

    def test_per_image_independence(self):
        """Score for image i should not depend on image j."""
        rx = RXPerBatch(eps=1e-6)
        data_single = torch.randn(1, 6, 6, 4)
        data_batch = torch.cat([data_single, torch.randn(1, 6, 6, 4)], dim=0)

        scores_single = rx.forward(data=data_single)["scores"]
        scores_batch = rx.forward(data=data_batch)["scores"]

        assert torch.allclose(scores_single[0], scores_batch[0], atol=1e-5)

    def test_scores_non_negative(self):
        """Squared Mahalanobis distances must be non-negative."""
        rx = RXPerBatch(eps=1e-6)
        data = torch.randn(3, 8, 8, 10)
        result = rx.forward(data=data)
        assert (result["scores"] >= 0).all()

    def test_output_dtype_float32(self):
        """Output scores should be float32."""
        rx = RXPerBatch(eps=1e-6)
        data = torch.randn(1, 4, 4, 5)
        result = rx.forward(data=data)
        assert result["scores"].dtype == torch.float32

    def test_single_pixel(self):
        """Edge case: 1x1 spatial dimensions."""
        rx = RXPerBatch(eps=1e-6)
        data = torch.randn(1, 1, 1, 3)
        result = rx.forward(data=data)
        assert result["scores"].shape == (1, 1, 1, 1)
