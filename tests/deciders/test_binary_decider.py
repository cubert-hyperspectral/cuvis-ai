"""Unit tests for BinaryDecider node."""

import pytest
import torch

from cuvis_ai.deciders.binary_decider import BinaryDecider

pytestmark = pytest.mark.unit


class TestBinaryDecider:
    """Tests for BinaryDecider threshold-based classification."""

    def test_basic_thresholding_at_default(self):
        """sigmoid(0) = 0.5, so at threshold=0.5, logit=0 should be True."""
        decider = BinaryDecider(threshold=0.5)
        logits = torch.zeros(1, 2, 2, 1)
        result = decider.forward(logits=logits)
        assert result["decisions"].all()

    def test_threshold_low(self):
        """With threshold=0.3, sigmoid(0)=0.5 should still be True."""
        decider = BinaryDecider(threshold=0.3)
        logits = torch.zeros(1, 2, 2, 1)
        result = decider.forward(logits=logits)
        assert result["decisions"].all()

    def test_threshold_high(self):
        """With threshold=0.7, sigmoid(0)=0.5 should be False."""
        decider = BinaryDecider(threshold=0.7)
        logits = torch.zeros(1, 2, 2, 1)
        result = decider.forward(logits=logits)
        assert not result["decisions"].any()

    def test_output_dtype_is_bool(self):
        """Output decisions must be boolean tensors."""
        decider = BinaryDecider(threshold=0.5)
        logits = torch.randn(2, 4, 4, 1)
        result = decider.forward(logits=logits)
        assert result["decisions"].dtype == torch.bool

    def test_output_shape_matches_input(self):
        """Output shape must match input shape exactly."""
        decider = BinaryDecider(threshold=0.5)
        for shape in [(1, 4, 4, 1), (3, 8, 8, 1), (2, 16, 16, 4)]:
            logits = torch.randn(*shape)
            result = decider.forward(logits=logits)
            assert result["decisions"].shape == logits.shape

    def test_batch_processing(self):
        """Verify batch dimension is handled correctly."""
        decider = BinaryDecider(threshold=0.5)
        logits = torch.randn(4, 8, 8, 1)
        result = decider.forward(logits=logits)
        assert result["decisions"].shape[0] == 4

    def test_large_positive_logits_are_true(self):
        """Large positive logits should produce True decisions."""
        decider = BinaryDecider(threshold=0.5)
        logits = torch.full((1, 2, 2, 1), 10.0)
        result = decider.forward(logits=logits)
        assert result["decisions"].all()

    def test_large_negative_logits_are_false(self):
        """Large negative logits should produce False decisions."""
        decider = BinaryDecider(threshold=0.5)
        logits = torch.full((1, 2, 2, 1), -10.0)
        result = decider.forward(logits=logits)
        assert not result["decisions"].any()
