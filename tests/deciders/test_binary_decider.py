"""Unit tests for BinaryDecider node."""

import pytest
import torch

from cuvis_ai.node.deciders.binary_decider import BinaryDecider

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


class TestBinaryDeciderThresholdInput:
    """The optional ``[1]`` ``threshold`` input replaces the hparam for one call."""

    def test_the_input_replaces_the_hparam_for_one_call(self):
        decider = BinaryDecider(threshold=0.5)
        logits = torch.tensor([[[[0.0], [1.0]], [[2.0], [-1.0]]]])
        result = decider.forward(logits=logits, threshold=torch.tensor([0.8]))
        assert torch.equal(result["decisions"], torch.sigmoid(logits) >= 0.8)
        assert decider.threshold == 0.5
        assert decider.hparams["threshold"] == 0.5
        assert torch.equal(
            decider.forward(logits=logits)["decisions"], torch.sigmoid(logits) >= 0.5
        )

    def test_the_input_compares_in_float32_like_the_hparam(self):
        decider = BinaryDecider(threshold=0.5)
        logits = torch.full((1, 1, 1, 1), 0.75)
        cut = float(torch.sigmoid(logits).item())  # exactly the float32 probability
        result = decider.forward(logits=logits, threshold=torch.tensor([cut]))
        assert result["decisions"].all()

    @pytest.mark.parametrize(
        "value", [torch.tensor([0.4, 0.6]), torch.tensor(0.5), torch.tensor([float("nan")])]
    )
    def test_a_bad_input_is_refused_by_port_name(self, value):
        decider = BinaryDecider(threshold=0.5)
        with pytest.raises(ValueError, match="threshold"):
            decider.forward(logits=torch.zeros(1, 2, 2, 1), threshold=value)
