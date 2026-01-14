from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cuvis_ai.deciders.two_stage_decider import TwoStageBinaryDecider


def _make_linear_map() -> torch.Tensor:
    """Create a linear map from 1 to 100, reshaped to [1, 10, 10, 1]."""
    return torch.arange(1, 101, dtype=torch.float32).reshape(1, 10, 10, 1)


def _make_high_score_map() -> torch.Tensor:
    """Create a map with high scores that should pass the gate."""
    tensor = torch.zeros(1, 10, 10, 1, dtype=torch.float32)
    # Set top 10% to high values (0.8-1.0)
    tensor[0, -1, :, 0] = torch.linspace(0.8, 1.0, 10)
    return tensor


def _make_low_score_map() -> torch.Tensor:
    """Create a map with low scores that should fail the gate."""
    tensor = torch.zeros(1, 10, 10, 1, dtype=torch.float32)
    # Set all values to low (0.01-0.05)
    tensor[0, :, :, 0] = torch.linspace(0.01, 0.05, 100).reshape(10, 10)
    return tensor


def _compute_image_score(tensor: torch.Tensor, top_k_fraction: float) -> float:
    """Helper to compute image score manually."""
    # Remove batch dimension for computation
    scores = tensor[0]  # [H, W, C]
    if scores.dim() == 3:
        pixel_scores = scores.max(dim=-1)[0]  # [H, W]
    else:
        pixel_scores = scores
    flat = pixel_scores.reshape(-1)
    k = max(
        1, int(torch.ceil(torch.tensor(flat.numel() * top_k_fraction, dtype=torch.float32)).item())
    )
    topk_vals, _ = torch.topk(flat, k)
    return topk_vals.mean().item()


def test_two_stage_decider_gate_passes_and_applies_quantile():
    """Test that when gate passes, quantile thresholding is applied."""
    tensor = _make_high_score_map()
    # High scores should pass gate (image_score ~0.9 > 0.5)
    decider = TwoStageBinaryDecider(
        image_threshold=0.5,
        top_k_fraction=0.1,  # Top 10% of pixels
        quantile=0.995,
    )

    mask = decider.forward(logits=tensor)["decisions"]

    # Should not be all False (gate passed, so quantile thresholding applied)
    assert mask.sum().item() > 0
    assert mask.dtype == torch.bool
    assert mask.shape == tensor.shape


def test_two_stage_decider_gate_fails_returns_blank_mask():
    """Test that when gate fails, blank mask is returned."""
    tensor = _make_low_score_map()
    # Low scores should fail gate (image_score ~0.03 < 0.5)
    decider = TwoStageBinaryDecider(
        image_threshold=0.5,
        top_k_fraction=0.1,
        quantile=0.995,
    )

    mask = decider.forward(logits=tensor)["decisions"]

    # Should be all False (gate failed, blank mask returned)
    assert mask.sum().item() == 0
    assert mask.dtype == torch.bool
    assert mask.shape == tensor.shape


def test_two_stage_decider_gate_boundary_condition():
    """Test gate behavior at threshold boundary."""
    # Use normalized tensor (values in [0, 1]) to match threshold constraint
    tensor = _make_linear_map() / 100.0  # Normalize to [0.01, 1.0]
    # Compute what image_score will be
    image_score = _compute_image_score(tensor, top_k_fraction=0.001)

    # Ensure image_score is within [0, 1] for threshold validation
    assert 0.0 <= image_score <= 1.0, f"image_score {image_score} must be in [0, 1]"

    # Set threshold exactly at image_score
    decider = TwoStageBinaryDecider(
        image_threshold=image_score,
        top_k_fraction=0.001,
        quantile=0.995,
    )

    mask = decider.forward(logits=tensor)["decisions"]

    # Should pass (>= threshold) and apply quantile thresholding
    assert mask.sum().item() > 0
    assert mask.dtype == torch.bool


def test_two_stage_decider_different_quantiles():
    """Test that different quantiles produce different masks."""
    tensor = _make_high_score_map()

    decider_high = TwoStageBinaryDecider(
        image_threshold=0.5,
        top_k_fraction=0.1,
        quantile=0.99,  # Lower quantile = more pixels selected
    )
    decider_low = TwoStageBinaryDecider(
        image_threshold=0.5,
        top_k_fraction=0.1,
        quantile=0.999,  # Higher quantile = fewer pixels selected
    )

    mask_high = decider_high.forward(logits=tensor)["decisions"]
    mask_low = decider_low.forward(logits=tensor)["decisions"]

    # Higher quantile should select fewer or equal pixels
    assert mask_low.sum().item() <= mask_high.sum().item()
    assert mask_high.dtype == torch.bool
    assert mask_low.dtype == torch.bool


def test_two_stage_decider_different_top_k_fractions():
    """Test that different top_k_fractions affect gate decision."""
    tensor = _make_linear_map()

    # With very small top_k_fraction, image_score will be high (top pixels)
    decider_small = TwoStageBinaryDecider(
        image_threshold=0.5,
        top_k_fraction=0.001,  # Top 0.1% - should be high
        quantile=0.995,
    )

    # With large top_k_fraction, image_score will be lower (includes more pixels)
    decider_large = TwoStageBinaryDecider(
        image_threshold=0.5,
        top_k_fraction=0.5,  # Top 50% - should be lower
        quantile=0.995,
    )

    mask_small = decider_small.forward(logits=tensor)["decisions"]
    mask_large = decider_large.forward(logits=tensor)["decisions"]

    # Both should produce boolean masks
    assert mask_small.dtype == torch.bool
    assert mask_large.dtype == torch.bool
    assert mask_small.shape == tensor.shape
    assert mask_large.shape == tensor.shape


def test_two_stage_decider_multi_channel():
    """Test with multi-channel input (H, W, C where C > 1)."""
    # Create [1, 10, 10, 3] tensor
    tensor = torch.rand(1, 10, 10, 3, dtype=torch.float32) * 0.1
    # Set some high values in one channel
    tensor[0, -1, :, 0] = torch.linspace(0.8, 1.0, 10)

    decider = TwoStageBinaryDecider(
        image_threshold=0.5,
        top_k_fraction=0.1,
        quantile=0.995,
    )

    mask = decider.forward(logits=tensor)["decisions"]

    assert mask.dtype == torch.bool
    assert mask.shape == (1, 10, 10, 1)  # Should reduce to single channel
    # Should pass gate and apply quantile (high scores in channel 0)
    assert mask.sum().item() > 0


def test_two_stage_decider_batch_processing():
    """Test that batch processing works correctly."""
    # Create batch of 2: one high score, one low score
    tensor = torch.zeros(2, 10, 10, 1, dtype=torch.float32)
    tensor[0, -1, :, 0] = torch.linspace(0.8, 1.0, 10)  # High scores
    tensor[1, :, :, 0] = 0.01  # Low scores

    decider = TwoStageBinaryDecider(
        image_threshold=0.5,
        top_k_fraction=0.1,
        quantile=0.995,
    )

    mask = decider.forward(logits=tensor)["decisions"]

    assert mask.shape == (2, 10, 10, 1)
    assert mask.dtype == torch.bool
    # First batch item should have detections (gate passed)
    assert mask[0].sum().item() > 0
    # Second batch item should be blank (gate failed)
    assert mask[1].sum().item() == 0


def test_two_stage_decider_validation_errors():
    """Test that invalid parameters raise errors."""
    # Invalid image_threshold
    with pytest.raises(ValueError, match="image_threshold must be within"):
        TwoStageBinaryDecider(image_threshold=1.5)

    with pytest.raises(ValueError, match="image_threshold must be within"):
        TwoStageBinaryDecider(image_threshold=-0.1)

    # Invalid top_k_fraction
    with pytest.raises(ValueError, match="top_k_fraction must be in"):
        TwoStageBinaryDecider(top_k_fraction=0.0)

    with pytest.raises(ValueError, match="top_k_fraction must be in"):
        TwoStageBinaryDecider(top_k_fraction=1.5)

    # Invalid quantile
    with pytest.raises(ValueError, match="quantile must be within"):
        TwoStageBinaryDecider(quantile=1.5)

    with pytest.raises(ValueError, match="quantile must be within"):
        TwoStageBinaryDecider(quantile=-0.1)


def test_two_stage_decider_serialization_roundtrip(tmp_path: Path):
    """Test serialization and deserialization."""
    tensor = _make_high_score_map()
    decider = TwoStageBinaryDecider(
        image_threshold=0.5,
        top_k_fraction=0.1,
        quantile=0.995,
        reduce_dims=None,
    )

    original = decider.forward(logits=tensor)["decisions"]

    # Save state
    state_path = tmp_path / "decider_state.pt"
    torch.save(decider.state_dict(), state_path)

    # Verify hparams (may not exist if Serializable isn't set up, so check first)
    if hasattr(decider, "hparams"):
        assert decider.hparams["image_threshold"] == decider.image_threshold
        assert decider.hparams["top_k_fraction"] == decider.top_k_fraction
        assert decider.hparams["quantile"] == decider.quantile
        assert decider.hparams["reduce_dims"] == decider.reduce_dims

        # Restore using hparams
        restored = TwoStageBinaryDecider(**decider.hparams)
    else:
        # Fallback: restore using explicit parameters
        restored = TwoStageBinaryDecider(
            image_threshold=decider.image_threshold,
            top_k_fraction=decider.top_k_fraction,
            quantile=decider.quantile,
            reduce_dims=decider.reduce_dims,
        )

    state = torch.load(state_path)
    restored.load_state_dict(state)

    recreated = restored.forward(logits=tensor)["decisions"]
    assert torch.equal(original, recreated)
    assert restored.image_threshold == decider.image_threshold
    assert restored.top_k_fraction == decider.top_k_fraction
    assert restored.quantile == decider.quantile
    assert restored.reduce_dims == decider.reduce_dims


def test_two_stage_decider_edge_case_all_zeros():
    """Test with all-zero input."""
    tensor = torch.zeros(1, 10, 10, 1, dtype=torch.float32)
    decider = TwoStageBinaryDecider(
        image_threshold=0.5,
        top_k_fraction=0.1,
        quantile=0.995,
    )

    mask = decider.forward(logits=tensor)["decisions"]

    # All zeros -> image_score = 0 -> gate fails -> blank mask
    assert mask.sum().item() == 0
    assert mask.dtype == torch.bool


def test_two_stage_decider_edge_case_all_ones():
    """Test with all-ones input."""
    tensor = torch.ones(1, 10, 10, 1, dtype=torch.float32)
    decider = TwoStageBinaryDecider(
        image_threshold=0.5,
        top_k_fraction=0.1,
        quantile=0.995,
    )

    mask = decider.forward(logits=tensor)["decisions"]

    # All ones -> image_score = 1.0 -> gate passes -> quantile thresholding
    # With quantile=0.995, should select top 0.5% pixels
    assert mask.dtype == torch.bool
    # Should have some detections (quantile thresholding applied)
    assert mask.sum().item() > 0


def test_two_stage_decider_small_top_k_fraction():
    """Test with very small top_k_fraction (should use at least 1 pixel)."""
    tensor = _make_linear_map()
    # Very small fraction that would round to 0
    decider = TwoStageBinaryDecider(
        image_threshold=0.5,
        top_k_fraction=0.0001,  # 0.01% of 100 = 0.01, should round to 1
        quantile=0.995,
    )

    mask = decider.forward(logits=tensor)["decisions"]

    # Should still work (k should be at least 1)
    assert mask.dtype == torch.bool
    assert mask.shape == tensor.shape
