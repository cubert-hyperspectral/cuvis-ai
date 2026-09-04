from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cuvis_ai.node.deciders.binary_decider import QuantileBinaryDecider
from cuvis_ai.node.deciders.two_stage_decider import TwoStageBinaryDecider

pytestmark = pytest.mark.unit


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
    tensor = _make_linear_map() / 100.0  # values 0.01 .. 1.00
    # Compute what image_score will be. The gate compares in raw score space, so the
    # threshold itself carries no [0, 1] constraint.
    image_score = _compute_image_score(tensor, top_k_fraction=0.001)

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
    # image_threshold is compared in raw score space (unbounded), so values outside
    # [0, 1] are legal; only non-finite values are rejected.
    assert TwoStageBinaryDecider(image_threshold=1.5).image_threshold == 1.5
    assert TwoStageBinaryDecider(image_threshold=-0.1).image_threshold == -0.1

    with pytest.raises(ValueError, match="image_threshold must be a finite"):
        TwoStageBinaryDecider(image_threshold=float("nan"))

    with pytest.raises(ValueError, match="image_threshold must be a finite"):
        TwoStageBinaryDecider(image_threshold=float("inf"))

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


def test_two_stage_decider_pixel_threshold_exact_selection():
    """An absolute pixel_threshold flags exactly the pixels at or above it."""
    tensor = _make_linear_map() / 100.0  # values 0.01 .. 1.00
    decider = TwoStageBinaryDecider(
        image_threshold=0.5,
        top_k_fraction=0.1,
        pixel_threshold=0.75,
    )

    mask = decider.forward(logits=tensor)["decisions"]

    # 0.75 .. 1.00 inclusive = 26 pixels; boundary uses >=
    assert mask.sum().item() == 26
    assert bool(mask[0, 7, 4, 0])  # value 0.75 exactly
    assert not bool(mask[0, 7, 3, 0])  # value 0.74
    assert mask.dtype == torch.bool


def test_two_stage_decider_pixel_threshold_region_follows_anomaly_size():
    """Unlike the quantile budget, the flagged region grows with the anomaly."""
    small = torch.zeros(1, 10, 10, 1, dtype=torch.float32)
    small[0, 0, :2, 0] = 0.9  # 2 hot pixels
    large = torch.zeros(1, 10, 10, 1, dtype=torch.float32)
    large[0, :5, :, 0] = 0.9  # 50 hot pixels

    decider = TwoStageBinaryDecider(image_threshold=0.5, top_k_fraction=0.01, pixel_threshold=0.5)

    assert decider.forward(logits=small)["decisions"].sum().item() == 2
    assert decider.forward(logits=large)["decisions"].sum().item() == 50


def test_two_stage_decider_pixel_threshold_gate_still_blanks():
    """A failed image gate returns a blank mask even with pixel_threshold set."""
    tensor = _make_low_score_map()
    decider = TwoStageBinaryDecider(
        image_threshold=0.5,
        top_k_fraction=0.1,
        pixel_threshold=0.01,  # would flag pixels, but the gate must veto first
    )

    mask = decider.forward(logits=tensor)["decisions"]

    assert mask.sum().item() == 0
    assert mask.dtype == torch.bool


def test_two_stage_decider_pixel_threshold_none_matches_quantile_behavior():
    """pixel_threshold=None preserves the per-frame quantile behavior exactly."""
    tensor = _make_high_score_map()
    with_default = TwoStageBinaryDecider(image_threshold=0.5, top_k_fraction=0.1, quantile=0.995)
    with_explicit_none = TwoStageBinaryDecider(
        image_threshold=0.5, top_k_fraction=0.1, quantile=0.995, pixel_threshold=None
    )

    assert torch.equal(
        with_default.forward(logits=tensor)["decisions"],
        with_explicit_none.forward(logits=tensor)["decisions"],
    )


def test_two_stage_decider_pixel_threshold_multi_channel():
    """Multi-channel scores reduce to the per-pixel max before the absolute cutoff."""
    tensor = torch.zeros(1, 10, 10, 3, dtype=torch.float32)
    tensor[0, 0, 0, 1] = 0.9  # only one channel carries the hot value
    tensor[0, 0, 1, :] = 0.4  # below the cutoff on every channel

    decider = TwoStageBinaryDecider(image_threshold=0.5, top_k_fraction=0.01, pixel_threshold=0.5)

    mask = decider.forward(logits=tensor)["decisions"]

    assert mask.shape == (1, 10, 10, 1)
    assert bool(mask[0, 0, 0, 0])
    assert not bool(mask[0, 0, 1, 0])
    assert mask.sum().item() == 1


def test_two_stage_decider_pixel_threshold_validation():
    """Non-finite pixel_threshold values are rejected; None is accepted."""
    with pytest.raises(ValueError, match="pixel_threshold must be a finite"):
        TwoStageBinaryDecider(pixel_threshold=float("nan"))

    with pytest.raises(ValueError, match="pixel_threshold must be a finite"):
        TwoStageBinaryDecider(pixel_threshold=float("inf"))

    assert TwoStageBinaryDecider(pixel_threshold=None).pixel_threshold is None


def test_two_stage_decider_pixel_threshold_serialization_roundtrip(tmp_path: Path):
    """pixel_threshold survives the hparams round-trip and reproduces decisions."""
    tensor = _make_high_score_map()
    decider = TwoStageBinaryDecider(
        image_threshold=0.5,
        top_k_fraction=0.1,
        quantile=0.995,
        pixel_threshold=0.85,
    )

    original = decider.forward(logits=tensor)["decisions"]

    state_path = tmp_path / "decider_state.pt"
    torch.save(decider.state_dict(), state_path)

    assert decider.hparams["pixel_threshold"] == 0.85
    restored = TwoStageBinaryDecider(**decider.hparams)
    restored.load_state_dict(torch.load(state_path))

    assert restored.pixel_threshold == 0.85
    assert torch.equal(original, restored.forward(logits=tensor)["decisions"])


def test_two_stage_decider_default_gate_is_off():
    """No image_threshold means no gate: the default constructor never blanks a frame."""
    decider = TwoStageBinaryDecider()
    assert decider.image_threshold is None
    assert decider.hparams["image_threshold"] is None

    mask = decider.forward(logits=_make_low_score_map())["decisions"]
    assert mask.sum().item() > 0  # the quantile fallback flags the top 0.5 %


def test_two_stage_decider_gate_off_matches_quantile_decider_exactly():
    """With the gate off and no pixel_threshold, decisions equal QuantileBinaryDecider's."""
    generator = torch.Generator().manual_seed(1234)
    tensor = torch.rand(3, 16, 12, 1, generator=generator, dtype=torch.float32)
    tensor[1] *= 0.05  # a frame every finite gate would blank

    for quantile in (0.9, 0.995):
        two_stage = TwoStageBinaryDecider(image_threshold=None, quantile=quantile)
        reference = QuantileBinaryDecider(quantile=quantile)
        assert torch.equal(
            two_stage.forward(logits=tensor)["decisions"],
            reference.forward(logits=tensor)["decisions"],
        )


def test_two_stage_decider_gate_off_pixel_threshold_applies_to_every_frame():
    """Gate off + absolute pixel_threshold: the cutoff decides alone, no frame is blanked."""
    tensor = torch.zeros(2, 10, 10, 1, dtype=torch.float32)
    tensor[0, -1, :, 0] = torch.linspace(0.8, 1.0, 10)  # would pass a 0.5 gate
    tensor[1, 0, :3, 0] = 0.03  # would fail it

    decider = TwoStageBinaryDecider(image_threshold=None, top_k_fraction=0.1, pixel_threshold=0.02)
    mask = decider.forward(logits=tensor)["decisions"]

    assert mask[0].sum().item() == 10
    assert mask[1].sum().item() == 3

    gated = TwoStageBinaryDecider(image_threshold=0.5, top_k_fraction=0.1, pixel_threshold=0.02)
    assert gated.forward(logits=tensor)["decisions"][1].sum().item() == 0


def test_two_stage_decider_none_gate_hparams_roundtrip():
    """image_threshold=None survives the hparams round trip and stays off."""
    decider = TwoStageBinaryDecider(image_threshold=None, quantile=0.99, pixel_threshold=None)
    assert decider.hparams["image_threshold"] is None
    assert decider.hparams["pixel_threshold"] is None

    restored = TwoStageBinaryDecider(**decider.hparams)
    assert restored.image_threshold is None
    tensor = _make_low_score_map()
    assert torch.equal(
        decider.forward(logits=tensor)["decisions"], restored.forward(logits=tensor)["decisions"]
    )


@pytest.mark.parametrize("bad", ["0.5", True, [0.5], float("nan"), float("inf")])
def test_two_stage_decider_non_numeric_image_threshold_rejected(bad):
    """Anything but a finite real number or None is refused, naming the hparam."""
    with pytest.raises(ValueError, match="image_threshold must be a finite number or None"):
        TwoStageBinaryDecider(image_threshold=bad)


@pytest.mark.parametrize("bad", ["0.1", False, [0.1]])
def test_two_stage_decider_non_numeric_pixel_threshold_rejected(bad):
    """pixel_threshold gets the same by-name refusal as image_threshold."""
    with pytest.raises(ValueError, match="pixel_threshold must be a finite number or None"):
        TwoStageBinaryDecider(pixel_threshold=bad)
