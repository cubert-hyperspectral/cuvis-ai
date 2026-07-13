"""Tests for FixedWavelengthSelector generalization to n-channel output.

Covers:
- Backward compat: 3-channel default still works exactly as before
- n-channel (e.g. 6) band stacking — shape, dtype, correct band pick
- Empty / single wavelength edge cases
- normalize_output=True with n != 3 warns ONCE at construction (not per forward)
- OUTPUT_SPECS shape (-1,-1,-1,-1) overridden on FixedWavelengthSelector only;
  the base ChannelSelectorBase stays at the tight (-1,-1,-1,3) contract so the
  sibling fixed-3-channel selectors keep their pipeline-validation safety net
- norm_mode={'statistical','running'} is rejected at construction for n != 3
  (the running/statistical paths assume 3-element buffers + reshape(-1, 3))
- band_info['normalized_output'] reflects what actually happened
  (False for n != 3 regardless of the normalize_output flag)
- band_info['strategy'] = 'baseline_false_rgb' for n==3, 'stacked_bands' otherwise
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.channel_selector import (
    ChannelSelectorBase,
    FixedWavelengthSelector,
    NormMode,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _cube(B: int = 1, H: int = 4, W: int = 4, C: int = 10) -> torch.Tensor:
    return torch.rand(B, H, W, C)


def _wavelengths(n: int = 10, lo: float = 400.0, hi: float = 900.0) -> np.ndarray:
    return np.linspace(lo, hi, n, dtype=np.float32)


# ---------------------------------------------------------------------------
# OUTPUT_SPECS shape relaxation
# ---------------------------------------------------------------------------


class TestOutputSpecsShape:
    def test_base_class_keeps_tight_3channel_contract(self) -> None:
        """ChannelSelectorBase keeps shape (-1,-1,-1,3) so the 11 sibling
        selectors (FastRGBSelector, CIRSelector, NDVI variants, etc.) — all of
        which emit exactly 3 channels — keep their pipeline-validation safety
        net. Only FixedWavelengthSelector overrides this."""
        spec = ChannelSelectorBase.OUTPUT_SPECS["rgb_image"]
        assert spec.shape == (-1, -1, -1, 3), (
            f"Expected (-1,-1,-1,3), got {spec.shape}. Loosening the base class "
            "would remove validation for every fixed-3-channel selector subclass."
        )

    def test_fixed_selector_overrides_to_n_channel(self) -> None:
        """FixedWavelengthSelector specifically overrides OUTPUT_SPECS to
        (-1,-1,-1,-1) so it can emit an n-channel stack. The base class
        is unaffected."""
        spec = FixedWavelengthSelector.OUTPUT_SPECS["rgb_image"]
        assert spec.shape == (-1, -1, -1, -1)
        # Verify base is independent of subclass override.
        assert ChannelSelectorBase.OUTPUT_SPECS["rgb_image"].shape == (-1, -1, -1, 3)
        # And the subclass spec dict still inherits the other ports from base.
        assert "band_info" in FixedWavelengthSelector.OUTPUT_SPECS


# ---------------------------------------------------------------------------
# Backward compatibility — 3-channel default
# ---------------------------------------------------------------------------


class TestThreeChannelBackwardCompat:
    def test_default_wavelengths_output_shape(self) -> None:
        """Default (650, 550, 450) must still produce [B, H, W, 3]."""
        sel = FixedWavelengthSelector(normalize_output=False)
        cube = _cube(C=10)
        wl = _wavelengths(10)
        out = sel.forward(cube, wl)
        assert out["rgb_image"].shape == (1, 4, 4, 3)

    def test_three_channel_with_normalize_uses_compose_rgb(self) -> None:
        """normalize_output=True + 3 channels → _compose_rgb is called → 0-1 range."""
        sel = FixedWavelengthSelector(
            target_wavelengths=(650.0, 550.0, 450.0),
            normalize_output=True,
            norm_mode="per_frame",
        )
        # Use a cube with known max so output lands in [0,1]
        cube = torch.rand(1, 4, 4, 10) * 500.0
        wl = _wavelengths(10)
        out = sel.forward(cube, wl)
        assert out["rgb_image"].shape == (1, 4, 4, 3)
        assert out["rgb_image"].dtype == torch.float32
        assert float(out["rgb_image"].max()) <= 1.0 + 1e-5

    def test_band_info_strategy_and_fields(self) -> None:
        sel = FixedWavelengthSelector(
            target_wavelengths=(650.0, 550.0, 450.0), normalize_output=False
        )
        cube = _cube(C=10)
        wl = _wavelengths(10)
        out = sel.forward(cube, wl)
        info = out["band_info"]
        assert info["strategy"] == "baseline_false_rgb"
        assert len(info["band_indices"]) == 3
        assert len(info["band_wavelengths_nm"]) == 3
        assert info["target_wavelengths_nm"] == [650.0, 550.0, 450.0]
        assert info["normalized_output"] is False

    def test_picks_nearest_band(self) -> None:
        """The selected band index should be the one closest to each target."""
        wl = np.array([400.0, 500.0, 600.0, 700.0], dtype=np.float32)
        sel = FixedWavelengthSelector(
            target_wavelengths=(610.0, 490.0, 410.0), normalize_output=False
        )
        cube = _cube(C=4)
        out = sel.forward(cube, wl)
        # 610 → nearest is 600 (index 2), 490 → 500 (index 1), 410 → 400 (index 0)
        assert out["band_info"]["band_indices"] == [2, 1, 0]


# ---------------------------------------------------------------------------
# n-channel generalization
# ---------------------------------------------------------------------------


class TestNChannelOutput:
    def test_six_channel_output_shape(self) -> None:
        """6-target-wavelengths → output shape [B, H, W, 6]."""
        targets = (625.0, 550.0, 450.0, 1450.0, 1200.0, 1050.0)
        sel = FixedWavelengthSelector(
            target_wavelengths=targets, normalize_output=False, norm_mode="per_frame"
        )
        cube = _cube(B=2, H=8, W=8, C=20)
        wl = np.linspace(400.0, 1500.0, 20, dtype=np.float32)
        out = sel.forward(cube, wl)
        assert out["rgb_image"].shape == (2, 8, 8, 6)

    def test_six_channel_dtype_is_float32(self) -> None:
        targets = (625.0, 550.0, 450.0, 1450.0, 1200.0, 1050.0)
        sel = FixedWavelengthSelector(
            target_wavelengths=targets, normalize_output=False, norm_mode="per_frame"
        )
        cube = _cube(C=20)
        wl = np.linspace(400.0, 1500.0, 20, dtype=np.float32)
        out = sel.forward(cube, wl)
        assert out["rgb_image"].dtype == torch.float32

    def test_six_channel_band_info_length(self) -> None:
        targets = (625.0, 550.0, 450.0, 1450.0, 1200.0, 1050.0)
        sel = FixedWavelengthSelector(
            target_wavelengths=targets, normalize_output=False, norm_mode="per_frame"
        )
        cube = _cube(C=20)
        wl = np.linspace(400.0, 1500.0, 20, dtype=np.float32)
        out = sel.forward(cube, wl)
        assert len(out["band_info"]["band_indices"]) == 6
        assert len(out["band_info"]["band_wavelengths_nm"]) == 6

    def test_six_channel_values_match_cube_bands(self) -> None:
        """Each output channel must be exactly the nearest cube band, not a mix."""
        wl = np.array([450.0, 550.0, 625.0, 1050.0, 1200.0, 1450.0], dtype=np.float32)
        targets = (625.0, 550.0, 450.0, 1450.0, 1200.0, 1050.0)
        sel = FixedWavelengthSelector(
            target_wavelengths=targets, normalize_output=False, norm_mode="per_frame"
        )
        # Cube: each channel c is filled with float(c) so we can identify which band landed where
        cube = torch.zeros(1, 4, 4, 6)
        for c in range(6):
            cube[..., c] = float(c)
        out = sel.forward(cube, wl)
        rgb = out["rgb_image"]  # [1, 4, 4, 6]
        # target order: 625→idx2, 550→idx1, 450→idx0, 1450→idx5, 1200→idx4, 1050→idx3
        expected_values = [2.0, 1.0, 0.0, 5.0, 4.0, 3.0]
        for ch, expected in enumerate(expected_values):
            assert float(rgb[0, 0, 0, ch]) == pytest.approx(expected), (
                f"channel {ch}: expected {expected}, got {float(rgb[0, 0, 0, ch])}"
            )

    def test_single_channel_output_shape(self) -> None:
        """Single target wavelength → [B, H, W, 1]."""
        sel = FixedWavelengthSelector(
            target_wavelengths=(550.0,), normalize_output=False, norm_mode="per_frame"
        )
        cube = _cube(C=10)
        wl = _wavelengths(10)
        out = sel.forward(cube, wl)
        assert out["rgb_image"].shape == (1, 4, 4, 1)


# ---------------------------------------------------------------------------
# normalize_output=True warning for n != 3
# ---------------------------------------------------------------------------


class TestNormalizeOutputWarning:
    def test_six_channel_normalize_true_warns_once_at_construction(self) -> None:
        """normalize_output=True with n=6 must warn ONCE at __init__ (not per
        forward call) and the forward must return raw stacked bands.

        Regression for review point 4: previously the warning was emitted from
        inside forward(), which in a video run is one identical warning per
        frame. The fix hoists it to __init__ where n and normalize_output are
        both known and don't change later.
        """
        from loguru import logger

        targets = (625.0, 550.0, 450.0, 1450.0, 1200.0, 1050.0)
        warnings_seen: list[str] = []
        sink_id = logger.add(lambda m: warnings_seen.append(m.record["message"]), level="WARNING")
        try:
            sel = FixedWavelengthSelector(
                target_wavelengths=targets, normalize_output=True, norm_mode="per_frame"
            )
            warns_after_init = len(warnings_seen)
            # Multiple forward calls must NOT add new warnings.
            cube = _cube(C=20) * 500.0
            wl = np.linspace(400.0, 1500.0, 20, dtype=np.float32)
            for _ in range(5):
                sel.forward(cube, wl)
            warns_after_forwards = len(warnings_seen)
        finally:
            logger.remove(sink_id)

        assert warns_after_init == 1, (
            f"Expected exactly 1 warning at __init__, got {warns_after_init}"
        )
        assert "normalize_output=True is only supported" in warnings_seen[0]
        assert warns_after_forwards == 1, (
            f"forward() must not emit further warnings; got {warns_after_forwards} total."
        )

    def test_six_channel_normalize_true_returns_raw(self) -> None:
        """Forward returns raw stacked bands (not normalised) for n != 3."""
        targets = (625.0, 550.0, 450.0, 1450.0, 1200.0, 1050.0)
        sel = FixedWavelengthSelector(
            target_wavelengths=targets, normalize_output=True, norm_mode="per_frame"
        )
        cube = _cube(C=20) * 500.0  # large values so normalisation would crush them
        wl = np.linspace(400.0, 1500.0, 20, dtype=np.float32)
        out = sel.forward(cube, wl)
        assert out["rgb_image"].shape == (1, 4, 4, 6)
        # Raw cube values are ~0-500, so at least some pixels should be > 1
        assert float(out["rgb_image"].max()) > 1.0, (
            "Expected raw (unnormalized) values > 1 for n=6 with normalize_output=True"
        )

    def test_four_channel_normalize_true_returns_raw(self) -> None:
        sel = FixedWavelengthSelector(
            target_wavelengths=(700.0, 600.0, 500.0, 400.0),
            normalize_output=True,
            norm_mode="per_frame",
        )
        cube = _cube(C=10) * 200.0
        wl = _wavelengths(10)
        out = sel.forward(cube, wl)
        assert out["rgb_image"].shape == (1, 4, 4, 4)

    def test_three_channel_normalize_false_does_not_warn(self) -> None:
        """normalize_output=False on the standard 3-channel default must NOT
        trigger the n != 3 warning."""
        from loguru import logger

        warnings_seen: list[str] = []
        sink_id = logger.add(lambda m: warnings_seen.append(m.record["message"]), level="WARNING")
        try:
            FixedWavelengthSelector(
                target_wavelengths=(650.0, 550.0, 450.0),
                normalize_output=False,
                norm_mode="per_frame",
            )
        finally:
            logger.remove(sink_id)
        relevant = [w for w in warnings_seen if "normalize_output" in w]
        assert relevant == [], (
            f"Unexpected warning(s) at __init__ for the 3-channel default: {relevant}"
        )


# ---------------------------------------------------------------------------
# norm_mode guard (review point 1) — running/statistical require n == 3
# ---------------------------------------------------------------------------


class TestNormModeGuard:
    """The running/statistical normalisation paths in ChannelSelectorBase rely
    on 3-element buffers and ``reshape(-1, 3)`` of the raw RGB tensor. For
    ``n != 3`` they would silently mix unrelated channels (e.g. n=6 reshapes
    the 6-channel pixels into pairs of c0+c3, c1+c4, c2+c5 per row) or raise a
    ``RuntimeError`` on non-divisible totals (n=4). Reject these modes at
    construction so the failure mode is loud and obvious."""

    @pytest.mark.parametrize("mode", ["statistical", "running"])
    def test_n6_with_unsupported_mode_raises(self, mode: str) -> None:
        targets = (625.0, 550.0, 450.0, 1450.0, 1200.0, 1050.0)
        with pytest.raises(ValueError, match="norm_mode.*requires exactly 3"):
            FixedWavelengthSelector(target_wavelengths=targets, norm_mode=mode)

    @pytest.mark.parametrize("mode", ["statistical", "running"])
    def test_n4_with_unsupported_mode_raises(self, mode: str) -> None:
        with pytest.raises(ValueError, match="norm_mode.*requires exactly 3"):
            FixedWavelengthSelector(target_wavelengths=(700.0, 600.0, 500.0, 400.0), norm_mode=mode)

    @pytest.mark.parametrize("mode", ["statistical", "running"])
    def test_n3_with_either_mode_is_allowed(self, mode: str) -> None:
        """3-channel selectors with running/statistical mode must still
        construct cleanly — that's the original supported configuration."""
        sel = FixedWavelengthSelector(target_wavelengths=(650.0, 550.0, 450.0), norm_mode=mode)
        assert sel.norm_mode.value == mode

    @pytest.mark.parametrize("n", [1, 4, 6, 9])
    def test_n_neq_3_with_per_frame_is_allowed(self, n: int) -> None:
        """per_frame is the only mode supported for n != 3 today."""
        targets = tuple(450.0 + 100.0 * i for i in range(n))
        sel = FixedWavelengthSelector(target_wavelengths=targets, norm_mode="per_frame")
        assert len(sel.target_wavelengths) == n


# ---------------------------------------------------------------------------
# band_info honesty (review point 2) + strategy label (review point 8)
# ---------------------------------------------------------------------------


class TestBandInfoHonesty:
    def test_band_info_normalized_output_false_when_n_neq_3(self) -> None:
        """band_info['normalized_output'] must reflect what actually happened.
        For n != 3 the forward returns raw bands, so the flag must be False
        even if the user passed normalize_output=True at construction.

        Regression for review point 2: previously the flag mirrored the input
        kwarg, so a consumer that trusted the flag would skip a normalisation
        it still needed."""
        targets = (625.0, 550.0, 450.0, 1450.0, 1200.0, 1050.0)
        sel = FixedWavelengthSelector(
            target_wavelengths=targets, normalize_output=True, norm_mode="per_frame"
        )
        cube = _cube(C=20)
        wl = np.linspace(400.0, 1500.0, 20, dtype=np.float32)
        out = sel.forward(cube, wl)
        assert out["band_info"]["normalized_output"] is False, (
            "n != 3 returns raw bands, so band_info['normalized_output'] must be False"
        )

    def test_band_info_normalized_output_true_when_n3_and_normalized(self) -> None:
        sel = FixedWavelengthSelector(
            target_wavelengths=(650.0, 550.0, 450.0),
            normalize_output=True,
            norm_mode="per_frame",
        )
        cube = _cube(C=10) * 500.0
        out = sel.forward(cube, _wavelengths(10))
        assert out["band_info"]["normalized_output"] is True

    def test_band_info_strategy_label_branches_on_n(self) -> None:
        """Review point 8: ``band_info['strategy']`` was hard-coded
        ``baseline_false_rgb`` even for 6-channel output. Now it branches:
        ``baseline_false_rgb`` for n == 3, ``stacked_bands`` otherwise."""
        sel3 = FixedWavelengthSelector(
            target_wavelengths=(650.0, 550.0, 450.0), normalize_output=False
        )
        sel6 = FixedWavelengthSelector(
            target_wavelengths=(625.0, 550.0, 450.0, 1450.0, 1200.0, 1050.0),
            normalize_output=False,
            norm_mode="per_frame",
        )
        wl = _wavelengths(20, lo=400.0, hi=1500.0)
        cube = _cube(C=20)
        assert sel3.forward(cube, wl)["band_info"]["strategy"] == "baseline_false_rgb"
        assert sel6.forward(cube, wl)["band_info"]["strategy"] == "stacked_bands"


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_empty_wavelengths_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one wavelength"):
            FixedWavelengthSelector(target_wavelengths=())

    def test_string_wavelengths_rejected_by_tuple_coercion(self) -> None:
        """Non-numeric strings should raise during float coercion."""
        with pytest.raises((ValueError, TypeError)):
            FixedWavelengthSelector(target_wavelengths=("red", "green", "blue"))

    def test_wavelengths_coerced_to_float(self) -> None:
        """Integer wavelengths should be accepted and stored as float."""
        sel = FixedWavelengthSelector(target_wavelengths=(650, 550, 450))
        assert all(isinstance(w, float) for w in sel.target_wavelengths)


# ---------------------------------------------------------------------------
# requires_initial_fit per norm mode
# ---------------------------------------------------------------------------


class TestRequiresInitialFit:
    def test_statistical_mode_requires_fit(self) -> None:
        sel = FixedWavelengthSelector(norm_mode=NormMode.STATISTICAL)
        assert sel.requires_initial_fit is True

    @pytest.mark.parametrize("mode", [NormMode.RUNNING, NormMode.PER_FRAME])
    def test_non_statistical_modes_do_not_require_fit(self, mode: NormMode) -> None:
        """RUNNING/PER_FRAME need no StatisticalTrainer pass; the base sets the
        override so core's auto-detect (which sees statistical_initialization
        implemented) does not force one."""
        sel = FixedWavelengthSelector(norm_mode=mode)
        assert sel.requires_initial_fit is False

    def test_subclass_with_own_initialization_keeps_autodetect(self) -> None:
        """A subclass carrying genuine init logic must still report True in
        RUNNING mode — the base only claims 'no fit needed' for its own
        statistical_initialization."""

        class _CustomInitSelector(FixedWavelengthSelector):
            def statistical_initialization(self, input_stream) -> None:
                return None

        sel = _CustomInitSelector(norm_mode=NormMode.RUNNING)
        assert sel.requires_initial_fit is True
