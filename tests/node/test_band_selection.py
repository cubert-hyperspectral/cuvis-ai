"""Tests for supervised channel selection refactoring.

Covers the template method hooks in SupervisedSelectorBase and
the concrete subclass implementations.
"""

import numpy as np
import pytest
import torch

from cuvis_ai.node.channel_selector import (
    SupervisedCIRSelector,
    SupervisedFullSpectrumSelector,
    SupervisedSelectorBase,
    SupervisedWindowedSelector,
)

# ---------------------------------------------------------------------------
# Base class hook / abstract method tests
# ---------------------------------------------------------------------------


class TestSupervisedSelectorBase:
    def test_strategy_name_is_empty_on_base(self):
        assert SupervisedSelectorBase._strategy_name == ""

    def test_select_bands_raises_not_implemented(self):
        sel = SupervisedFullSpectrumSelector(num_spectral_bands=10)
        with pytest.raises(NotImplementedError):
            SupervisedSelectorBase._select_bands(
                sel,
                np.zeros(10),
                np.linspace(400, 900, 10),
                np.eye(10),
            )

    def test_extra_band_info_returns_empty_dict(self):
        # FullSpectrum does NOT override _extra_band_info
        sel = SupervisedFullSpectrumSelector(num_spectral_bands=10)
        result = sel._extra_band_info(np.linspace(400, 900, 10))
        assert result == {}

    def test_forward_raises_when_not_fitted(self):
        sel = SupervisedFullSpectrumSelector(num_spectral_bands=20)
        cube = torch.randn(1, 8, 8, 20)
        wavelengths = np.linspace(430, 910, 20)

        with pytest.raises(RuntimeError, match="not fitted"):
            sel.forward(cube, wavelengths)


# ---------------------------------------------------------------------------
# Strategy name / _extra_band_info for concrete subclasses
# ---------------------------------------------------------------------------


class TestStrategyNames:
    def test_cir_strategy_name(self):
        assert SupervisedCIRSelector._strategy_name == "supervised_cir"

    def test_windowed_strategy_name(self):
        assert SupervisedWindowedSelector._strategy_name == "supervised_windowed_false_rgb"

    def test_full_spectrum_strategy_name(self):
        assert SupervisedFullSpectrumSelector._strategy_name == "supervised_full_spectrum"


class TestExtraBandInfo:
    def test_cir_returns_windows(self):
        sel = SupervisedCIRSelector(num_spectral_bands=20)
        info = sel._extra_band_info(np.linspace(430, 910, 20, dtype=np.float32))
        assert "windows_nm" in info
        assert len(info["windows_nm"]) == 3

    def test_windowed_returns_windows(self):
        sel = SupervisedWindowedSelector(num_spectral_bands=20)
        info = sel._extra_band_info(np.linspace(430, 910, 20, dtype=np.float32))
        assert "windows_nm" in info
        assert len(info["windows_nm"]) == 3

    def test_full_spectrum_returns_empty(self):
        sel = SupervisedFullSpectrumSelector(num_spectral_bands=20)
        info = sel._extra_band_info(np.linspace(430, 910, 20, dtype=np.float32))
        assert info == {}


# ---------------------------------------------------------------------------
# Forward pass after manual initialization
# ---------------------------------------------------------------------------


class TestForwardAfterManualInit:
    def test_forward_produces_rgb_and_band_info(self):
        num_bands = 20
        sel = SupervisedFullSpectrumSelector(num_spectral_bands=num_bands)

        # Manually set fitted state
        sel._statistically_initialized = True
        sel.selected_indices = torch.tensor([0, 5, 10], dtype=torch.long)

        cube = torch.randn(2, 8, 8, num_bands)
        wavelengths = np.linspace(430, 910, num_bands, dtype=np.float32)

        result = sel.forward(cube, wavelengths)

        assert "rgb_image" in result
        assert "band_info" in result
        assert result["rgb_image"].shape == (2, 8, 8, 3)
        assert result["band_info"]["strategy"] == "supervised_full_spectrum"
        assert result["band_info"]["band_indices"] == [0, 5, 10]
        assert "score_weights" in result["band_info"]
        assert "lambda_penalty" in result["band_info"]

    def test_forward_cir_includes_windows(self):
        num_bands = 20
        sel = SupervisedCIRSelector(num_spectral_bands=num_bands)

        sel._statistically_initialized = True
        sel.selected_indices = torch.tensor([2, 8, 15], dtype=torch.long)

        cube = torch.randn(1, 4, 4, num_bands)
        wavelengths = np.linspace(430, 910, num_bands, dtype=np.float32)

        result = sel.forward(cube, wavelengths)

        assert result["band_info"]["strategy"] == "supervised_cir"
        assert "windows_nm" in result["band_info"]


# ---------------------------------------------------------------------------
# Full-flow integration: statistical_initialization → forward
# ---------------------------------------------------------------------------


class TestStatisticalInitialization:
    def test_full_spectrum_fit_and_forward(self, synthetic_anomaly_datamodule):
        channels = 20
        dm = synthetic_anomaly_datamodule(
            batch_size=4,
            num_samples=16,
            height=8,
            width=8,
            channels=channels,
            include_labels=True,
            mode="random",
            seed=42,
            dtype=torch.float32,
        )

        sel = SupervisedFullSpectrumSelector(num_spectral_bands=channels)

        input_stream = dm.train_dataloader()
        sel.statistical_initialization(input_stream)

        assert sel._statistically_initialized is True

        batch = next(iter(dm.val_dataloader()))
        result = sel.forward(
            cube=batch["cube"],
            wavelengths=batch["wavelengths"][0].numpy().astype(np.float32),
        )

        assert result["rgb_image"].shape[0] == batch["cube"].shape[0]
        assert result["rgb_image"].shape[-1] == 3
        assert len(result["band_info"]["band_indices"]) == 3
        assert result["band_info"]["strategy"] == "supervised_full_spectrum"

    def test_cir_fit_and_forward(self, synthetic_anomaly_datamodule):
        channels = 61  # More channels needed for CIR windows
        dm = synthetic_anomaly_datamodule(
            batch_size=4,
            num_samples=16,
            height=8,
            width=8,
            channels=channels,
            include_labels=True,
            mode="random",
            seed=42,
            dtype=torch.float32,
            wavelength_range=(430.0, 910.0),
        )

        sel = SupervisedCIRSelector(num_spectral_bands=channels)

        input_stream = dm.train_dataloader()
        sel.statistical_initialization(input_stream)

        assert sel._statistically_initialized is True

        batch = next(iter(dm.val_dataloader()))
        result = sel.forward(
            cube=batch["cube"],
            wavelengths=batch["wavelengths"][0].numpy().astype(np.float32),
        )

        assert result["band_info"]["strategy"] == "supervised_cir"
        assert "windows_nm" in result["band_info"]
        assert result["rgb_image"].shape[-1] == 3

    def test_windowed_fit_and_forward(self, synthetic_anomaly_datamodule):
        channels = 61  # Enough channels to span visible RGB windows
        dm = synthetic_anomaly_datamodule(
            batch_size=4,
            num_samples=16,
            height=8,
            width=8,
            channels=channels,
            include_labels=True,
            mode="random",
            seed=42,
            dtype=torch.float32,
            wavelength_range=(430.0, 910.0),
        )

        sel = SupervisedWindowedSelector(num_spectral_bands=channels)

        input_stream = dm.train_dataloader()
        sel.statistical_initialization(input_stream)

        assert sel._statistically_initialized is True

        batch = next(iter(dm.val_dataloader()))
        result = sel.forward(
            cube=batch["cube"],
            wavelengths=batch["wavelengths"][0].numpy().astype(np.float32),
        )

        assert result["band_info"]["strategy"] == "supervised_windowed_false_rgb"
        assert "windows_nm" in result["band_info"]
        assert result["rgb_image"].shape[-1] == 3
