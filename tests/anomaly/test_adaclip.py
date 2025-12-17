"""Tests for AdaCLIP anomaly detection module.

This module tests:
- Weight download manager
- Band selection nodes
- AdaCLIP detector node
- Integration with cuvis.ai pipeline

Note: Some tests require GPU and internet access for downloading weights.
These are marked with appropriate pytest markers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from cuvis_ai.anomaly.adaclip import (
    ADACLIP_WEIGHTS,
    download_weights,
    get_weights_dir,
    list_available_weights,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _compute_pixel_metrics(gt_mask: np.ndarray | None, anomaly_map: np.ndarray) -> dict[str, float]:
    """Compute pixel-level metrics matching the legacy AdaCLIP pipeline.

    The F1 score is computed as the maximum F1 over all thresholds (optimal threshold),
    matching the original AdaCLIP tools/metrics.py implementation.
    """
    metrics = {
        "auroc": float("nan"),
        "f1": float("nan"),
        "ap": float("nan"),
        "auprc": float("nan"),
    }
    if gt_mask is None:
        return metrics

    gt_flat = gt_mask.reshape(-1).astype(np.uint8)
    pred_flat = anomaly_map.reshape(-1).astype(np.float32)

    # Check for single-class case (all zeros or all ones)
    if gt_flat.sum() == 0 or gt_flat.sum() == gt_flat.shape[0]:
        return metrics

    try:
        metrics["auroc"] = float(roc_auc_score(gt_flat, pred_flat))
    except ValueError:
        pass

    try:
        metrics["ap"] = float(average_precision_score(gt_flat, pred_flat))
    except ValueError:
        pass

    try:
        precision, recall, _ = precision_recall_curve(gt_flat, pred_flat)
        metrics["auprc"] = float(auc(recall, precision))
    except ValueError:
        pass

    # F1: Use maximum F1 over all thresholds (optimal threshold), matching legacy code
    try:
        precision, recall, _ = precision_recall_curve(gt_flat, pred_flat)
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)
        metrics["f1"] = float(np.max(f1_scores[np.isfinite(f1_scores)]))
    except ValueError:
        pass

    return metrics


# ============================================================================
# Weight Manager Tests
# ============================================================================


class TestWeightManager:
    """Tests for weight download manager."""

    def test_list_available_weights(self):
        """Test that available weights are listed correctly."""
        weights = list_available_weights()
        assert isinstance(weights, list)
        assert len(weights) >= 3
        assert "pretrained_all" in weights
        assert "pretrained_mvtec_colondb" in weights
        assert "pretrained_visa_clinicdb" in weights

    def test_get_weights_dir_creates_directory(self, tmp_path, monkeypatch):
        """Test that weights directory is created if it doesn't exist."""
        # Mock the home directory
        mock_home = tmp_path / "home"
        monkeypatch.setattr(Path, "home", lambda: mock_home)

        weights_dir = get_weights_dir()
        assert weights_dir.exists()
        assert "cuvis_ai" in str(weights_dir)
        assert "adaclip" in str(weights_dir)

    def test_adaclip_weights_registry(self):
        """Test that weight registry has correct structure."""
        for _name, cfg in ADACLIP_WEIGHTS.items():
            assert "gdrive_id" in cfg
            assert "description" in cfg
            assert "filename" in cfg
            assert cfg["gdrive_id"]  # Non-empty
            assert cfg["filename"].endswith(".pth")

    def test_download_weights_invalid_name(self):
        """Test that invalid weight name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown weight"):
            download_weights("invalid_weight_name")


# ============================================================================
# Band Selection Tests
# ============================================================================


class TestBandSelection:
    """Tests for band selection nodes."""

    @pytest.fixture
    def sample_cube(self):
        """Create a sample hyperspectral cube for testing."""
        # [B, H, W, C] format
        return torch.rand(2, 64, 64, 100, dtype=torch.float32)

    @pytest.fixture
    def sample_wavelengths(self):
        """Create sample wavelengths from 400-900nm."""
        return torch.linspace(400, 900, 100, dtype=torch.float32)

    def test_baseline_false_rgb_selector(self, sample_cube, sample_wavelengths):
        """Test BaselineFalseRGBSelector node."""
        from cuvis_ai.node import BaselineFalseRGBSelector

        selector = BaselineFalseRGBSelector(target_wavelengths=(650.0, 550.0, 450.0))

        result = selector.forward(cube=sample_cube, wavelengths=sample_wavelengths)

        assert "rgb_image" in result
        assert "band_info" in result
        assert result["rgb_image"].shape == (2, 64, 64, 3)
        assert result["rgb_image"].dtype == torch.float32
        assert result["band_info"]["strategy"] == "baseline_false_rgb"
        assert len(result["band_info"]["band_indices"]) == 3

    def test_high_contrast_band_selector(self, sample_cube, sample_wavelengths):
        """Test HighContrastBandSelector node."""
        from cuvis_ai.node import HighContrastBandSelector

        windows = ((610, 700), (500, 580), (440, 500))
        selector = HighContrastBandSelector(windows=windows, alpha=0.1)

        result = selector.forward(cube=sample_cube, wavelengths=sample_wavelengths)

        assert "rgb_image" in result
        assert "band_info" in result
        assert result["rgb_image"].shape == (2, 64, 64, 3)
        assert result["band_info"]["strategy"] == "high_contrast"
        assert len(result["band_info"]["band_indices"]) == 3

    def test_cir_false_color_selector(self, sample_cube, sample_wavelengths):
        """Test CIRFalseColorSelector node."""
        from cuvis_ai.node import CIRFalseColorSelector

        selector = CIRFalseColorSelector()

        result = selector.forward(cube=sample_cube, wavelengths=sample_wavelengths)

        assert "rgb_image" in result
        assert "band_info" in result
        assert result["rgb_image"].shape == (2, 64, 64, 3)
        assert result["band_info"]["strategy"] == "cir_false_color"
        assert result["band_info"]["channel_mapping"]["R"] == "NIR"
        # Defaults should match legacy wavelengths (860, 670, 560)
        assert selector.nir_nm == 860.0
        assert selector.red_nm == 670.0
        assert selector.green_nm == 560.0

    def test_band_selector_normalize_output(self, sample_cube, sample_wavelengths):
        """Test that band selector outputs are normalized to 0-1 range."""
        from cuvis_ai.node import BaselineFalseRGBSelector

        selector = BaselineFalseRGBSelector()
        result = selector.forward(cube=sample_cube, wavelengths=sample_wavelengths)

        rgb = result["rgb_image"]
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_supervised_cir_selector_fit_and_forward(self):
        """Test SupervisedCIRBandSelector fit() and forward()."""
        from cuvis_ai.node import SupervisedCIRBandSelector

        # Create synthetic training data
        wavelengths = torch.linspace(400.0, 900.0, 9, dtype=torch.float32)  # 9 bands
        # Two batches, 4x4 spatial, 9 channels
        cube_train = torch.rand(2, 4, 4, 9, dtype=torch.float32)
        # Binary mask with both classes present
        mask_train = torch.zeros(2, 4, 4, 1, dtype=torch.bool)
        mask_train[:, :2, :2, :] = True  # mark some pixels as positive

        def stream():
            yield {"cube": cube_train, "mask": mask_train, "wavelengths": wavelengths}

        selector = SupervisedCIRBandSelector(
            windows=((840.0, 900.0), (650.0, 720.0), (500.0, 570.0)),
            score_weights=(1.0, 1.0, 1.0),
            lambda_penalty=0.5,
        )

        # Before fit, forward should fail
        with pytest.raises(RuntimeError):
            selector.forward(cube=cube_train, wavelengths=wavelengths)

        # Fit with synthetic stream
        selector.statistical_initialization(stream())

        # After fit, forward should work
        result = selector.forward(cube=cube_train, wavelengths=wavelengths)
        assert "rgb_image" in result
        assert "band_info" in result
        assert result["rgb_image"].shape == (2, 4, 4, 3)
        band_indices = result["band_info"]["band_indices"]
        assert len(band_indices) == 3

    def test_supervised_windowed_false_rgb_windows_respected(self):
        """Test that SupervisedWindowedFalseRGBSelector picks one band per window."""
        from cuvis_ai.node import SupervisedWindowedFalseRGBSelector

        # Define 9 bands, 3 in each window
        wavelengths = torch.tensor(
            [450.0, 460.0, 470.0, 520.0, 530.0, 540.0, 620.0, 630.0, 640.0],
            dtype=torch.float32,
        )
        cube_train = torch.rand(1, 4, 4, 9, dtype=torch.float32)
        mask_train = torch.zeros(1, 4, 4, 1, dtype=torch.bool)
        mask_train[:, :2, :2, :] = True

        def stream():
            yield {"cube": cube_train, "mask": mask_train, "wavelengths": wavelengths}

        windows = ((440.0, 500.0), (500.0, 580.0), (610.0, 700.0))
        selector = SupervisedWindowedFalseRGBSelector(
            windows=windows,
            score_weights=(1.0, 1.0, 1.0),
            lambda_penalty=0.1,
        )
        selector.statistical_initialization(stream())
        result = selector.forward(cube=cube_train, wavelengths=wavelengths)

        indices = result["band_info"]["band_indices"]
        assert len(indices) == 3
        # Check that each selected index lies in exactly one window
        wavelengths_np = wavelengths.numpy()
        for idx, (start, end) in zip(indices, windows, strict=True):
            wl = wavelengths_np[idx]
            assert start <= wl <= end

    def test_supervised_full_spectrum_selector_selects_three_bands(self):
        """Test SupervisedFullSpectrumBandSelector selects 3 bands globally."""
        from cuvis_ai.node import SupervisedFullSpectrumBandSelector

        wavelengths = torch.linspace(400.0, 900.0, 12, dtype=torch.float32)
        cube_train = torch.rand(1, 4, 4, 12, dtype=torch.float32)
        mask_train = torch.zeros(1, 4, 4, 1, dtype=torch.bool)
        mask_train[:, :2, :2, :] = True

        def stream():
            yield {"cube": cube_train, "mask": mask_train, "wavelengths": wavelengths}

        selector = SupervisedFullSpectrumBandSelector(
            score_weights=(1.0, 1.0, 1.0),
            lambda_penalty=0.2,
        )
        selector.statistical_initialization(stream())
        result = selector.forward(cube=cube_train, wavelengths=wavelengths)

        indices = result["band_info"]["band_indices"]
        assert len(indices) == 3
        # Indices should be unique
        assert len(set(indices)) == 3


# ============================================================================
# AdaCLIP Detector Tests
# ============================================================================


class TestAdaCLIPDetector:
    """Tests for AdaCLIPDetector node."""

    def test_detector_initialization(self):
        """Test that AdaCLIPDetector initializes with correct parameters."""
        from cuvis_ai.node import AdaCLIPDetector

        detector = AdaCLIPDetector(
            weight_name="pretrained_all",
            prompt_text="test prompt",
            gaussian_sigma=4.0,
        )

        assert detector.weight_name == "pretrained_all"
        assert detector.prompt_text == "test prompt"
        assert detector.gaussian_sigma == 4.0
        assert detector._model is None  # Lazy loading (via property)
        assert detector._adaclip_model is None  # Direct access
        # Verify _statistically_initialized attribute is False
        assert hasattr(detector, "_statistically_initialized")
        assert detector._statistically_initialized is False

    def test_detector_port_specs(self):
        """Test that AdaCLIPDetector has correct port specs."""
        from cuvis_ai.node import AdaCLIPDetector

        detector = AdaCLIPDetector()

        # Check input specs
        assert "rgb_image" in detector.INPUT_SPECS
        input_spec = detector.INPUT_SPECS["rgb_image"]
        assert input_spec.shape == (-1, -1, -1, 3)

        # Check output specs
        assert "scores" in detector.OUTPUT_SPECS
        assert "anomaly_score" in detector.OUTPUT_SPECS
        output_spec = detector.OUTPUT_SPECS["scores"]
        assert output_spec.shape == (-1, -1, -1, 1)

    @pytest.mark.skip(reason="Requires downloading weights (~500MB)")
    def test_detector_forward_with_real_weights(self):
        """Test forward pass with real weights (requires download)."""
        from cuvis_ai.node import AdaCLIPDetector

        detector = AdaCLIPDetector(weight_name="pretrained_all")

        # Create dummy RGB input
        rgb_input = torch.rand(1, 64, 64, 3, dtype=torch.float32)

        result = detector.forward(rgb_image=rgb_input)

        assert "scores" in result
        assert "anomaly_score" in result
        assert result["scores"].shape == (1, 64, 64, 1)
        assert result["anomaly_score"].shape == (1,)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for AdaCLIP with cuvis.ai pipeline."""

    def test_pipeline_connection_types(self):
        """Test that nodes can be connected in a pipeline."""
        from cuvis_ai.node import AdaCLIPDetector, BaselineFalseRGBSelector
        from cuvis_ai.pipeline.pipeline import CuvisPipeline

        _pipeline = CuvisPipeline("test_adaclip")

        selector = BaselineFalseRGBSelector()
        detector = AdaCLIPDetector()

        # Test that port types are compatible
        selector_output = selector.OUTPUT_SPECS["rgb_image"]
        detector_input = detector.INPUT_SPECS["rgb_image"]

        # Both should be float32 with 3 channels
        assert selector_output.dtype == detector_input.dtype

    def test_band_selector_to_detector_pipeline(self):
        """Test a complete band selector -> detector pipeline setup."""
        from cuvis_ai.node import AdaCLIPDetector, BaselineFalseRGBSelector
        from cuvis_ai.pipeline.pipeline import CuvisPipeline

        pipeline = CuvisPipeline("test_band_to_adaclip")

        selector = BaselineFalseRGBSelector()
        detector = AdaCLIPDetector()

        # This should not raise any errors
        pipeline.connect(
            (selector.outputs.rgb_image, detector.inputs.rgb_image),
        )

        # Check that nodes are in the pipeline
        assert len(list(pipeline._graph.nodes())) == 2

    def test_adaclip_model_wrapper(self):
        """Test AdaCLIPModel wrapper class."""
        from cuvis_ai.anomaly.adaclip import AdaCLIPModel

        # Test initialization (without loading weights)
        model = AdaCLIPModel(
            backbone="ViT-L-14-336",
            image_size=518,
            prompting_depth=4,
            prompting_length=5,
        )

        assert model.backbone == "ViT-L-14-336"
        assert model.image_size == 518
        assert model._model is None  # Lazy initialization


# ============================================================================
# Core Model Tests
# ============================================================================


class TestCoreModels:
    """Tests for core AdaCLIP model components."""

    def test_transformer_layer_norm(self):
        """Test LayerNorm variants work correctly."""
        from cuvis_ai.anomaly.adaclip.core.transformer import LayerNorm, LayerNormFp32

        dim = 768
        ln = LayerNorm(dim)
        ln_fp32 = LayerNormFp32(dim)

        x = torch.randn(2, 10, dim)
        x_fp16 = x.half()

        # Test standard LayerNorm
        out = ln(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

        # Test FP32 LayerNorm with FP16 input
        out_fp32 = ln_fp32(x_fp16)
        assert out_fp32.shape == x_fp16.shape
        assert out_fp32.dtype == x_fp16.dtype

    def test_quick_gelu(self):
        """Test QuickGELU activation."""
        from cuvis_ai.anomaly.adaclip.core.transformer import QuickGELU

        gelu = QuickGELU()
        x = torch.randn(2, 10, 768)
        out = gelu(x)

        assert out.shape == x.shape
        # QuickGELU should be bounded
        assert torch.isfinite(out).all()

    def test_simple_tokenizer(self):
        """Test SimpleTokenizer for text encoding."""
        from cuvis_ai.anomaly.adaclip.core.simple_tokenizer import SimpleTokenizer

        tokenizer = SimpleTokenizer()

        # Test encoding
        tokens = tokenizer.encode("hello world")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

        # Test decoding
        text = tokenizer.decode(tokens)
        assert isinstance(text, str)
        assert "hello" in text.lower()

    def test_utils_to_2tuple(self):
        """Test to_2tuple utility function."""
        from cuvis_ai.anomaly.adaclip.core.utils import to_2tuple

        # Single int -> tuple
        result = to_2tuple(224)
        assert result == (224, 224)

        # Already a tuple -> unchanged
        result = to_2tuple((224, 336))
        assert result == (224, 336)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_band_selector_empty_window(self):
        """Test band selector handles window with no bands."""
        from cuvis_ai.node import HighContrastBandSelector

        # Create wavelengths that don't overlap with one window
        wavelengths = torch.linspace(500, 700, 50)
        cube = torch.rand(1, 32, 32, 50)

        # Window that has no bands
        selector = HighContrastBandSelector(windows=((300, 350), (550, 600), (650, 700)))

        result = selector.forward(cube=cube, wavelengths=wavelengths)

        # Should fallback to nearest band
        assert "rgb_image" in result
        assert result["rgb_image"].shape == (1, 32, 32, 3)

    def test_detector_wrong_channel_count(self):
        """Test that detector with wrong channel count is caught by PortSpec.

        Note: Shape validation is handled by PortSpec at connection time,
        not by explicit asserts in forward(). This test verifies the INPUT_SPECS
        are correctly defined.
        """
        from cuvis_ai.node import AdaCLIPDetector

        detector = AdaCLIPDetector()

        # Verify INPUT_SPECS expects 3 channels
        input_spec = detector.INPUT_SPECS["rgb_image"]
        assert input_spec.shape[-1] == 3, "INPUT_SPECS should require 3 channels"


# ============================================================================
# Regression Tests for Legacy Parity
# ============================================================================


class DummyResize:
    """Simple stand-in for torchvision.transforms.Resize."""

    def __init__(self, size):
        self.size = size
        self.interpolation = None


class DummyCenterCrop:
    """Simple stand-in for torchvision.transforms.CenterCrop."""

    def __init__(self, size):
        self.size = size


class DummyPreprocess:
    """Container mimicking torchvision Compose."""

    def __init__(self, transforms):
        self.transforms = transforms


class DummyAdaCLIP:
    """Lightweight AdaCLIP stub for unit tests."""

    def __init__(self, *args, **kwargs):
        self.device = kwargs.get("device", "cpu")

    def to(self, device):
        self.device = device
        return self


class TestLegacyParity:
    """Regression tests ensuring new pipeline matches legacy behaviors."""

    def test_adaclip_model_preprocess_uses_tuple_resize(self, monkeypatch):
        """Ensure the preprocessing resize matches legacy (exact square)."""
        from cuvis_ai.anomaly.adaclip import AdaCLIPModel

        dummy_clip_model = MagicMock()
        preprocess_val = DummyPreprocess(
            transforms=[
                DummyResize(518),  # Will be replaced with tuple
                DummyCenterCrop(518),
                object(),
            ]
        )

        def fake_create_model_and_transforms(**_):
            return dummy_clip_model, None, preprocess_val

        monkeypatch.setattr(
            "cuvis_ai.anomaly.adaclip.model.create_model_and_transforms",
            fake_create_model_and_transforms,
        )
        monkeypatch.setattr(
            "cuvis_ai.anomaly.adaclip.core.custom_clip.get_model_config",
            lambda name: {"embed_dim": 768, "vision_cfg": {"width": 1024}},
        )
        monkeypatch.setattr(
            "cuvis_ai.anomaly.adaclip.model.AdaCLIP",
            DummyAdaCLIP,
        )

        model = AdaCLIPModel(backbone="ViT-L-14-336", image_size=518)
        _ = model.get_preprocess()  # triggers _init_model

        resize = preprocess_val.transforms[0]
        crop = preprocess_val.transforms[1]
        assert resize.size == (518, 518)
        assert crop.size == (518, 518)

    def test_compute_pixel_metrics_uses_optimal_f1(self):
        """Verify pixel metrics compute maximal F1 rather than fixed threshold."""
        gt_mask = np.array(
            [
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        # Scores that require threshold < 0.5 for best F1
        pred = np.array(
            [
                [0.8, 0.7, 0.6, 0.2],
                [0.75, 0.4, 0.3, 0.2],
                [0.1, 0.1, 0.1, 0.1],
                [0.05, 0.05, 0.05, 0.05],
            ],
            dtype=np.float32,
        )

        metrics = _compute_pixel_metrics(gt_mask, pred)

        # Manually compute best F1 across unique scores
        thresholds = sorted({0.0, *pred.flatten()}, reverse=True)
        best_f1 = 0.0
        for t in thresholds:
            preds_bin = (pred >= t).astype(np.uint8)
            tp = (preds_bin * gt_mask).sum()
            fp = preds_bin.sum() - tp
            fn = gt_mask.sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
            best_f1 = max(best_f1, f1)

        assert np.isclose(metrics["f1"], best_f1, atol=1e-6)

    def test_compute_pixel_metrics_handles_none_mask(self):
        """Test that _compute_pixel_metrics handles None mask gracefully."""
        pred = np.array([[0.5, 0.6], [0.4, 0.3]], dtype=np.float32)
        metrics = _compute_pixel_metrics(None, pred)
        assert all(np.isnan(v) for v in metrics.values())

    def test_compute_pixel_metrics_handles_single_class(self):
        """Test that _compute_pixel_metrics handles single-class cases."""
        # All zeros
        gt_mask = np.zeros((4, 4), dtype=np.uint8)
        pred = np.random.rand(4, 4).astype(np.float32)
        metrics = _compute_pixel_metrics(gt_mask, pred)
        assert all(np.isnan(v) for v in metrics.values())

        # All ones
        gt_mask = np.ones((4, 4), dtype=np.uint8)
        metrics = _compute_pixel_metrics(gt_mask, pred)
        assert all(np.isnan(v) for v in metrics.values())


# ============================================================================
# Device Handling Tests
# ============================================================================


class TestDeviceHandling:
    """Tests for device consistency in AdaCLIP nodes."""

    def test_adaclip_detector_outputs_on_model_device(self):
        """Test that AdaCLIPDetector returns outputs on the model's device.

        Note: Per the migration guidelines, nodes should NOT move outputs to
        match input device. Device placement is handled by pipeline.to().
        """
        from cuvis_ai.node import AdaCLIPDetector

        detector = AdaCLIPDetector()
        # Mock the model to avoid downloading weights
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.predict.return_value = (
            torch.rand(1, 32, 32, device="cpu"),
            torch.tensor([0.5], device="cpu"),
        )
        # Use _adaclip_model directly since _model is now a read-only property
        detector._adaclip_model = mock_model
        detector._preprocess = MagicMock(return_value=torch.rand(1, 3, 518, 518))

        # Input on CPU
        rgb_input = torch.rand(1, 64, 64, 3, dtype=torch.float32)
        result = detector.forward(rgb_image=rgb_input)

        # Outputs should be on the model's device (pipeline.to() handles placement)
        assert result["scores"].device.type == "cpu"
        assert result["anomaly_score"].device.type == "cpu"

    def test_band_selector_preserves_device(self):
        """Test that band selectors preserve input device."""
        from cuvis_ai.node import BaselineFalseRGBSelector

        selector = BaselineFalseRGBSelector()
        cube = torch.rand(1, 32, 32, 50, dtype=torch.float32)
        wavelengths = torch.linspace(400, 900, 50, dtype=torch.float32)

        result = selector.forward(cube=cube, wavelengths=wavelengths)
        assert result["rgb_image"].device == cube.device


# ============================================================================
# Pipeline Integration Tests
# ============================================================================


class TestPipelineIntegration:
    """Tests for full pipeline integration with AdaCLIP pipeline."""

    def test_pipeline_with_baseline_strategy(self):
        """Test pipeline setup with baseline band selector and AdaCLIP."""
        from cuvis_ai.node import AdaCLIPDetector, BaselineFalseRGBSelector
        from cuvis_ai.node.data import LentilsAnomalyDataNode
        from cuvis_ai.pipeline.pipeline import CuvisPipeline

        pipeline = CuvisPipeline("test_baseline_adaclip")
        data_node = LentilsAnomalyDataNode(
            normal_class_ids=[0, 1],
            wavelengths=np.linspace(400, 900, 50),
        )
        band_selector = BaselineFalseRGBSelector()
        detector = AdaCLIPDetector()

        pipeline.connect(
            (data_node.outputs.cube, band_selector.inputs.cube),
            (data_node.outputs.wavelengths, band_selector.inputs.wavelengths),
            (band_selector.outputs.rgb_image, detector.inputs.rgb_image),
        )

        assert len(list(pipeline._graph.nodes())) == 3

    def test_pipeline_with_supervised_strategy(self):
        """Test pipeline setup with supervised band selector."""
        from cuvis_ai.node import AdaCLIPDetector, SupervisedCIRBandSelector
        from cuvis_ai.node.data import LentilsAnomalyDataNode
        from cuvis_ai.pipeline.pipeline import CuvisPipeline

        pipeline = CuvisPipeline("test_supervised_adaclip")
        data_node = LentilsAnomalyDataNode(
            normal_class_ids=[0, 1],
            wavelengths=np.linspace(400, 900, 50),
        )
        band_selector = SupervisedCIRBandSelector()
        detector = AdaCLIPDetector()

        pipeline.connect(
            (data_node.outputs.cube, band_selector.inputs.cube),
            (data_node.outputs.wavelengths, band_selector.inputs.wavelengths),
            (data_node.outputs.mask, band_selector.inputs.mask),
            (band_selector.outputs.rgb_image, detector.inputs.rgb_image),
        )

        assert len(list(pipeline._graph.nodes())) == 3
        assert "mask" in band_selector.INPUT_SPECS

    def test_pipeline_handles_missing_mask_for_unsupervised(self):
        """Test that pipeline works when mask is not provided to unsupervised selectors."""
        from cuvis_ai.node import BaselineFalseRGBSelector
        from cuvis_ai.node.data import LentilsAnomalyDataNode
        from cuvis_ai.pipeline.pipeline import CuvisPipeline

        pipeline = CuvisPipeline("test_no_mask")
        data_node = LentilsAnomalyDataNode(
            normal_class_ids=[0, 1],
            wavelengths=np.linspace(400, 900, 50),
        )
        band_selector = BaselineFalseRGBSelector()

        # Should not require mask connection
        pipeline.connect(
            (data_node.outputs.cube, band_selector.inputs.cube),
            (data_node.outputs.wavelengths, band_selector.inputs.wavelengths),
        )

        assert len(list(pipeline._graph.nodes())) == 2


# ============================================================================
# Statistical Scripts Tests
# ============================================================================


class TestStatisticalScripts:
    """Smoke tests for statistical_*.py example scripts."""

    def test_statistical_baseline_imports(self):
        """Test that statistical_baseline.py can be imported."""
        import sys
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Just test that the module can be imported
        from examples.adaclip import statistical_baseline  # noqa: F401

    def test_statistical_cir_false_color_imports(self):
        """Test that statistical_cir_false_color.py can be imported."""
        import sys
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from examples.adaclip import statistical_cir_false_color  # noqa: F401

    def test_statistical_supervised_cir_imports(self):
        """Test that statistical_supervised_cir.py can be imported."""
        import sys
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from examples.adaclip import statistical_supervised_cir  # noqa: F401


# ============================================================================
# Additional Edge Cases
# ============================================================================


class TestAdditionalEdgeCases:
    """Additional edge case tests."""

    def test_high_contrast_default_windows_order(self):
        """Test that high contrast uses correct legacy window order (B, G, R)."""
        from cuvis_ai.node import HighContrastBandSelector

        selector = HighContrastBandSelector()
        windows = selector.windows
        # Should be Blue (440-500), Green (500-580), Red (610-700)
        assert windows[0] == (440.0, 500.0)
        assert windows[1] == (500.0, 580.0)
        assert windows[2] == (610.0, 700.0)

    def test_supervised_selector_requires_fit_before_forward(self):
        """Test that supervised selectors raise error if forward() called before fit()."""
        from cuvis_ai.node import SupervisedFullSpectrumBandSelector

        selector = SupervisedFullSpectrumBandSelector()
        cube = torch.rand(1, 4, 4, 10, dtype=torch.float32)
        wavelengths = torch.linspace(400, 900, 10, dtype=torch.float32)

        with pytest.raises(RuntimeError, match="not fitted"):
            selector.forward(cube=cube, wavelengths=wavelengths)

    def test_band_selector_wavelength_dtype_conversion(self):
        """Test that band selectors handle int32 wavelengths correctly."""
        from cuvis_ai.node import BaselineFalseRGBSelector

        selector = BaselineFalseRGBSelector()
        cube = torch.rand(1, 4, 4, 50, dtype=torch.float32)
        # Simulate int32 wavelengths from LentilsAnomalyDataNode
        wavelengths = np.linspace(400, 900, 50, dtype=np.int32)

        result = selector.forward(cube=cube, wavelengths=wavelengths)
        assert "rgb_image" in result
        assert result["rgb_image"].shape == (1, 4, 4, 3)

    def test_adaclip_detector_preprocess_handles_float32_input(self):
        """Test that AdaCLIPDetector handles float32 RGB inputs (0-1 range)."""
        from cuvis_ai.node import AdaCLIPDetector

        detector = AdaCLIPDetector()
        # Mock preprocessing to avoid model loading
        mock_preprocess = MagicMock(return_value=torch.rand(1, 3, 518, 518))
        detector._preprocess = mock_preprocess
        # Use _adaclip_model directly since _model is now a read-only property
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.predict.return_value = (
            torch.rand(1, 32, 32),
            torch.tensor([0.5]),
        )
        detector._adaclip_model = mock_model

        # Float32 input in 0-1 range
        rgb_float = torch.rand(1, 64, 64, 3, dtype=torch.float32)
        result = detector.forward(rgb_image=rgb_float)
        assert "scores" in result
        assert "anomaly_score" in result

    def test_adaclip_detector_preprocess_handles_uint8_input(self):
        """Test that AdaCLIPDetector handles uint8 RGB inputs (0-255 range)."""
        from cuvis_ai.node import AdaCLIPDetector

        detector = AdaCLIPDetector()
        mock_preprocess = MagicMock(return_value=torch.rand(1, 3, 518, 518))
        detector._preprocess = mock_preprocess
        # Use _adaclip_model directly since _model is now a read-only property
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.predict.return_value = (
            torch.rand(1, 32, 32),
            torch.tensor([0.5]),
        )
        detector._adaclip_model = mock_model

        # Uint8 input in 0-255 range
        rgb_uint8 = torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8)
        result = detector.forward(rgb_image=rgb_uint8)
        assert "scores" in result
        assert "anomaly_score" in result
