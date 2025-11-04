"""Tests for visualization leaf nodes."""

import pickle
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.normalization.normalization import MinMaxNormalizer
from cuvis_ai.training.visualizations import (
    AnomalyHeatmap,
    PCAVisualization,
    ScoreHistogram,
)


@pytest.fixture
def sample_spectral_data():
    """Create sample spectral cube data (B, C, H, W)."""
    torch.manual_seed(42)
    return torch.randn(2, 61, 10, 15)  # 2 batches, 61 channels, 10x15 spatial


@pytest.fixture
def sample_anomaly_scores():
    """Create sample anomaly score maps (B, H, W)."""
    torch.manual_seed(42)
    scores = torch.randn(2, 10, 15)
    # Make some pixels clearly anomalous
    scores[0, 2:4, 3:6] = 10.0  # High anomaly region
    return scores


@pytest.fixture
def sample_batch():
    """Create sample batch dictionary."""
    return {
        "cube": torch.randn(2, 61, 10, 15),
        "mask": torch.zeros(2, 10, 15),
    }


class TestPCAVisualization:
    """Tests for PCAVisualization leaf node."""
    
    def test_initialization_2d(self):
        """Test 2D PCA visualization initialization."""
        viz = PCAVisualization(n_components=2, log_every_n_epochs=5)
        assert viz.n_components == 2
        assert viz.log_every_n_epochs == 5
        assert viz.max_samples == 1000
    
    def test_initialization_3d(self):
        """Test 3D PCA visualization initialization."""
        viz = PCAVisualization(n_components=3)
        assert viz.n_components == 3
    
    def test_invalid_n_components(self):
        """Test that invalid n_components raises error."""
        with pytest.raises(ValueError, match="n_components must be 2 or 3"):
            PCAVisualization(n_components=4)
    
    def test_visualize_2d(self, sample_spectral_data, sample_batch):
        """Test 2D PCA visualization generation."""
        viz = PCAVisualization(n_components=2)
        
        artifacts = viz.visualize(
            parent_output=sample_spectral_data,
            batch=sample_batch,
            logger=None,
            current_epoch=0
        )
        
        # Check artifact structure
        assert isinstance(artifacts, dict)
        assert 'figure' in artifacts
        assert 'type' in artifacts
        assert artifacts['type'] == 'pca_projection'
        assert 'explained_variance' in artifacts
        assert 'n_components' in artifacts
        assert artifacts['n_components'] == 2
        
        # Check figure
        fig = artifacts['figure']
        assert isinstance(fig, plt.Figure)
        
        # Check explained variance
        assert len(artifacts['explained_variance']) == 2
        assert all(0 <= v <= 1 for v in artifacts['explained_variance'])
        
        plt.close(fig)
    
    def test_visualize_3d(self, sample_spectral_data, sample_batch):
        """Test 3D PCA visualization generation."""
        viz = PCAVisualization(n_components=3)
        
        artifacts = viz.visualize(
            parent_output=sample_spectral_data,
            batch=sample_batch,
            logger=None,
            current_epoch=5
        )
        
        assert artifacts['n_components'] == 3
        assert len(artifacts['explained_variance']) == 3
        
        plt.close(artifacts['figure'])
    
    def test_frequency_checking(self):
        """Test that frequency checking works correctly."""
        viz = PCAVisualization(n_components=2, log_every_n_epochs=3)
        
        assert viz.should_log(0) is True   # First epoch
        assert viz.should_log(1) is False  # Too soon
        assert viz.should_log(2) is False  # Too soon
        assert viz.should_log(3) is True   # 3 epochs since last
        assert viz.should_log(4) is False  # Too soon
        assert viz.should_log(6) is True   # 3 epochs since last
    
    def test_subsampling(self, sample_batch):
        """Test that large datasets are subsampled."""
        # Create large dataset
        large_data = torch.randn(100, 61, 10, 15)  # 100 batches -> 15000 samples
        
        viz = PCAVisualization(n_components=2, max_samples=500)
        
        artifacts = viz.visualize(
            parent_output=large_data,
            batch=sample_batch,
            logger=None,
            current_epoch=0
        )
        
        # Should have subsampled
        assert artifacts is not None
        plt.close(artifacts['figure'])


class TestAnomalyHeatmap:
    """Tests for AnomalyHeatmap leaf node."""
    
    def test_initialization(self):
        """Test anomaly heatmap initialization."""
        viz = AnomalyHeatmap(log_every_n_epochs=2, cmap='hot')
        assert viz.cmap == 'hot'
        assert viz.log_every_n_epochs == 2
        assert viz.vmin is None
        assert viz.vmax is None
    
    def test_visualize_3d_scores(self, sample_anomaly_scores, sample_batch):
        """Test heatmap generation with 3D scores (B, H, W)."""
        viz = AnomalyHeatmap(cmap='hot')
        
        artifacts = viz.visualize(
            parent_output=sample_anomaly_scores,
            batch=sample_batch,
            logger=None,
            current_epoch=0
        )
        
        # Check artifact structure
        assert isinstance(artifacts, dict)
        assert 'figure' in artifacts
        assert 'type' in artifacts
        assert artifacts['type'] == 'anomaly_heatmap'
        assert 'statistics' in artifacts
        
        # Check statistics
        stats = artifacts['statistics']
        assert 'min' in stats
        assert 'max' in stats
        assert 'mean' in stats
        assert 'std' in stats
        assert 'threshold' in stats
        assert 'anomaly_count' in stats
        assert 'total_pixels' in stats
        
        # Check figure has at least 2 subplots (heatmap + binary mask, plus colorbar axes)
        fig = artifacts['figure']
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2  # May have additional colorbar axes
        
        plt.close(fig)
    
    def test_visualize_4d_scores(self, sample_batch):
        """Test heatmap generation with 4D scores (B, 1, H, W)."""
        scores_4d = torch.randn(2, 1, 10, 15)
        viz = AnomalyHeatmap()
        
        artifacts = viz.visualize(
            parent_output=scores_4d,
            batch=sample_batch,
            logger=None,
            current_epoch=0
        )
        
        assert artifacts['type'] == 'anomaly_heatmap'
        plt.close(artifacts['figure'])
    
    def test_invalid_shape(self, sample_batch):
        """Test that invalid shapes raise error."""
        invalid_scores = torch.randn(2, 3, 10, 15)  # Wrong channel dimension
        viz = AnomalyHeatmap()
        
        with pytest.raises(ValueError, match="Expected .* got"):
            viz.visualize(
                parent_output=invalid_scores,
                batch=sample_batch,
                logger=None,
                current_epoch=0
            )
    
    def test_custom_colormap_range(self, sample_anomaly_scores, sample_batch):
        """Test custom vmin/vmax settings."""
        viz = AnomalyHeatmap(vmin=0.0, vmax=1.0)
        
        artifacts = viz.visualize(
            parent_output=sample_anomaly_scores,
            batch=sample_batch,
            logger=None,
            current_epoch=0
        )
        
        assert artifacts is not None
        plt.close(artifacts['figure'])


class TestScoreHistogram:
    """Tests for ScoreHistogram leaf node."""
    
    def test_initialization(self):
        """Test score histogram initialization."""
        viz = ScoreHistogram(log_every_n_epochs=3, bins=100)
        assert viz.bins == 100
        assert viz.log_every_n_epochs == 3
    
    def test_visualize(self, sample_anomaly_scores, sample_batch):
        """Test histogram generation."""
        viz = ScoreHistogram(bins=50)
        
        artifacts = viz.visualize(
            parent_output=sample_anomaly_scores,
            batch=sample_batch,
            logger=None,
            current_epoch=0
        )
        
        # Check artifact structure
        assert isinstance(artifacts, dict)
        assert 'figure' in artifacts
        assert 'type' in artifacts
        assert artifacts['type'] == 'score_histogram'
        assert 'statistics' in artifacts
        
        # Check statistics
        stats = artifacts['statistics']
        assert 'count' in stats
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'threshold' in stats
        assert 'anomaly_count' in stats
        assert 'anomaly_percentage' in stats
        
        # Check figure
        fig = artifacts['figure']
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        plt.close(fig)
    
    def test_anomaly_detection_threshold(self, sample_batch):
        """Test that threshold is computed correctly (mean + 2*std)."""
        # Create scores with known distribution
        torch.manual_seed(42)
        scores = torch.randn(10, 10, 10) * 2 + 5  # mean~5, std~2
        
        viz = ScoreHistogram()
        artifacts = viz.visualize(
            parent_output=scores,
            batch=sample_batch,
            logger=None,
            current_epoch=0
        )
        
        stats = artifacts['statistics']
        expected_threshold = stats['mean'] + 2 * stats['std']
        assert abs(stats['threshold'] - expected_threshold) < 0.001
        
        plt.close(artifacts['figure'])
    
    def test_various_data_shapes(self, sample_batch):
        """Test histogram with various input shapes."""
        viz = ScoreHistogram()
        
        # Test different shapes
        shapes = [
            (2, 10, 15),           # 3D
            (2, 1, 10, 15),        # 4D with channel=1
            (2, 61, 10, 15),       # 4D with channels
            (100,),                # 1D
        ]
        
        for shape in shapes:
            data = torch.randn(shape)
            artifacts = viz.visualize(
                parent_output=data,
                batch=sample_batch,
                logger=None,
                current_epoch=0
            )
            assert artifacts['type'] == 'score_histogram'
            plt.close(artifacts['figure'])


class TestVisualizationParentValidation:
    """Tests for parent validation in visualization nodes."""
    
    def test_pca_accepts_any_parent(self):
        """Test that PCA visualization doesn't restrict parent types."""
        viz = PCAVisualization(n_components=2)
        
        # Should accept any node (no parent type restrictions)
        normalizer = MinMaxNormalizer()
        viz.validate_parent(normalizer)  # Should not raise
        
        rx = RXGlobal()
        viz.validate_parent(rx)  # Should not raise
    
    def test_all_visualizations_are_nn_modules(self):
        """Test that all visualization nodes inherit from nn.Module."""
        from torch import nn
        
        viz_classes = [PCAVisualization, AnomalyHeatmap, ScoreHistogram]
        
        for viz_cls in viz_classes:
            viz = viz_cls()
            assert isinstance(viz, nn.Module)


@pytest.mark.integration
def test_visualization_artifact_serialization(sample_anomaly_scores, sample_batch):
    """Test that visualization artifacts can be pickled and unpickled."""
    viz = AnomalyHeatmap()
    
    artifacts = viz.visualize(
        parent_output=sample_anomaly_scores,
        batch=sample_batch,
        logger=None,
        current_epoch=0
    )
    
    # Serialize and deserialize
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
        pickle.dump(artifacts, f)
        pkl_path = f.name
    
    try:
        with open(pkl_path, 'rb') as f:
            loaded_artifacts = pickle.load(f)
        
        # Check that deserialization worked
        assert loaded_artifacts['type'] == 'anomaly_heatmap'
        assert 'statistics' in loaded_artifacts
        assert isinstance(loaded_artifacts['figure'], plt.Figure)
        
        plt.close(artifacts['figure'])
        plt.close(loaded_artifacts['figure'])
    finally:
        Path(pkl_path).unlink()
