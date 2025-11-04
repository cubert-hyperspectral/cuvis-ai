"""Tests for metric leaf nodes."""

import pytest
import torch
import torch.nn as nn

from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.training.leaf_nodes import MetricNode
from cuvis_ai.training.metrics import (
    AnomalyDetectionMetrics,
    ComponentOrthogonalityMetric,
    ExplainedVarianceMetric,
    ScoreStatisticsMetric,
)


@pytest.fixture
def trainable_pca():
    """Create a TrainablePCA node for testing."""
    pca = TrainablePCA(n_components=3, trainable=True)
    
    # Initialize with dummy data
    data_iterator = [
        (torch.randn(2, 10, 10, 5), None, None)
        for _ in range(3)
    ]
    pca.initialize_from_data(data_iterator)
    pca.prepare_for_train()
    
    return pca


@pytest.fixture
def dummy_parent():
    """Create a dummy parent node."""
    class DummyNode(nn.Module):
        def forward(self, x, y=None, m=None):
            return x, y, m
    
    return DummyNode()


class TestExplainedVarianceMetric:
    """Tests for ExplainedVarianceMetric."""
    
    def test_initialization(self):
        """Test ExplainedVarianceMetric initialization."""
        metric_node = ExplainedVarianceMetric()
        assert isinstance(metric_node, MetricNode)
    
    def test_compatible_parent_types(self):
        """Test that ExplainedVarianceMetric requires TrainablePCA parent."""
        assert TrainablePCA in ExplainedVarianceMetric.compatible_parent_types
    
    def test_compute_metric(self, trainable_pca):
        """Test explained variance metric computation."""
        metric_node = ExplainedVarianceMetric()
        metric_node.parent = trainable_pca
        
        # Create dummy parent output
        dummy_output = (torch.randn(2, 10, 10, 3), None, None)
        
        # Compute metrics
        metrics = metric_node.compute_metric(dummy_output)
        
        # Check metrics dict
        assert isinstance(metrics, dict)
        
        # Check per-component variance
        assert "pca/explained_variance_pc1" in metrics
        assert "pca/explained_variance_pc2" in metrics
        assert "pca/explained_variance_pc3" in metrics
        
        # Check total variance
        assert "pca/total_explained_variance" in metrics
        
        # Check cumulative variance
        assert "pca/cumulative_variance_pc1" in metrics
        assert "pca/cumulative_variance_pc2" in metrics
        assert "pca/cumulative_variance_pc3" in metrics
        
        # Variance ratios should sum to ~1
        total_variance = metrics["pca/total_explained_variance"]
        assert 0.9 < total_variance <= 1.0  # Allow small numerical error
    
    def test_cumulative_variance_increases(self, trainable_pca):
        """Test that cumulative variance increases monotonically."""
        metric_node = ExplainedVarianceMetric()
        metric_node.parent = trainable_pca
        
        dummy_output = (torch.randn(2, 10, 10, 3), None, None)
        metrics = metric_node.compute_metric(dummy_output)
        
        cum1 = metrics["pca/cumulative_variance_pc1"]
        cum2 = metrics["pca/cumulative_variance_pc2"]
        cum3 = metrics["pca/cumulative_variance_pc3"]
        
        # Should be monotonically increasing
        assert cum1 <= cum2 <= cum3


class TestAnomalyDetectionMetrics:
    """Tests for AnomalyDetectionMetrics."""
    
    def test_initialization(self):
        """Test AnomalyDetectionMetrics initialization."""
        metric_node = AnomalyDetectionMetrics(threshold=0.5)
        assert metric_node.threshold == 0.5
        assert isinstance(metric_node, MetricNode)
    
    def test_compute_metric_with_labels(self, dummy_parent):
        """Test anomaly detection metrics computation."""
        metric_node = AnomalyDetectionMetrics(threshold=0.5)
        metric_node.parent = dummy_parent
        
        # Create dummy scores and labels
        B, H, W = 2, 10, 10
        scores = torch.randn(B, H, W)  # Will be passed through sigmoid
        labels = torch.randint(0, 2, (B, H, W)).float()
        
        parent_output = (scores, labels, None)
        
        # Compute metrics
        metrics = metric_node.compute_metric(parent_output, labels=labels)
        
        # Check all expected metrics exist
        expected_metrics = [
            "anomaly/precision",
            "anomaly/recall",
            "anomaly/f1_score",
            "anomaly/accuracy",
            "anomaly/specificity",
            "anomaly/balanced_accuracy",
            "anomaly/iou",
            "anomaly/true_positives",
            "anomaly/false_positives",
            "anomaly/true_negatives",
            "anomaly/false_negatives",
            "anomaly/positive_rate",
            "anomaly/pred_positive_rate",
        ]
        
        for metric_name in expected_metrics:
            assert metric_name in metrics
            assert isinstance(metrics[metric_name], float)
    
    def test_perfect_predictions(self, dummy_parent):
        """Test metrics with perfect predictions."""
        metric_node = AnomalyDetectionMetrics(threshold=0.5)
        metric_node.parent = dummy_parent
        
        # Create perfect predictions
        labels = torch.randint(0, 2, (2, 10, 10)).float()
        # Scores that match labels perfectly (high positive, low negative)
        scores = labels * 10.0 - (1 - labels) * 10.0  # High for 1, low for 0
        
        parent_output = (scores, labels, None)
        
        metrics = metric_node.compute_metric(parent_output, labels=labels)
        
        # Should have perfect scores
        assert metrics["anomaly/accuracy"] > 0.99
        assert metrics["anomaly/precision"] > 0.99
        assert metrics["anomaly/recall"] > 0.99
        assert metrics["anomaly/f1_score"] > 0.99
    
    def test_all_negative_labels(self, dummy_parent):
        """Test metrics with all negative labels."""
        metric_node = AnomalyDetectionMetrics(threshold=0.5)
        metric_node.parent = dummy_parent
        
        # All negative labels
        labels = torch.zeros(2, 10, 10)
        scores = torch.randn(2, 10, 10)
        
        parent_output = (scores, labels, None)
        
        # Should not raise error
        metrics = metric_node.compute_metric(parent_output, labels=labels)
        
        assert "anomaly/true_negatives" in metrics
        assert metrics["anomaly/positive_rate"] == 0.0
    
    def test_no_labels_returns_warning(self, dummy_parent):
        """Test that missing labels returns warning dict."""
        metric_node = AnomalyDetectionMetrics()
        metric_node.parent = dummy_parent
        
        scores = torch.randn(2, 10, 10)
        parent_output = (scores, None, None)
        
        metrics = metric_node.compute_metric(parent_output, labels=None)
        
        assert "warning" in metrics
        assert metrics["warning"] == "no_labels_available"
    
    def test_4d_tensor_handling(self, dummy_parent):
        """Test metrics with 4D tensors [B, H, W, 1]."""
        metric_node = AnomalyDetectionMetrics()
        metric_node.parent = dummy_parent
        
        # Create 4D tensors
        scores = torch.randn(2, 10, 10, 1)
        labels = torch.randint(0, 2, (2, 10, 10, 1)).float()
        
        parent_output = (scores, labels, None)
        
        # Should handle 4D correctly
        metrics = metric_node.compute_metric(parent_output, labels=labels)
        
        assert "anomaly/accuracy" in metrics


class TestScoreStatisticsMetric:
    """Tests for ScoreStatisticsMetric."""
    
    def test_initialization(self):
        """Test ScoreStatisticsMetric initialization."""
        metric_node = ScoreStatisticsMetric()
        assert isinstance(metric_node, MetricNode)
    
    def test_compute_metric(self, dummy_parent):
        """Test score statistics computation."""
        metric_node = ScoreStatisticsMetric()
        metric_node.parent = dummy_parent
        
        # Create dummy scores with known distribution
        scores = torch.randn(2, 10, 10)
        parent_output = (scores, None, None)
        
        # Compute metrics
        metrics = metric_node.compute_metric(parent_output)
        
        # Check all expected metrics exist
        expected_metrics = [
            "scores/mean",
            "scores/std",
            "scores/min",
            "scores/max",
            "scores/median",
            "scores/q25",
            "scores/q75",
            "scores/q95",
            "scores/q99",
        ]
        
        for metric_name in expected_metrics:
            assert metric_name in metrics
            assert isinstance(metrics[metric_name], float)
    
    def test_quantiles_ordered(self, dummy_parent):
        """Test that quantiles are in ascending order."""
        metric_node = ScoreStatisticsMetric()
        metric_node.parent = dummy_parent
        
        scores = torch.randn(2, 10, 10)
        parent_output = (scores, None, None)
        
        metrics = metric_node.compute_metric(parent_output)
        
        # Quantiles should be ordered
        assert metrics["scores/min"] <= metrics["scores/q25"]
        assert metrics["scores/q25"] <= metrics["scores/median"]
        assert metrics["scores/median"] <= metrics["scores/q75"]
        assert metrics["scores/q75"] <= metrics["scores/q95"]
        assert metrics["scores/q95"] <= metrics["scores/q99"]
        assert metrics["scores/q99"] <= metrics["scores/max"]
    
    def test_known_distribution(self, dummy_parent):
        """Test with known distribution."""
        metric_node = ScoreStatisticsMetric()
        metric_node.parent = dummy_parent
        
        # Create uniform distribution from 0 to 1
        scores = torch.rand(10, 100, 100)  # Large sample for stable statistics
        parent_output = (scores, None, None)
        
        metrics = metric_node.compute_metric(parent_output)
        
        # Mean should be close to 0.5
        assert 0.4 < metrics["scores/mean"] < 0.6
        
        # Min should be close to 0
        assert 0.0 <= metrics["scores/min"] < 0.1
        
        # Max should be close to 1
        assert 0.9 < metrics["scores/max"] <= 1.0


class TestComponentOrthogonalityMetric:
    """Tests for ComponentOrthogonalityMetric."""
    
    def test_initialization(self):
        """Test ComponentOrthogonalityMetric initialization."""
        metric_node = ComponentOrthogonalityMetric()
        assert isinstance(metric_node, MetricNode)
    
    def test_compatible_parent_types(self):
        """Test that ComponentOrthogonalityMetric requires TrainablePCA parent."""
        assert TrainablePCA in ComponentOrthogonalityMetric.compatible_parent_types
    
    def test_compute_metric(self, trainable_pca):
        """Test orthogonality metric computation."""
        metric_node = ComponentOrthogonalityMetric()
        metric_node.parent = trainable_pca
        
        # Create dummy parent output
        dummy_output = (torch.randn(2, 10, 10, 3), None, None)
        
        # Compute metrics
        metrics = metric_node.compute_metric(dummy_output)
        
        # Check all expected metrics exist
        assert "pca/orthogonality_error" in metrics
        assert "pca/avg_off_diagonal" in metrics
        assert "pca/diagonal_mean" in metrics
        assert "pca/diagonal_std" in metrics
        
        # All should be floats
        for value in metrics.values():
            assert isinstance(value, float)
    
    def test_orthogonal_components_low_error(self, trainable_pca):
        """Test that freshly initialized PCA has low orthogonality error."""
        metric_node = ComponentOrthogonalityMetric()
        metric_node.parent = trainable_pca
        
        dummy_output = (torch.randn(2, 10, 10, 3), None, None)
        metrics = metric_node.compute_metric(dummy_output)
        
        # SVD-initialized components should be nearly orthogonal
        assert metrics["pca/orthogonality_error"] < 1e-4
        
        # Diagonal should be close to 1
        assert 0.99 < metrics["pca/diagonal_mean"] < 1.01
        
        # Off-diagonal should be close to 0
        assert metrics["pca/avg_off_diagonal"] < 0.01
    
    def test_degraded_orthogonality_detection(self, trainable_pca):
        """Test detection of degraded orthogonality."""
        metric_node = ComponentOrthogonalityMetric()
        metric_node.parent = trainable_pca
        
        # Degrade orthogonality by adding noise
        trainable_pca.components.data += 0.1 * torch.randn_like(trainable_pca.components)
        
        dummy_output = (torch.randn(2, 10, 10, 3), None, None)
        metrics = metric_node.compute_metric(dummy_output)
        
        # Error should be higher now
        assert metrics["pca/orthogonality_error"] > 0.01


class TestMetricNodeProtocol:
    """Tests for metric node protocol compliance."""
    
    def test_all_metrics_are_metric_nodes(self):
        """Test that all metric classes inherit from MetricNode."""
        metric_classes = [
            ExplainedVarianceMetric,
            AnomalyDetectionMetrics,
            ScoreStatisticsMetric,
            ComponentOrthogonalityMetric,
        ]
        
        for metric_class in metric_classes:
            assert issubclass(metric_class, MetricNode)
    
    def test_all_metrics_have_compute_metric(self):
        """Test that all metrics implement compute_metric."""
        metric_classes = [
            ExplainedVarianceMetric(),
            AnomalyDetectionMetrics(),
            ScoreStatisticsMetric(),
            ComponentOrthogonalityMetric(),
        ]
        
        for metric_node in metric_classes:
            assert hasattr(metric_node, 'compute_metric')
            assert callable(metric_node.compute_metric)
    
    def test_metrics_return_dict(self):
        """Test that all metrics return dictionaries."""
        # This is tested implicitly in other tests, but good to be explicit
        pass


class TestMetricIntegration:
    """Integration tests for metrics."""
    
    def test_multiple_metrics_together(self, trainable_pca, dummy_parent):
        """Test using multiple metrics on the same parent."""
        # PCA metrics
        pca_variance = ExplainedVarianceMetric()
        pca_variance.parent = trainable_pca
        
        pca_orthog = ComponentOrthogonalityMetric()
        pca_orthog.parent = trainable_pca
        
        dummy_output = (torch.randn(2, 10, 10, 3), None, None)
        
        variance_metrics = pca_variance.compute_metric(dummy_output)
        orthog_metrics = pca_orthog.compute_metric(dummy_output)
        
        # Both should produce metrics
        assert len(variance_metrics) > 0
        assert len(orthog_metrics) > 0
        
        # No key collisions
        all_keys = set(variance_metrics.keys()) | set(orthog_metrics.keys())
        assert len(all_keys) == len(variance_metrics) + len(orthog_metrics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
