"""Tests for metric leaf nodes."""

import pytest
import torch

from cuvis_ai.node import Node
from cuvis_ai.node.metrics import (
    AnomalyDetectionMetrics,
    ComponentOrthogonalityMetric,
    ExplainedVarianceMetric,
    ScoreStatisticsMetric,
)
from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.utils.types import Context, ExecutionStage


@pytest.fixture
def trainable_pca():
    """Create a TrainablePCA node for testing."""
    pca = TrainablePCA(n_components=3)

    # Initialize with dummy data (using port-based dict format)
    data_iterator = ({"data": torch.randn(2, 10, 10, 5)} for _ in range(3))
    pca.fit(data_iterator)
    pca.unfreeze()  # Convert buffers to parameters for gradient training

    return pca


class TestExplainedVarianceMetric:
    """Tests for ExplainedVarianceMetric."""

    def test_initialization(self):
        """Test ExplainedVarianceMetric initialization."""
        metric_node = ExplainedVarianceMetric()
        assert isinstance(metric_node, Node)

    def test_has_proper_execution_stages(self):
        """Test that ExplainedVarianceMetric has proper execution stages."""
        metric_node = ExplainedVarianceMetric()
        assert (
            ExecutionStage.VAL in metric_node.execution_stages
            or ExecutionStage.TEST in metric_node.execution_stages
        )

    def test_compute_metric(self, trainable_pca):
        """Test explained variance metric computation."""
        metric_node = ExplainedVarianceMetric()
        context = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

        # Get explained variance ratio from PCA
        explained_variance_ratio = (
            trainable_pca._explained_variance / trainable_pca._explained_variance.sum()
        )

        # Compute metrics using forward()
        outputs = metric_node.forward(
            explained_variance_ratio=explained_variance_ratio, context=context
        )

        # Check metrics list
        metrics_list = outputs["metrics"]
        assert isinstance(metrics_list, list)

        # Convert to dict for easier checking
        metrics = {m.name: m.value for m in metrics_list}

        # Check per-component variance
        assert "explained_variance_pc1" in metrics
        assert "explained_variance_pc2" in metrics
        assert "explained_variance_pc3" in metrics

        # Check total variance
        assert "total_explained_variance" in metrics

        # Check cumulative variance
        assert "cumulative_variance_pc1" in metrics
        assert "cumulative_variance_pc2" in metrics
        assert "cumulative_variance_pc3" in metrics

        # Variance ratios should sum to ~1
        total_variance = metrics["total_explained_variance"]
        assert 0.9 < total_variance <= 1.0  # Allow small numerical error

    def test_cumulative_variance_increases(self, trainable_pca):
        """Test that cumulative variance increases monotonically."""
        metric_node = ExplainedVarianceMetric()
        context = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

        explained_variance_ratio = (
            trainable_pca._explained_variance / trainable_pca._explained_variance.sum()
        )
        outputs = metric_node.forward(
            explained_variance_ratio=explained_variance_ratio, context=context
        )

        metrics = {m.name: m.value for m in outputs["metrics"]}

        cum1 = metrics["cumulative_variance_pc1"]
        cum2 = metrics["cumulative_variance_pc2"]
        cum3 = metrics["cumulative_variance_pc3"]

        # Should be monotonically increasing
        assert cum1 <= cum2 <= cum3


class TestAnomalyDetectionMetrics:
    """Tests for AnomalyDetectionMetrics."""

    def test_initialization(self):
        """Test AnomalyDetectionMetrics initialization."""
        metric_node = AnomalyDetectionMetrics()
        assert isinstance(metric_node, Node)

    def test_compute_metric_with_labels(self):
        """Test anomaly detection metrics computation."""
        metric_node = AnomalyDetectionMetrics()
        context = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

        # Create dummy decisions and targets
        B, H, W = 2, 10, 10
        decisions = torch.randint(0, 2, (B, H, W, 1)).bool()
        targets = torch.randint(0, 2, (B, H, W, 1)).bool()

        # Compute metrics using forward()
        outputs = metric_node.forward(decisions=decisions, targets=targets, context=context)

        # Check metrics list
        metrics_list = outputs["metrics"]
        assert isinstance(metrics_list, list)

        # Convert to dict for easier checking
        metrics = {m.name: m.value for m in metrics_list}

        # Check expected metrics exist (based on actual implementation)
        expected_metrics = [
            "precision",
            "recall",
            "f1_score",
            "iou",
        ]

        for metric_name in expected_metrics:
            assert metric_name in metrics
            assert isinstance(metrics[metric_name], float)

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        metric_node = AnomalyDetectionMetrics()
        context = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

        # Create perfect predictions
        targets = torch.randint(0, 2, (2, 10, 10, 1)).bool()
        decisions = targets.clone()

        outputs = metric_node.forward(decisions=decisions, targets=targets, context=context)

        metrics = {m.name: m.value for m in outputs["metrics"]}

        # Should have perfect scores
        assert metrics["precision"] > 0.99
        assert metrics["recall"] > 0.99
        assert metrics["f1_score"] > 0.99

    def test_all_negative_labels(self):
        """Test metrics with all negative labels."""
        metric_node = AnomalyDetectionMetrics()
        context = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

        # All negative labels
        targets = torch.zeros(2, 10, 10, 1).bool()
        decisions = torch.randint(0, 2, (2, 10, 10, 1)).bool()

        # Should not raise error
        outputs = metric_node.forward(decisions=decisions, targets=targets, context=context)

        metrics = {m.name: m.value for m in outputs["metrics"]}

        # Should have some metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_no_labels_raises_error(self):
        """Test that missing labels raises error."""
        metric_node = AnomalyDetectionMetrics()
        context = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

        decisions = torch.randint(0, 2, (2, 10, 10, 1)).bool()

        # forward() with missing required input should raise TypeError
        with pytest.raises(TypeError):
            metric_node.forward(decisions=decisions, context=context)

    def test_4d_tensor_handling(self):
        """Test metrics with 4D tensors [B, H, W, 1]."""
        metric_node = AnomalyDetectionMetrics()
        context = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

        # Create 4D tensors
        decisions = torch.randint(0, 2, (2, 10, 10, 1)).bool()
        targets = torch.randint(0, 2, (2, 10, 10, 1)).bool()

        # Should handle 4D correctly
        outputs = metric_node.forward(decisions=decisions, targets=targets, context=context)

        metrics = {m.name: m.value for m in outputs["metrics"]}

        assert "precision" in metrics


class TestScoreStatisticsMetric:
    """Tests for ScoreStatisticsMetric."""

    def test_initialization(self):
        """Test ScoreStatisticsMetric initialization."""
        metric_node = ScoreStatisticsMetric()
        assert isinstance(metric_node, Node)

    def test_compute_metric(self):
        """Test score statistics computation."""
        metric_node = ScoreStatisticsMetric()
        context = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

        # Create dummy scores with known distribution
        scores = torch.randn(2, 10, 10)

        # Compute metrics using forward()
        outputs = metric_node.forward(scores=scores, context=context)

        # Check metrics list
        metrics_list = outputs["metrics"]
        assert isinstance(metrics_list, list)

        # Convert to dict for easier checking
        metrics = {m.name: m.value for m in metrics_list}

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

    def test_quantiles_ordered(self):
        """Test that quantiles are in ascending order."""
        metric_node = ScoreStatisticsMetric()
        context = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

        scores = torch.randn(2, 10, 10)
        outputs = metric_node.forward(scores=scores, context=context)

        metrics = {m.name: m.value for m in outputs["metrics"]}

        # Quantiles should be ordered
        assert metrics["scores/min"] <= metrics["scores/q25"]
        assert metrics["scores/q25"] <= metrics["scores/median"]
        assert metrics["scores/median"] <= metrics["scores/q75"]
        assert metrics["scores/q75"] <= metrics["scores/q95"]
        assert metrics["scores/q95"] <= metrics["scores/q99"]
        assert metrics["scores/q99"] <= metrics["scores/max"]

    def test_known_distribution(self):
        """Test with known distribution."""
        metric_node = ScoreStatisticsMetric()
        context = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

        # Create uniform distribution from 0 to 1
        scores = torch.rand(10, 100, 100)  # Large sample for stable statistics
        outputs = metric_node.forward(scores=scores, context=context)

        metrics = {m.name: m.value for m in outputs["metrics"]}

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
        assert isinstance(metric_node, Node)

    def test_has_proper_execution_stages(self):
        """Test that ComponentOrthogonalityMetric has proper execution stages."""
        metric_node = ComponentOrthogonalityMetric()
        assert (
            ExecutionStage.VAL in metric_node.execution_stages
            or ExecutionStage.TEST in metric_node.execution_stages
        )

    def test_compute_metric(self, trainable_pca):
        """Test orthogonality metric computation."""
        metric_node = ComponentOrthogonalityMetric()
        context = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

        # Get components from PCA
        components = trainable_pca._components

        # Compute metrics using forward()
        outputs = metric_node.forward(components=components, context=context)

        # Check metrics list
        metrics_list = outputs["metrics"]
        assert isinstance(metrics_list, list)

        # Convert to dict for easier checking
        metrics = {m.name: m.value for m in metrics_list}

        # Check all expected metrics exist
        assert "orthogonality_error" in metrics
        assert "avg_off_diagonal" in metrics
        assert "diagonal_mean" in metrics
        assert "diagonal_std" in metrics

        # All should be floats
        for value in metrics.values():
            assert isinstance(value, float)

    def test_orthogonal_components_low_error(self, trainable_pca):
        """Test that freshly initialized PCA has low orthogonality error."""
        metric_node = ComponentOrthogonalityMetric()
        context = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

        components = trainable_pca._components
        outputs = metric_node.forward(components=components, context=context)

        metrics = {m.name: m.value for m in outputs["metrics"]}

        # SVD-initialized components should be nearly orthogonal
        assert metrics["orthogonality_error"] < 1e-4

        # Diagonal should be close to 1
        assert 0.99 < metrics["diagonal_mean"] < 1.01

        # Off-diagonal should be close to 0
        assert metrics["avg_off_diagonal"] < 0.01

    def test_degraded_orthogonality_detection(self, trainable_pca):
        """Test detection of degraded orthogonality."""
        metric_node = ComponentOrthogonalityMetric()
        context = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

        # Degrade orthogonality by adding noise
        trainable_pca._components.data += 0.1 * torch.randn_like(trainable_pca._components)

        components = trainable_pca._components
        outputs = metric_node.forward(components=components, context=context)

        metrics = {m.name: m.value for m in outputs["metrics"]}

        # Error should be higher now
        assert metrics["orthogonality_error"] > 0.01


class TestMetricNodeProtocol:
    """Tests for metric node protocol compliance."""

    def test_all_metrics_are_nodes(self):
        """Test that all metric classes inherit from Node."""
        metric_classes = [
            ExplainedVarianceMetric,
            AnomalyDetectionMetrics,
            ScoreStatisticsMetric,
            ComponentOrthogonalityMetric,
        ]

        for metric_class in metric_classes:
            assert issubclass(metric_class, Node)

    def test_all_metrics_have_forward(self):
        """Test that all metrics implement forward."""
        metric_classes = [
            ExplainedVarianceMetric(),
            AnomalyDetectionMetrics(),
            ScoreStatisticsMetric(),
            ComponentOrthogonalityMetric(),
        ]

        for metric_node in metric_classes:
            assert hasattr(metric_node, "forward")
            assert callable(metric_node.forward)
            assert "metrics" in metric_node.OUTPUT_SPECS
            assert (
                ExecutionStage.VAL in metric_node.execution_stages
                or ExecutionStage.TEST in metric_node.execution_stages
            )

    def test_metrics_return_list(self):
        """Test that all metrics return list of Metric objects."""
        # This is tested implicitly in other tests, but good to be explicit
        metric_node = ScoreStatisticsMetric()
        context = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)
        scores = torch.randn(2, 10, 10)

        outputs = metric_node.forward(scores=scores, context=context)

        assert "metrics" in outputs
        assert isinstance(outputs["metrics"], list)


class TestMetricIntegration:
    """Integration tests for metrics."""

    def test_multiple_metrics_together(self, trainable_pca):
        """Test using multiple metrics together."""
        # PCA metrics
        pca_variance = ExplainedVarianceMetric()
        pca_orthog = ComponentOrthogonalityMetric()

        context = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

        # Get data for metrics
        explained_variance_ratio = (
            trainable_pca._explained_variance / trainable_pca._explained_variance.sum()
        )
        components = trainable_pca._components

        # Compute metrics using forward()
        variance_outputs = pca_variance.forward(
            explained_variance_ratio=explained_variance_ratio, context=context
        )
        orthog_outputs = pca_orthog.forward(components=components, context=context)

        variance_metrics_list = variance_outputs["metrics"]
        orthog_metrics_list = orthog_outputs["metrics"]

        # Both should produce metrics
        assert len(variance_metrics_list) > 0
        assert len(orthog_metrics_list) > 0

        # Convert to dicts and check for no key collisions
        variance_metrics = {m.name: m.value for m in variance_metrics_list}
        orthog_metrics = {m.name: m.value for m in orthog_metrics_list}

        all_keys = set(variance_metrics.keys()) | set(orthog_metrics.keys())
        assert len(all_keys) == len(variance_metrics) + len(orthog_metrics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
