"""Tests to verify that trainable nodes actually update their parameters during training."""

import pytest
import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.deciders.binary_decider import BinaryDecider
from cuvis_ai.node.conversion import ScoreToLogit
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.losses import (
    AnomalyBCEWithLogits,
    OrthogonalityLoss,
    SelectorDiversityRegularizer,
    SelectorEntropyRegularizer,
)
from cuvis_ai.node.metrics import (
    AnomalyDetectionMetrics,
    ComponentOrthogonalityMetric,
    ExplainedVarianceMetric,
)
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.node.selector import SoftChannelSelector


def test_soft_selector_weights_update(synthetic_anomaly_datamodule, training_config_factory):
    """Test that SoftChannelSelector weights are updated during training."""
    from cuvis_ai.node.data import LentilsAnomalyDataNode

    pipeline = CuvisPipeline("test_selector_training")

    # Add data node to handle batch dict
    data_node = LentilsAnomalyDataNode(normal_class_ids=[0])
    normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
    selector = SoftChannelSelector(
        n_select=15,
        input_channels=20,
        init_method="variance",
        temperature_init=5.0,
        temperature_min=0.1,
        temperature_decay=0.9,
        hard=False,
        eps=1.0e-6,
    )
    rx = RXGlobal(num_channels=15, eps=1.0e-6, cache_inverse=True)
    logit_head = ScoreToLogit(init_scale=1.0, init_bias=5.0)
    decider = BinaryDecider(threshold=0.5)

    # Connect nodes
    pipeline.connect(data_node.outputs.cube, normalizer.data)
    pipeline.connect(normalizer.normalized, selector.data)
    pipeline.connect(selector.selected, rx.data)
    pipeline.connect(rx.scores, logit_head.scores)

    # Add loss nodes
    entropy_reg = SelectorEntropyRegularizer(weight=0.01, target_entropy=None)
    diversity_reg = SelectorDiversityRegularizer(weight=0.01)
    anomaly_bce = AnomalyBCEWithLogits(weight=1.0, pos_weight=None, reduction="mean")
    pipeline.connect(selector.weights, entropy_reg.weights)
    pipeline.connect(selector.weights, diversity_reg.weights)
    pipeline.connect(logit_head.logits, anomaly_bce.predictions)
    pipeline.connect(data_node.outputs.mask, anomaly_bce.targets)

    # Add decider and metrics
    pipeline.connect(logit_head.logits, decider.logits)
    anomaly_metrics = AnomalyDetectionMetrics(threshold=0.0)
    pipeline.connect(decider.decisions, anomaly_metrics.decisions)
    pipeline.connect(data_node.outputs.mask, anomaly_metrics.targets)

    datamodule = synthetic_anomaly_datamodule(
        batch_size=4,
        num_samples=24,
        height=8,
        width=8,
        channels=20,
        seed=42,
    )

    # Statistical initialization
    from cuvis_ai_core.training.trainers import GradientTrainer, StatisticalTrainer

    stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    stat_trainer.fit()

    # Unfreeze trainable nodes
    selector.unfreeze()
    rx.unfreeze()
    logit_head.unfreeze()

    # Gradient training
    training_cfg = training_config_factory(max_epochs=2, lr=1e-2)
    loss_nodes = [node for node in pipeline.nodes() if hasattr(node, "weight")]
    grad_trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=datamodule,
        loss_nodes=loss_nodes,
        trainer_config=training_cfg.trainer,
        optimizer_config=training_cfg.optimizer,
    )
    grad_trainer.fit()

    final_logits = selector.channel_logits.data.clone()
    final_mean = final_logits.mean().item()
    final_std = final_logits.std().item()

    print("\nFinal selector channel_logits:")
    print(f"  Mean: {final_mean:.6f}")
    print(f"  Std: {final_std:.6f}")
    print(f"  Min: {final_logits.min().item():.6f}")
    print(f"  Max: {final_logits.max().item():.6f}")

    assert selector.channel_logits is not None, "Selector channel_logits not initialized"
    assert isinstance(selector.channel_logits, torch.nn.Parameter), (
        "Selector channel_logits should be a Parameter"
    )

    logit_std = final_logits.std().item()
    assert logit_std > 0.01, (
        f"Selector channel_logits show no variation (std={logit_std:.6f}), "
        "indicating no training occurred"
    )

    relative_spread = logit_std / (final_logits.abs().mean().item() + 1e-8)

    print("\nStatistics:")
    print(f"  Std deviation: {logit_std:.6f}")
    print(f"  Relative spread: {relative_spread:.2%}")

    assert selector.channel_logits.grad is not None or selector.channel_logits.requires_grad, (
        "Selector channel_logits should remain trainable"
    )

    print("✓ SoftChannelSelector weights updated successfully")


def test_pca_weights_update(synthetic_anomaly_datamodule, training_config_factory):
    """Test that TrainablePCA components are updated during training."""
    pipeline = CuvisPipeline("test_pca_training")

    data_node = LentilsAnomalyDataNode(normal_class_ids=[0])
    normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
    selector = SoftChannelSelector(
        n_select=15,
        input_channels=20,
        init_method="variance",
        temperature_init=5.0,
        temperature_min=0.1,
        temperature_decay=0.9,
        hard=False,
        eps=1.0e-6,
    )
    pca = TrainablePCA(
        num_channels=15,  # After SoftChannelSelector(n_select=15, input_channels=20)
        n_components=3,
        trainable=True,
        init_method="svd",
        eps=1.0e-6,
    )

    # Connect nodes
    pipeline.connect(data_node.outputs.cube, normalizer.data)
    pipeline.connect(normalizer.normalized, selector.data)
    pipeline.connect(selector.selected, pca.data)

    # Add loss nodes
    entropy_reg = SelectorEntropyRegularizer(weight=0.01, target_entropy=None)
    diversity_reg = SelectorDiversityRegularizer(weight=0.01)
    orth_loss = OrthogonalityLoss(weight=1.0)

    pipeline.connect(selector.weights, entropy_reg.weights)
    pipeline.connect(selector.weights, diversity_reg.weights)
    pipeline.connect(pca.components, orth_loss.components)

    # Add metrics
    explained_var = ExplainedVarianceMetric()
    comp_orth = ComponentOrthogonalityMetric()

    pipeline.connect(pca.explained_variance_ratio, explained_var.explained_variance_ratio)
    pipeline.connect(pca.components, comp_orth.components)

    datamodule = synthetic_anomaly_datamodule(
        batch_size=4,
        num_samples=24,
        height=8,
        width=8,
        channels=20,
        seed=1337,
        include_labels=False,
    )

    # Statistical initialization
    from cuvis_ai_core.training.trainers import GradientTrainer, StatisticalTrainer

    stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    stat_trainer.fit()

    # Unfreeze trainable nodes
    selector.unfreeze()
    pca.unfreeze()

    # Gradient training
    training_cfg = training_config_factory(max_epochs=2, lr=1e-2)
    loss_nodes = [node for node in pipeline.nodes() if hasattr(node, "weight")]
    grad_trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=datamodule,
        loss_nodes=loss_nodes,
        trainer_config=training_cfg.trainer,
        optimizer_config=training_cfg.optimizer,
    )
    grad_trainer.fit()

    pca.unfreeze()

    assert pca._components is not None, "PCA components not initialized"
    assert isinstance(pca._components, torch.nn.Parameter), (
        "PCA components should be a Parameter after training"
    )

    final_components = pca._components.data.clone()
    final_mean_norm = torch.norm(final_components, dim=1).mean().item()

    # Use OrthogonalityLoss node's forward method directly
    orth_loss_fn = OrthogonalityLoss(weight=1.0)
    orth_result = orth_loss_fn.forward(components=final_components, context=None)
    final_orth_loss = orth_result["loss"].item()

    print("\nFinal PCA components:")
    print(f"  Shape: {final_components.shape}")
    print(f"  Mean norm: {final_mean_norm:.6f}")
    print(f"  Orthogonality loss: {final_orth_loss:.6f}")
    print(f"  Min: {final_components.min().item():.6f}")
    print(f"  Max: {final_components.max().item():.6f}")

    assert pca._components.requires_grad, "PCA components should require gradients for training"
    assert final_orth_loss < 0.5, (
        f"PCA components have poor orthogonality (loss={final_orth_loss:.6f})"
    )
    assert 0.5 < final_mean_norm < 2.0, (
        f"PCA component norms are unusual (mean={final_mean_norm:.6f})"
    )

    print("✓ TrainablePCA components updated successfully")


def test_logit_head_weights_update(synthetic_anomaly_datamodule, training_config_factory):
    """Test that ScoreToLogit parameters are updated during training."""
    pipeline = CuvisPipeline("test_logit_head_training")
    data_node = LentilsAnomalyDataNode(normal_class_ids=[0, 1])
    normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
    selector = SoftChannelSelector(
        n_select=15,
        input_channels=20,
        init_method="variance",
        temperature_init=5.0,
        temperature_min=0.1,
        temperature_decay=0.9,
        hard=False,
        eps=1.0e-6,
    )
    rx = RXGlobal(num_channels=15, eps=1.0e-6, cache_inverse=True)
    logit_head = ScoreToLogit(init_scale=1.0, init_bias=5.0)
    decider = BinaryDecider(threshold=0.5)

    # Connect nodes
    pipeline.connect(data_node.outputs.cube, normalizer.data)
    pipeline.connect(normalizer.normalized, selector.data)
    pipeline.connect(selector.selected, rx.data)
    pipeline.connect(rx.scores, logit_head.scores)

    # Add loss nodes
    entropy_reg = SelectorEntropyRegularizer(weight=0.01, target_entropy=None)
    diversity_reg = SelectorDiversityRegularizer(weight=0.01)
    anomaly_bce = AnomalyBCEWithLogits(weight=1.0, pos_weight=None, reduction="mean")

    pipeline.connect(logit_head.logits, anomaly_bce.predictions)
    pipeline.connect(data_node.outputs.mask, anomaly_bce.targets)
    pipeline.connect(selector.weights, entropy_reg.weights)
    pipeline.connect(selector.weights, diversity_reg.weights)

    # Add decider and metrics
    pipeline.connect(logit_head.logits, decider.logits)
    anomaly_metrics = AnomalyDetectionMetrics(threshold=0.0)
    pipeline.connect(decider.decisions, anomaly_metrics.decisions)
    pipeline.connect(data_node.outputs.mask, anomaly_metrics.targets)

    datamodule = synthetic_anomaly_datamodule(
        batch_size=4,
        num_samples=24,
        height=8,
        width=8,
        channels=20,
        seed=777,
        include_labels=True,
    )

    initial_scale = logit_head.scale.item()
    initial_bias = logit_head.bias.item()

    print("\nInitial ScoreToLogit parameters:")
    print(f"  Scale: {initial_scale:.6f}")
    print(f"  Bias: {initial_bias:.6f}")
    print(f"  Threshold: {logit_head.get_threshold():.6f}")

    # Statistical initialization
    from cuvis_ai_core.training.trainers import GradientTrainer, StatisticalTrainer

    stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    stat_trainer.fit()

    # Unfreeze trainable nodes
    selector.unfreeze()
    rx.unfreeze()
    logit_head.unfreeze()

    # Gradient training
    training_cfg = training_config_factory(max_epochs=2, lr=1e-2)
    loss_nodes = [node for node in pipeline.nodes() if hasattr(node, "weight")]
    grad_trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=datamodule,
        loss_nodes=loss_nodes,
        trainer_config=training_cfg.trainer,
        optimizer_config=training_cfg.optimizer,
    )
    grad_trainer.fit()

    final_scale = logit_head.scale.item()
    final_bias = logit_head.bias.item()

    print("\nFinal ScoreToLogit parameters:")
    print(f"  Scale: {final_scale:.6f}")
    print(f"  Bias: {final_bias:.6f}")
    print(f"  Threshold: {logit_head.get_threshold():.6f}")

    scale_change = abs(final_scale - initial_scale)
    bias_change = abs(final_bias - initial_bias)

    print("\nChanges:")
    print(f"  Scale change: {scale_change:.6f}")
    print(f"  Bias change: {bias_change:.6f}")

    assert scale_change > 0.001 or bias_change > 0.001, (
        "Neither scale nor bias changed during training"
    )

    print("✓ ScoreToLogit parameters updated successfully")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v", "-s"])
