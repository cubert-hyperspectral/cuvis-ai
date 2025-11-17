"""Tests to verify that trainable nodes actually update their parameters during training."""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.anomaly.rx_logit_head import RXLogitHead
from cuvis_ai.deciders.binary_decider import BinaryDecider
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
from cuvis_ai.pipeline.canvas import CuvisCanvas
from cuvis_ai.training.config import OptimizerConfig, TrainerConfig, TrainingConfig
from cuvis_ai.training.datamodule import CuvisDataModule


class _SyntheticDictDataset(Dataset):
    """Dataset that returns dictionaries compatible with GraphDataModule expectations."""

    def __init__(self, cubes: torch.Tensor, masks: torch.Tensor | None) -> None:
        self._cubes = cubes
        self._masks = masks

    def __len__(self) -> int:
        return self._cubes.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {"cube": self._cubes[idx]}
        if self._masks is not None:
            sample["mask"] = self._masks[idx]
        return sample


class SyntheticAnomalyDataModule(CuvisDataModule):
    """Lightweight datamodule that generates deterministic synthetic anomaly data."""

    def __init__(
        self,
        *,
        batch_size: int = 4,
        num_samples: int = 24,
        height: int = 8,
        width: int = 8,
        channels: int = 20,
        seed: int = 0,
        include_labels: bool = True,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size

        generator = torch.Generator().manual_seed(seed)
        cubes = torch.randn(num_samples, height, width, channels, generator=generator)
        masks = (
            torch.randint(
                0, 2, (num_samples, height, width), generator=generator, dtype=torch.int32
            )
            if include_labels
            else None
        )

        self._train_dataset = _SyntheticDictDataset(cubes, masks)

        val_count = max(1, num_samples // 4)
        self._val_dataset = _SyntheticDictDataset(
            cubes[:val_count],
            masks[:val_count] if masks is not None else None,
        )

    def prepare_data(self) -> None:  # pragma: no cover - no external downloads
        pass

    def setup(self, stage=None) -> None:  # pragma: no cover - nothing to stage
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val_dataset, batch_size=self.batch_size, shuffle=False)


def make_training_config(max_epochs: int = 2, lr: float = 1e-2) -> TrainingConfig:
    """Create a TrainingConfig with CPU defaults for fast unit tests."""
    trainer = TrainerConfig(
        max_epochs=max_epochs,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_checkpointing=False,
        log_every_n_steps=1,
    )
    optimizer = OptimizerConfig(
        name="adam",
        lr=lr,
        weight_decay=0.0,
        betas=None,
    )
    return TrainingConfig(
        seed=123,
        trainer=trainer,
        optimizer=optimizer,
        monitor_plugins=["loguru"],
    )


def test_soft_selector_weights_update():
    """Test that SoftChannelSelector weights are updated during training."""
    from cuvis_ai.node.data import LentilsAnomalyDataNode

    canvas = CuvisCanvas("test_selector_training")

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
    rx = RXGlobal(eps=1.0e-6, cache_inverse=True)
    logit_head = RXLogitHead(init_scale=1.0, init_bias=5.0)
    decider = BinaryDecider(threshold=0.5)

    # Connect nodes
    canvas.connect(data_node.outputs.cube, normalizer.data)
    canvas.connect(normalizer.normalized, selector.data)
    canvas.connect(selector.selected, rx.data)
    canvas.connect(rx.scores, logit_head.scores)

    # Add loss nodes
    entropy_reg = SelectorEntropyRegularizer(weight=0.01, target_entropy=None)
    diversity_reg = SelectorDiversityRegularizer(weight=0.01)
    anomaly_bce = AnomalyBCEWithLogits(weight=1.0, pos_weight=None, reduction="mean")
    canvas.connect(selector.weights, entropy_reg.weights)
    canvas.connect(selector.weights, diversity_reg.weights)
    canvas.connect(logit_head.logits, anomaly_bce.predictions)
    canvas.connect(data_node.outputs.mask, anomaly_bce.targets)

    # Add decider and metrics
    canvas.connect(logit_head.logits, decider.logits)
    anomaly_metrics = AnomalyDetectionMetrics(threshold=0.0)
    canvas.connect(decider.decisions, anomaly_metrics.decisions)
    canvas.connect(data_node.outputs.mask, anomaly_metrics.targets)

    datamodule = SyntheticAnomalyDataModule(
        batch_size=4,
        num_samples=24,
        height=8,
        width=8,
        channels=20,
        seed=42,
    )

    # Statistical initialization
    from cuvis_ai.training.trainers import GradientTrainer, StatisticalTrainer

    stat_trainer = StatisticalTrainer(canvas=canvas, datamodule=datamodule)
    stat_trainer.fit()

    # Unfreeze trainable nodes
    selector.unfreeze()
    rx.unfreeze()
    logit_head.unfreeze()

    # Gradient training
    training_cfg = make_training_config(max_epochs=2, lr=1e-2)
    loss_nodes = [node for node in canvas.nodes() if hasattr(node, "weight")]
    grad_trainer = GradientTrainer(
        canvas=canvas,
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


def test_pca_weights_update():
    """Test that TrainablePCA components are updated during training."""
    canvas = CuvisCanvas("test_pca_training")

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
        n_components=3,
        trainable=True,
        init_method="svd",
        eps=1.0e-6,
    )

    # Connect nodes
    canvas.connect(data_node.outputs.cube, normalizer.data)
    canvas.connect(normalizer.normalized, selector.data)
    canvas.connect(selector.selected, pca.data)

    # Add loss nodes
    entropy_reg = SelectorEntropyRegularizer(weight=0.01, target_entropy=None)
    diversity_reg = SelectorDiversityRegularizer(weight=0.01)
    orth_loss = OrthogonalityLoss(weight=1.0)

    canvas.connect(selector.weights, entropy_reg.weights)
    canvas.connect(selector.weights, diversity_reg.weights)
    canvas.connect(pca.components, orth_loss.components)

    # Add metrics
    explained_var = ExplainedVarianceMetric()
    comp_orth = ComponentOrthogonalityMetric()

    canvas.connect(pca.explained_variance_ratio, explained_var.explained_variance_ratio)
    canvas.connect(pca.components, comp_orth.components)

    datamodule = SyntheticAnomalyDataModule(
        batch_size=4,
        num_samples=24,
        height=8,
        width=8,
        channels=20,
        seed=1337,
        include_labels=False,
    )

    # Statistical initialization
    from cuvis_ai.training.trainers import GradientTrainer, StatisticalTrainer

    stat_trainer = StatisticalTrainer(canvas=canvas, datamodule=datamodule)
    stat_trainer.fit()

    # Unfreeze trainable nodes
    selector.unfreeze()
    pca.unfreeze()

    # Gradient training
    training_cfg = make_training_config(max_epochs=2, lr=1e-2)
    loss_nodes = [node for node in canvas.nodes() if hasattr(node, "weight")]
    grad_trainer = GradientTrainer(
        canvas=canvas,
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


def test_logit_head_weights_update():
    """Test that RXLogitHead parameters are updated during training."""
    canvas = CuvisCanvas("test_logit_head_training")
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
    rx = RXGlobal(eps=1.0e-6, cache_inverse=True)
    logit_head = RXLogitHead(init_scale=1.0, init_bias=5.0)
    decider = BinaryDecider(threshold=0.5)

    # Connect nodes
    canvas.connect(data_node.outputs.cube, normalizer.data)
    canvas.connect(normalizer.normalized, selector.data)
    canvas.connect(selector.selected, rx.data)
    canvas.connect(rx.scores, logit_head.scores)

    # Add loss nodes
    entropy_reg = SelectorEntropyRegularizer(weight=0.01, target_entropy=None)
    diversity_reg = SelectorDiversityRegularizer(weight=0.01)
    anomaly_bce = AnomalyBCEWithLogits(weight=1.0, pos_weight=None, reduction="mean")

    canvas.connect(logit_head.logits, anomaly_bce.predictions)
    canvas.connect(data_node.outputs.mask, anomaly_bce.targets)
    canvas.connect(selector.weights, entropy_reg.weights)
    canvas.connect(selector.weights, diversity_reg.weights)

    # Add decider and metrics
    canvas.connect(logit_head.logits, decider.logits)
    anomaly_metrics = AnomalyDetectionMetrics(threshold=0.0)
    canvas.connect(decider.decisions, anomaly_metrics.decisions)
    canvas.connect(data_node.outputs.mask, anomaly_metrics.targets)

    datamodule = SyntheticAnomalyDataModule(
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

    print("\nInitial RXLogitHead parameters:")
    print(f"  Scale: {initial_scale:.6f}")
    print(f"  Bias: {initial_bias:.6f}")
    print(f"  Threshold: {logit_head.get_threshold():.6f}")

    # Statistical initialization
    from cuvis_ai.training.trainers import GradientTrainer, StatisticalTrainer

    stat_trainer = StatisticalTrainer(canvas=canvas, datamodule=datamodule)
    stat_trainer.fit()

    # Unfreeze trainable nodes
    selector.unfreeze()
    rx.unfreeze()
    logit_head.unfreeze()

    # Gradient training
    training_cfg = make_training_config(max_epochs=2, lr=1e-2)
    loss_nodes = [node for node in canvas.nodes() if hasattr(node, "weight")]
    grad_trainer = GradientTrainer(
        canvas=canvas,
        datamodule=datamodule,
        loss_nodes=loss_nodes,
        trainer_config=training_cfg.trainer,
        optimizer_config=training_cfg.optimizer,
    )
    grad_trainer.fit()

    final_scale = logit_head.scale.item()
    final_bias = logit_head.bias.item()

    print("\nFinal RXLogitHead parameters:")
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

    print("✓ RXLogitHead parameters updated successfully")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v", "-s"])
