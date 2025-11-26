"""Phase 3 gradient training with a SoftChannelSelector plus trainable PCA.

The pipeline stacks min-max normalization, selector initialization, PCA
regularization, callbacks (early stopping, checkpoints, LR scheduling),
diagnostic metrics, channel visualizations, and TensorBoard logging in a
full train/val/test workflow.

Why Channel Selector + PCA?
----------------------------
Channel Selector performs feature SELECTION (61→15 channels) - picking which physical
wavelengths to keep, providing high interpretability and physics-informed preprocessing.
PCA performs feature EXTRACTION (15→3 components) - creating orthogonal combinations that
capture maximum variance. Using both in sequence enables: (1) physics-informed preprocessing
to remove noisy/irrelevant spectral regions, (2) computational efficiency, (3) better PCA
quality by focusing on meaningful patterns, (4) joint trainability where selector learns
informative wavelengths while PCA learns optimal combinations, and (5) different regularization
strategies (entropy/diversity for selector, orthogonality for PCA).

Run with:
    python examples_torch/02_gradient_training.py

Graph Visualization:
Pipeline Flow:
==============
Data → MinMax → Selector ─┬→ PCA ─┬→ Losses (Entropy, Diversity, Orthogonality)
                          │       ├→ Metrics (SelEntropy, SelDiversity, ExplVar, CompOrth, PCAQuality)
                          │       └→ Visualizations (PCA, RGB) ─→ TensorBoard
                          └→ Selector Metrics (Entropy, Diversity)

Training: (1) Statistical Init [Frozen] → (2) Unfreeze → (3) Gradient Optimize [Losses+Metrics+Callbacks]


Training Phases:
----------------
1. Statistical Initialization: Initialize selector (variance) and PCA (SVD) with frozen weights
2. Unfreeze: Enable gradient flow through selector and PCA
3. Gradient Training: Optimize with:
   - Losses: entropy_loss, diversity_loss, orthogonality_loss
   - Metrics: selector_entropy, selector_diversity, explained_variance, component_orthogonality, pca_quality
   - Callbacks: early_stopping, model_checkpoint, lr_monitor
"""

from __future__ import annotations

from typing import Any

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig
from torch import Tensor

from cuvis_ai.data.lentils_anomaly import SingleCu3sDataModule
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.losses import (
    OrthogonalityLoss,
    SelectorDiversityRegularizer,
    SelectorEntropyRegularizer,
)
from cuvis_ai.node.metrics import (
    ComponentOrthogonalityMetric,
    ExplainedVarianceMetric,
    SelectorDiversityMetric,
    SelectorEntropyMetric,
)
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.node import Node
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.node.selector import SoftChannelSelector
from cuvis_ai.node.visualizations import CubeRGBVisualizer, PCAVisualization
from cuvis_ai.pipeline.canvas import CuvisCanvas
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.training import GradientTrainer, StatisticalTrainer
from cuvis_ai.training.config import (
    CallbacksConfig,
    EarlyStoppingConfig,
    LearningRateMonitorConfig,
    ModelCheckpointConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
    TrainingConfig,
)
from cuvis_ai.utils.types import Context, ExecutionStage, Metric


def summarize_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a rounded dict for the first metrics record."""
    if not results:
        return {}

    summary = results[0]
    formatted: dict[str, Any] = {}
    for key, value in summary.items():
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.numel() == 1:
                value = float(value.item())
            else:
                value = value.cpu().tolist()

        if isinstance(value, (int, float)):
            formatted[key] = round(float(value), 4)
        else:
            formatted[key] = value
    return formatted


class PCAQualityMetrics(Node):
    """Report PCA variance coverage and component spread during val/test."""

    INPUT_SPECS = {
        "explained_variance_ratio": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Explained variance ratio per component",
        ),
        "projected": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="PCA projected data [B, H, W, n_components]",
        ),
    }

    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects")}

    def __init__(self, **kwargs) -> None:
        super().__init__(execution_stages={ExecutionStage.VAL, ExecutionStage.TEST}, **kwargs)

    def forward(
        self, explained_variance_ratio: Tensor, projected: Tensor, context: Context
    ) -> dict[str, Any]:
        """Return PCA variance and component-std metrics for the batch."""
        # Cumulative explained variance
        cumulative_variance = float(explained_variance_ratio.sum().item())

        # Component spread across PCA dimensions
        component_stds = projected.std(dim=(0, 1, 2))
        avg_component_std = float(component_stds.mean().item())
        min_component_std = float(component_stds.min().item())
        max_component_std = float(component_stds.max().item())

        metrics = [
            Metric(
                name="pca/cumulative_variance",
                value=cumulative_variance,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="pca/avg_component_std",
                value=avg_component_std,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="pca/min_component_std",
                value=min_component_std,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="pca/max_component_std",
                value=max_component_std,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
        ]

        return {"metrics": metrics}

    def serialize(self, serial_dir: str) -> dict:
        return {**self.hparams}

    def load(self, params: dict, serial_dir: str) -> None:
        pass


@hydra.main(version_base=None, config_path="../cuvis_ai/conf", config_name="general")
def main(cfg: DictConfig) -> None:
    """Main training function with gradient-based optimization."""

    logger.info("SoftChannelSelector -> TrainablePCA with callbacks.")

    # Stage 1: Prepare datamodule
    datamodule = SingleCu3sDataModule(
        data_dir="../data/Lentils",
        batch_size=4,
        train_ids=[0, 1, 2],
        val_ids=[3, 4, 5],
        test_ids=[9, 10, 11, 12, 13],
    )
    datamodule.setup("fit")

    wavelengths = datamodule.train_ds.wavelengths
    wavelengths_tensor = torch.as_tensor(wavelengths)
    lambda_min = float(wavelengths_tensor.min().item())
    lambda_max = float(wavelengths_tensor.max().item())

    logger.info(
        f"Spectral coverage: {wavelengths_tensor.numel()} channels spanning {lambda_min:.1f}-{lambda_max:.1f} nm"
    )

    # Stage 2: Build canvas
    canvas = CuvisCanvas("Gradient_Training_Channel_Selector_PCA")

    # Data node
    data_node = LentilsAnomalyDataNode(normal_class_ids=[0, 1], wavelengths=wavelengths)

    # Stage 3: Processing nodes
    normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
    selector = SoftChannelSelector(
        n_select=15,
        input_channels=61,  # Lentils dataset has 61 channels
        init_method="variance",
        temperature_init=5.0,
        temperature_min=0.1,
        temperature_decay=0.9,
        hard=False,
        eps=1.0e-6,
    )
    pca = TrainablePCA(
        n_components=3,
        whiten=False,
        init_method="svd",
        eps=1.0e-6,
    )

    logger.info(
        f"Selector targets {selector.n_select}/{selector.input_channels} channels; PCA outputs {pca.n_components} components."
    )

    # Loss nodes (named for logging)
    entropy_loss = SelectorEntropyRegularizer(name="entropy", weight=0.01, target_entropy=None)
    diversity_loss = SelectorDiversityRegularizer(name="diversity", weight=0.01)
    orthogonality_loss = OrthogonalityLoss(name="orthogonality", weight=1.0)

    # Metric nodes (named for logging)
    selector_entropy_metric = SelectorEntropyMetric(name="selector_entropy")
    selector_diversity_metric = SelectorDiversityMetric(name="selector_diversity")
    explained_variance_metric = ExplainedVarianceMetric(name="explained_var")
    component_orthogonality_metric = ComponentOrthogonalityMetric(name="component_orth")
    pca_quality_metrics = PCAQualityMetrics(name="pca_quality")

    # Visualization nodes
    pca_visualization = PCAVisualization(up_to=10)
    viz_rgb = CubeRGBVisualizer(name="rgb", up_to=5)

    # Monitor node
    tensorboard_node = TensorBoardMonitorNode(
        output_dir="./outputs/",
        run_name="PCA_Channel_Selection",
    )

    # Stage 4: Connect graph
    # Processing flow
    canvas.connect(
        (data_node.outputs.cube, normalizer.data),
        (normalizer.normalized, selector.data),
        (selector.selected, pca.data),
        # Loss flow (train)
        (selector.weights, entropy_loss.weights),
        (selector.weights, diversity_loss.weights),
        (pca.components, orthogonality_loss.components),
        # Metric flow (val)
        (selector.weights, selector_entropy_metric.weights),
        (selector.weights, selector_diversity_metric.weights),
        (pca.explained_variance_ratio, explained_variance_metric.explained_variance_ratio),
        (pca.components, component_orthogonality_metric.components),
        (pca.explained_variance_ratio, pca_quality_metrics.explained_variance_ratio),
        (pca.projected, pca_quality_metrics.projected),
        (selector_entropy_metric.metrics, tensorboard_node.metrics),
        (selector_diversity_metric.metrics, tensorboard_node.metrics),
        (explained_variance_metric.metrics, tensorboard_node.metrics),
        (component_orthogonality_metric.metrics, tensorboard_node.metrics),
        (pca_quality_metrics.metrics, tensorboard_node.metrics),
        # Visualization flow (val)
        (pca.projected, pca_visualization.data),
        (pca_visualization.artifacts, tensorboard_node.artifacts),
        (data_node.outputs.cube, viz_rgb.cube),
        (selector.weights, viz_rgb.weights),
        (data_node.outputs.wavelengths, viz_rgb.wavelengths),
        (viz_rgb.artifacts, tensorboard_node.artifacts),
    )

    # Stage 5: Training config & callbacks
    training_cfg = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(
            max_epochs=50,
            accelerator="auto",
            callbacks=CallbacksConfig(
                early_stopping=[
                    # Stop once explained variance plateaus
                    EarlyStoppingConfig(
                        monitor="explained_var/total_explained_variance",
                        mode="max",
                        patience=10,
                        min_delta=0.01,
                    ),
                ],
                model_checkpoint=ModelCheckpointConfig(
                    dirpath="./outputs/02_gradient_training_checkpoints",
                    monitor="explained_var/total_explained_variance",
                    mode="max",
                    verbose=True,
                ),
                learning_rate_monitor=LearningRateMonitorConfig(
                    logging_interval="epoch",
                ),
            ),
        ),
        optimizer=OptimizerConfig(
            name="adam",
            lr=1e-3,
            scheduler=SchedulerConfig(
                name="reduce_on_plateau",
                monitor="explained_var/total_explained_variance",
                mode="max",
                factor=0.5,
                patience=5,
                threshold=0.01,
            ),
        ),
        monitor_plugins=["loguru"],
    )
    logger.debug(f"Trainer configuration: {training_cfg}")

    canvas.visualize(
        format="render_graphviz",
        output_path=f"outputs/canvases/{canvas.name}.png",
        show_execution_stage=True,
    )

    canvas.visualize(
        format="render_mermaid",
        output_path=f"outputs/canvases/{canvas.name}.md",
        direction="LR",
        include_node_class=True,
        wrap_markdown=True,  # default; set False for raw Mermaid text
        show_execution_stage=True,
    )

    # Stage 6: Statistical initialization
    logger.info("Phase 1: statistical initialization (selector + PCA frozen).")
    stat_trainer = StatisticalTrainer(canvas=canvas, datamodule=datamodule)
    stat_trainer.fit()

    # Stage 7: Unfreeze for gradient training
    logger.info("Phase 2: unfreezing selector and PCA for gradient steps.")
    selector.unfreeze()
    pca.unfreeze()

    # Stage 8: Gradient training
    logger.info(
        f"Phase 3: gradient optimization for up to {training_cfg.trainer.max_epochs} epochs."
    )
    grad_trainer = GradientTrainer(
        canvas=canvas,
        datamodule=datamodule,
        loss_nodes=[entropy_loss, diversity_loss, orthogonality_loss],
        metric_nodes=[
            selector_entropy_metric,
            selector_diversity_metric,
            explained_variance_metric,
            component_orthogonality_metric,
            pca_quality_metrics,
        ],
        trainer_config=training_cfg.trainer,
        optimizer_config=training_cfg.optimizer,
        monitors=[tensorboard_node],
    )
    grad_trainer.fit()

    # Stage 9: Validate best checkpoint
    logger.info("Validating best checkpoint...")
    grad_trainer.validate()

    # Stage 10: Test best checkpoint
    logger.info("Testing best checkpoint...")
    grad_trainer.test()

    logger.info("Launch TensorBoard: uv run tensorboard --logdir=./outputs")


if __name__ == "__main__":
    main()
