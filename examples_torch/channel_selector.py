"""Soft Channel Selector for Anomaly Detection

This example demonstrates gradient-based channel selection for RX anomaly detection.
The goal is to select optimal channels that achieve 100% anomaly detection on training
data, using only 3 channels instead of all available channels.

Features:
- Soft channel selection with temperature annealing
- RX (Reed-Xiaoli) batch anomaly detector
- Named loss nodes with automatic individual logging
- Early stopping on specific losses or metrics (e.g., train/bce, train_iou)
- ReduceLROnPlateau scheduler that reduces learning rate based on IoU metrics
- False-color RGB visualization with wavelength annotations
- IoU-based evaluation metrics

Why Channel Selector + RX Anomaly Detection?
---------------------------------------------
Channel selection (61→3 channels) provides physics-informed dimensionality reduction
for anomaly detection. By learning which wavelengths are most discriminative for
separating normal lentils from anomalies (stones, other lentil types), the selector
enables: (1) computational efficiency through dramatic dimensionality reduction,
(2) interpretability by identifying key spectral bands, (3) trainable preprocessing
that adapts to the specific anomaly detection task, (4) regularization through
entropy and diversity losses that encourage sparse, diverse channel selection, and
(5) end-to-end optimization where channel selection and RX detection are jointly trained.

Run with:
    python examples_torch/channel_selector.py

Graph Visualization:

```mermaid
graph LR
    Data[LentilsAnomalyDataNode] --> |cube| MinMax[MinMaxNormalizer]
    MinMax --> |normalized| Selector[SoftChannelSelector<br/>61→3 channels]
    Selector --> |selected| RX[RXGlobal]
    RX --> |scores| LogitHead[RXLogitHead]

    LogitHead --> |logits| BCE[AnomalyBCEWithLogits]
    Selector --> |weights| Entropy[SelectorEntropyRegularizer]
    Selector --> |weights| Diversity[SelectorDiversityRegularizer]

    LogitHead --> |logits| Decider[BinaryDecider]
    Decider --> |decisions| Metrics[AnomalyDetectionMetrics]
    Data --> |mask| Metrics
    Data --> |mask| BCE

    Decider --> |decisions| MaskViz[AnomalyMask]
    Data --> |cube| MaskViz
    Data --> |mask| MaskViz

    Data --> |cube| RGBViz[CubeRGBVisualizer]
    Selector --> |weights| RGBViz
    Data --> |wavelengths| RGBViz

    Metrics --> |metrics| TB[TensorBoardMonitor]
    MaskViz --> |artifacts| TB
    RGBViz --> |artifacts| TB
```

Pipeline Flow:
Data → MinMax → Selector(61→3) ─┬→ RX → LogitHead ─┬→ BCE Loss
                                 │                   ├→ Decider → Metrics → TensorBoard
                                 │                   └→ Visualizations → TensorBoard
                                 └→ Entropy & Diversity Losses

Training Phases:
----------------
1. Statistical Initialization: Initialize normalizer and selector (variance-based) with frozen weights
2. Unfreeze: Enable gradient flow through selector, RX detector, and logit head
3. Gradient Training: Optimize with:
   - Losses: bce_loss (primary), entropy_loss, diversity_loss (regularizers)
   - Metrics: IoU, precision, recall, F1 score for anomaly detection
   - Callbacks: early_stopping (BCE & IoU), model_checkpoint, lr_monitor, reduce_lr_on_plateau

Usage:
    python examples_torch/channel_selector.py
"""

import hydra
from loguru import logger
from omegaconf import DictConfig

from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.anomaly.rx_logit_head import RXLogitHead
from cuvis_ai.data.lentils_anomaly import LentilsAnomaly
from cuvis_ai.deciders.binary_decider import BinaryDecider
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.losses import (
    AnomalyBCEWithLogits,
    SelectorDiversityRegularizer,
    SelectorEntropyRegularizer,
)
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.node.selector import SoftChannelSelector
from cuvis_ai.node.visualizations import AnomalyMask, CubeRGBVisualizer
from cuvis_ai.pipeline.canvas import CuvisCanvas
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


@hydra.main(version_base=None, config_path="../cuvis_ai/conf", config_name="general")
def main(cfg: DictConfig) -> None:
    """Main training function with channel selection optimization."""

    logger.info("=== Channel Selector for Anomaly Detection ===")

    # Stage 1: Setup datamodule
    datamodule = LentilsAnomaly(
        data_dir="../data/Lentils",
        batch_size=4,
        train_ids=[0, 1, 2],
        val_ids=[3, 4, 5],
        test_ids=[9, 10, 11, 12, 13],
    )
    datamodule.setup(stage="fit")

    wavelengths = datamodule.train_ds.wavelengths

    logger.info(f"Dataset wavelengths: {wavelengths.shape}")
    logger.info(f"Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")

    # Stage 2: Build graph
    canvas = CuvisCanvas("Channel_Selector")

    data_node = LentilsAnomalyDataNode(
        wavelengths=wavelengths,
        normal_class_ids=[
            0,
            1,
        ],  # {0: 'Unlabeled', 1: 'Lentils_black', 2: 'Lentils_brown', 3: 'Stone', 4: 'Background'}
    )
    normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
    selector = SoftChannelSelector(
        n_select=3,
        input_channels=61,
        init_method="variance",
        temperature_init=5.0,
        temperature_min=0.1,
        temperature_decay=0.9,
        hard=False,
        eps=1.0e-6,
    )
    rx = RXGlobal(eps=1.0e-6)
    logit_head = RXLogitHead(init_scale=1.0, init_bias=0.0)
    # zscore_norm = ZScoreNormalizer(dims=[1, 2], eps=1.0e-6, keepdim=True)

    # Single threshold for binary decisions
    decider = BinaryDecider(threshold=0.5)

    bce_loss = AnomalyBCEWithLogits(name="bce", weight=10.0, pos_weight=None)
    entropy_loss = SelectorEntropyRegularizer(name="entropy", weight=0.1, target_entropy=None)
    diversity_loss = SelectorDiversityRegularizer(name="diversity", weight=0.01)

    metrics_anomaly = AnomalyDetectionMetrics(name="metrics_anomaly")

    viz_mask = AnomalyMask(name="mask", channel=30, up_to=5)
    viz_rgb = CubeRGBVisualizer(name="rgb", up_to=5)

    tensorboard_node = TensorBoardMonitorNode(
        output_dir="./outputs/",
        run_name="Channel_Selector",
    )

    # Stage 3: Connect the Nodes
    canvas.connect(
        # Processing flow
        (data_node.outputs.cube, normalizer.data),
        (normalizer.normalized, selector.data),
        (selector.selected, rx.data),
        # Loss flow
        (rx.scores, logit_head.scores),
        (logit_head.logits, bce_loss.predictions),
        (data_node.outputs.mask, bce_loss.targets),
        (selector.weights, entropy_loss.weights),
        (selector.weights, diversity_loss.weights),
        # Decision flow - single threshold for consistency
        (logit_head.logits, decider.logits),
        # Metric flow - using binary decisions
        (decider.decisions, metrics_anomaly.decisions),
        (data_node.outputs.mask, metrics_anomaly.targets),
        (metrics_anomaly.metrics, tensorboard_node.metrics),
        # Visualization flow - using binary decisions
        (decider.decisions, viz_mask.decisions),
        (data_node.outputs.mask, viz_mask.mask),
        (data_node.outputs.cube, viz_mask.cube),
        (data_node.outputs.cube, viz_rgb.cube),
        (selector.weights, viz_rgb.weights),
        (data_node.outputs.wavelengths, viz_rgb.wavelengths),
        (viz_mask.artifacts, tensorboard_node.artifacts),
        (viz_rgb.artifacts, tensorboard_node.artifacts),
    )

    # Stage 4: Training configuration with callbacks
    training_cfg = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(
            max_epochs=100,
            accelerator="auto",
            callbacks=CallbacksConfig(
                early_stopping=[
                    # Early stopping on primary BCE loss
                    EarlyStoppingConfig(
                        monitor="train/bce",
                        mode="min",
                    ),
                    # Early stopping on IoU metric (stop when reaching high performance)
                    EarlyStoppingConfig(
                        monitor="metrics_anomaly/iou",
                        min_delta=0.01,
                    ),
                ],
                model_checkpoint=ModelCheckpointConfig(
                    dirpath="./outputs/03_channel_selector_checkpoints",
                    monitor="metrics_anomaly/iou",
                    verbose=True,
                ),
                learning_rate_monitor=LearningRateMonitorConfig(
                    logging_interval="epoch",
                ),
            ),
        ),
        optimizer=OptimizerConfig(
            lr=0.001,
            scheduler=SchedulerConfig(
                name="reduce_on_plateau",
                monitor="metrics_anomaly/iou",
                mode="max",
                factor=0.5,
                patience=5,
                threshold=0.01,
            ),
        ),
    )

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

    # Stage 5: Statistical initialization
    logger.info("Phase 1: Statistical initialization...")
    stat_trainer = StatisticalTrainer(canvas=canvas, datamodule=datamodule)
    stat_trainer.fit()

    # Stage 6: Unfreeze selector, RX, and logits head
    logger.info("Phase 2: Unfreezing selector and RX for gradient training...")
    selector.unfreeze()
    rx.unfreeze()
    logit_head.unfreeze()

    # Stage 7: Gradient training with callbacks (now configured in TrainerConfig)
    logger.info("Phase 3: Gradient-based channel selection optimization...")
    grad_trainer = GradientTrainer(
        canvas=canvas,
        datamodule=datamodule,
        loss_nodes=[bce_loss],  # , entropy_loss, diversity_loss],
        metric_nodes=[metrics_anomaly],
        trainer_config=training_cfg.trainer,
        optimizer_config=training_cfg.optimizer,
        monitors=[tensorboard_node],
    )
    grad_trainer.fit()

    logger.info("Running validation evaluation with best checkpoint...")
    val_results = grad_trainer.validate()
    logger.info(f"Validation results: {val_results}")

    # Stage 8: Evaluate on test set with best checkpoint
    logger.info("Running test evaluation with best checkpoint...")
    test_results = grad_trainer.test()
    logger.info(f"Test results: {test_results}")

    # Stage 9: Report results
    logger.info("=== Training Complete ===")
    # logger.info(f"Selected channels: {selector.get_top_k_channels().tolist()}")
    logger.info("Checkpoints saved to: ./outputs/03_channel_selector_checkpoints")
    logger.info(f"TensorBoard logs: {tensorboard_node.output_dir}")
    logger.info("View logs: uv run tensorboard --logdir=./outputs/03_channel_selector_runs")


if __name__ == "__main__":
    main()
