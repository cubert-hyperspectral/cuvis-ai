from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.anomaly.rx_logit_head import RXLogitHead
from cuvis_ai.data.lentils_anomaly import SingleCu3sDataModule
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
from cuvis_ai.pipeline.pipeline import CuvisPipeline
from cuvis_ai.training import GradientTrainer, StatisticalTrainer
from cuvis_ai.training.config import (
    CallbacksConfig,
    EarlyStoppingConfig,
    ExperimentConfig,
    ModelCheckpointConfig,
    PipelineMetadata,
    SchedulerConfig,
    TrainingConfig,
)


@hydra.main(config_path="../configs/", config_name="experiment/default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Channel Selector with gradient training and experiment config saving."""

    logger.info("=== Channel Selector for Anomaly Detection ===")

    output_dir = Path(cfg.output_dir)

    # Stage 1: Setup datamodule
    datamodule = SingleCu3sDataModule(**cfg.data)
    datamodule.setup(stage="fit")

    # Stage 2: Build graph
    pipeline = CuvisPipeline("Channel_Selector")

    data_node = LentilsAnomalyDataNode(
        normal_class_ids=[0, 1],
        # {0: 'Unlabeled', 1: 'Lentils_black', 2: 'Lentils_brown', 3: 'Stone', 4: 'Background'}
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
    rx = RXGlobal(num_channels=15, eps=1.0e-6)
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
        output_dir=str(output_dir / ".." / "tensorboard"),
        run_name=pipeline.name,
    )

    # Stage 3: Connect the Nodes
    pipeline.connect(
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

    pipeline.visualize(
        format="render_graphviz",
        output_path=str(output_dir / "pipeline" / f"{pipeline.name}.png"),
        show_execution_stage=True,
    )

    pipeline.visualize(
        format="render_mermaid",
        output_path=str(output_dir / "pipeline" / f"{pipeline.name}.md"),
        direction="LR",
        include_node_class=True,
        wrap_markdown=True,  # default; set False for raw Mermaid text
        show_execution_stage=True,
    )

    training_cfg = TrainingConfig.from_dict(OmegaConf.to_container(cfg.training, resolve=True))

    # programmatically add extra callbacks
    if training_cfg.trainer.callbacks is None:
        training_cfg.trainer.callbacks = CallbacksConfig()

    training_cfg.trainer.callbacks.early_stopping.append(
        EarlyStoppingConfig(monitor="train/bce", mode="min", patience=20)
    )

    training_cfg.optimizer.scheduler = SchedulerConfig(
        name="reduce_on_plateau",
        monitor="metrics_anomaly/iou",
        mode="max",
        factor=0.5,
        patience=5,
    )

    training_cfg.trainer.callbacks.model_checkpoint = ModelCheckpointConfig(
        dirpath=str(output_dir / "checkpoints"),
        monitor="metrics_anomaly/iou",
        mode="max",
        save_top_k=3,
        save_last=True,
        filename="{epoch:02d}",
        verbose=True,
    )

    # Stage 5: Statistical initialization
    logger.info("Phase 1: Statistical initialization...")
    stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    stat_trainer.fit()

    # Stage 6: Unfreeze selector, RX, and logits head
    logger.info("Phase 2: Unfreezing selector and RX for gradient training...")
    unfreeze_node_names = [selector.name, rx.name, logit_head.name]
    pipeline.unfreeze_nodes_by_name(unfreeze_node_names)
    logger.info(f"Unfrozen nodes: {unfreeze_node_names}")

    # Stage 7: Gradient training with callbacks (now configured in TrainerConfig)
    logger.info("Phase 3: Gradient-based channel selection optimization...")
    grad_trainer = GradientTrainer(
        pipeline=pipeline,
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

    # Identify metric and loss nodes from pipeline
    loss_node_names = [bce_loss.name]
    metric_node_names = [metrics_anomaly.name]
    logger.info(f"Loss nodes: {loss_node_names}")
    logger.info(f"Metric nodes: {metric_node_names}")

    # Stage 8: Evaluate on test set with best checkpoint
    logger.info("Running test evaluation with best checkpoint...")
    test_results = grad_trainer.test()
    logger.info(f"Test results: {test_results}")

    # Stage 9: Save trained pipeline and experiment config
    results_dir = output_dir / "trained_models"

    pipeline_output_path = results_dir / f"{pipeline.name}.yaml"
    logger.info(f"Saving trained pipeline to: {pipeline_output_path}")

    pipeline.save_to_file(
        str(pipeline_output_path),
        metadata=PipelineMetadata(
            name=pipeline.name,
            description=f"Trained model from {pipeline.name} experiment (statistical + gradient training)",
            tags=["gradient", "statistical", "channel_selector", "rx"],
            author="cuvis.ai",
        ),
    )
    logger.info(f"  Created: {pipeline_output_path}")
    logger.info(f"  Weights: {pipeline_output_path.with_suffix('.pt')}")

    # Create and save complete experiment config for reproducibility
    # Get pipeline config from serialize (contains full structure inline)
    pipeline_config = pipeline.serialize()

    experiment_config = ExperimentConfig(
        name=cfg.name,
        pipeline=pipeline_config,
        data=cfg.data,
        training=training_cfg,
        output_dir=str(output_dir),
        loss_nodes=loss_node_names,
        metric_nodes=metric_node_names,
        freeze_nodes=[],  # All other nodes remain frozen
        unfreeze_nodes=unfreeze_node_names,
    )

    experiment_output_path = results_dir / f"{cfg.name}_experiment.yaml"
    logger.info(f"Saving experiment config to: {experiment_output_path}")
    experiment_config.save_to_file(str(experiment_output_path))

    # Stage 10: Report results
    logger.info("=== Training Complete ===")
    logger.info(f"Trained pipeline saved: {pipeline_output_path}")
    logger.info(f"Experiment config saved: {experiment_output_path}")
    logger.info(f"TensorBoard logs: {tensorboard_node.output_dir}")
    logger.info("To restore this experiment:")
    logger.info(
        f"  uv run python examples/serlization/restore_experiment.py --experiment-path {experiment_output_path}"
    )
    logger.info(f"View logs: uv run tensorboard --logdir={output_dir}")


if __name__ == "__main__":
    main()
