"""Gradient-based DeepSVDD training demo using the port-based pipeline."""

from __future__ import annotations

from pathlib import Path

import hydra
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import GradientTrainer, StatisticalTrainer
from cuvis_ai_schemas.pipeline import PipelineMetadata
from cuvis_ai_schemas.training import (
    CallbacksConfig,
    EarlyStoppingConfig,
    ModelCheckpointConfig,
    SchedulerConfig,
    TrainingConfig,
    TrainRunConfig,
)
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cuvis_ai.anomaly.deep_svdd import (
    DeepSVDDCenterTracker,
    DeepSVDDProjection,
    DeepSVDDScores,
    ZScoreNormalizerGlobal,
)
from cuvis_ai.deciders.binary_decider import QuantileBinaryDecider
from cuvis_ai.node.anomaly_visualization import AnomalyMask, ScoreHeatmapVisualizer
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.losses import DeepSVDDSoftBoundaryLoss
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.normalization import PerPixelUnitNorm
from cuvis_ai.node.preprocessors import BandpassByWavelength
from cuvis_ai.utils.deep_svdd_factory import infer_channels_after_bandpass


@hydra.main(config_path="../../configs/", config_name="trainrun/deep_svdd", version_base=None)
def main(cfg: DictConfig) -> None:
    """Deep SVDD anomaly detection with gradient training and trainrun config saving."""

    logger.info("=== Deep SVDD Gradient Training ===")

    output_dir = Path(cfg.output_dir)

    # Stage 1: Setup datamodule
    datamodule = SingleCu3sDataModule(**cfg.data)

    datamodule.setup(stage="fit")

    # Access node parameters from cfg.pipeline if available, otherwise from cfg (for backward compatibility)
    pipeline_cfg = cfg.pipeline if hasattr(cfg, "pipeline") and cfg.pipeline else cfg

    # Infer post-bandpass channel count if not provided
    if pipeline_cfg.encoder.num_channels is None or pipeline_cfg.projection.in_channels is None:
        ch_cfg = infer_channels_after_bandpass(datamodule, pipeline_cfg.bandpass)
        if pipeline_cfg.encoder.num_channels is None:
            pipeline_cfg.encoder.num_channels = ch_cfg.num_channels
        if pipeline_cfg.projection.in_channels is None:
            pipeline_cfg.projection.in_channels = ch_cfg.in_channels
    logger.info(
        f"Wavelengths: min {datamodule.train_ds.wavelengths_nm.min()} nm, max {datamodule.train_ds.wavelengths_nm.max()} nm"
    )

    # Stage 2: Build graph
    pipeline = CuvisPipeline("DeepSVDD_Gradient")

    data_node = LentilsAnomalyDataNode(normal_class_ids=[0, 1])

    bandpass_node = BandpassByWavelength(
        min_wavelength_nm=pipeline_cfg.bandpass.min_wavelength_nm,
        max_wavelength_nm=pipeline_cfg.bandpass.max_wavelength_nm,
    )

    unit_norm_node = PerPixelUnitNorm(eps=1e-8)

    encoder = ZScoreNormalizerGlobal(
        num_channels=pipeline_cfg.encoder.num_channels,
        hidden=pipeline_cfg.encoder.hidden,
    )
    projection = DeepSVDDProjection(
        in_channels=pipeline_cfg.projection.in_channels,
        rep_dim=pipeline_cfg.projection.rep_dim,
        hidden=pipeline_cfg.projection.hidden,
    )
    center_tracker = DeepSVDDCenterTracker(
        rep_dim=pipeline_cfg.center_tracker.rep_dim, alpha=pipeline_cfg.center_tracker.alpha
    )
    loss_node = DeepSVDDSoftBoundaryLoss(name="deepsvdd_loss", nu=pipeline_cfg.loss.nu)
    score_node = DeepSVDDScores()

    decider_node = QuantileBinaryDecider(quantile=pipeline_cfg.decider.quantile)
    metrics_node = AnomalyDetectionMetrics(name="metrics_anomaly")
    viz_mask = AnomalyMask(
        name="mask",
        channel=pipeline_cfg.viz.mask_channel,
        up_to=pipeline_cfg.viz.up_to,
    )
    score_viz = ScoreHeatmapVisualizer(
        name="score_heatmap",
        normalize_scores=True,
        up_to=pipeline_cfg.viz.up_to,
    )

    tensorboard_node = TensorBoardMonitorNode(
        output_dir=str(output_dir / ".." / "tensorboard"),
        run_name=pipeline.name,
    )

    # Stage 3: Connect the Nodes
    pipeline.connect(
        # Preprocessing chain
        (data_node.outputs.cube, bandpass_node.data),
        (data_node.outputs.wavelengths, bandpass_node.wavelengths),
        (bandpass_node.filtered, unit_norm_node.data),
        (unit_norm_node.normalized, encoder.data),
        # Encoder outputs routed to loss + score computation
        (encoder.normalized, projection.data),
        (projection.embeddings, center_tracker.embeddings),
        (projection.embeddings, loss_node.embeddings),
        (projection.embeddings, score_node.embeddings),
        (center_tracker.center, loss_node.center),
        (center_tracker.center, score_node.center),
        # Scores → decision + metrics + visualizations
        (score_node.scores, decider_node.logits),
        (score_node.scores, metrics_node.logits),
        (score_node.scores, score_viz.scores),
        (score_node.scores, viz_mask.scores),
        # Mask + decision inputs for metrics/viz
        (decider_node.decisions, metrics_node.decisions),
        (data_node.outputs.mask, metrics_node.targets),
        (decider_node.decisions, viz_mask.decisions),
        (data_node.outputs.mask, viz_mask.mask),
        (data_node.outputs.cube, viz_mask.cube),
        # Monitoring
        (metrics_node.metrics, tensorboard_node.metrics),
        (center_tracker.metrics, tensorboard_node.metrics),
        (score_viz.artifacts, tensorboard_node.artifacts),
        (viz_mask.artifacts, tensorboard_node.artifacts),
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
        wrap_markdown=True,
        show_execution_stage=True,
    )

    training_cfg = TrainingConfig.from_dict(OmegaConf.to_container(cfg.training, resolve=True))  # type: ignore[arg-type]

    # Programmatically add extra callbacks if needed
    if training_cfg.trainer.callbacks is None:
        training_cfg.trainer.callbacks = CallbacksConfig()

    # Configure early stopping and checkpointing
    training_cfg.trainer.callbacks.early_stopping.append(
        EarlyStoppingConfig(monitor="train/deepsvdd_loss", mode="min", patience=15)
    )

    training_cfg.trainer.callbacks.checkpoint = ModelCheckpointConfig(
        dirpath=str(output_dir / "checkpoints"),
        monitor="metrics_anomaly/iou",
        mode="max",
        save_top_k=3,
        save_last=True,
        filename="{epoch:02d}",
        verbose=True,
    )

    # Configure learning rate scheduler
    if training_cfg.scheduler is None:
        training_cfg.scheduler = SchedulerConfig(
            name="reduce_on_plateau",
            monitor="metrics_anomaly/iou",
            mode="max",
            factor=0.5,
            patience=5,
        )

    # Stage 4: Statistical initialization
    logger.info("Phase 1: Statistical fit of DeepSVDD encoder...")
    stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    stat_trainer.fit()

    # Stage 5: Unfreeze encoder for gradient optimization
    logger.info("Phase 2: Unfreezing encoder for gradient training...")
    unfreeze_node_names = list(cfg.unfreeze_nodes) if "unfreeze_nodes" in cfg else [encoder.name]
    pipeline.unfreeze_nodes_by_name(unfreeze_node_names)
    logger.info(f"Unfrozen nodes: {unfreeze_node_names}")

    # Stage 6: Gradient training with DeepSVDDSoftBoundaryLoss
    logger.info("Phase 3: Gradient training with DeepSVDDSoftBoundaryLoss...")
    grad_trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=datamodule,
        loss_nodes=[loss_node],
        metric_nodes=[metrics_node],
        trainer_config=training_cfg.trainer,
        optimizer_config=training_cfg.optimizer,
        monitors=[tensorboard_node],
    )
    grad_trainer.fit()

    logger.info("Running validation evaluation with last checkpoint...")
    val_results = grad_trainer.validate(ckpt_path="last")
    logger.info(f"Validation results: {val_results}")

    # Identify metric and loss nodes from pipeline
    loss_node_names = [loss_node.name]
    metric_node_names = [metrics_node.name]
    logger.info(f"Loss nodes: {loss_node_names}")
    logger.info(f"Metric nodes: {metric_node_names}")

    # Stage 7: Evaluate on test set with best checkpoint
    logger.info("Running test evaluation with last checkpoint...")
    test_results = grad_trainer.test(ckpt_path="last")
    logger.info(f"Test results: {test_results}")

    # Stage 8: Save trained pipeline and experiment config
    results_dir = output_dir / "trained_models"

    pipeline_output_path = results_dir / f"{pipeline.name}.yaml"
    logger.info(f"Saving trained pipeline to: {pipeline_output_path}")

    pipeline.save_to_file(
        str(pipeline_output_path),
        metadata=PipelineMetadata(
            name=pipeline.name,
            description=f"Trained model from {pipeline.name} trainrun (statistical + gradient training)",
            tags=["gradient", "statistical", "deep_svdd", "anomaly_detection"],
            author="cuvis.ai",
        ),
    )
    logger.info(f"  Created: {pipeline_output_path}")
    logger.info(f"  Weights: {pipeline_output_path.with_suffix('.pt')}")

    # Create and save complete trainrun config for reproducibility
    pipeline_config = pipeline.serialize()

    trainrun_config = TrainRunConfig(
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

    trainrun_output_path = results_dir / f"{cfg.name}_trainrun.yaml"
    logger.info(f"Saving trainrun config to: {trainrun_output_path}")
    trainrun_config.save_to_file(str(trainrun_output_path))

    # Stage 9: Report results
    logger.info("=== Training Complete ===")
    logger.info(f"Trained pipeline saved: {pipeline_output_path}")
    logger.info(f"TrainRun config saved: {trainrun_output_path}")
    logger.info(f"TensorBoard logs: {tensorboard_node.output_dir}")
    logger.info("To restore this trainrun:")
    logger.info(
        f"  uv run python examples/serialization/restore_trainrun.py --trainrun-path {trainrun_output_path}"
    )
    logger.info(f"View logs: uv run tensorboard --logdir={output_dir}")


if __name__ == "__main__":
    main()
