from pathlib import Path

import hydra
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import StatisticalTrainer
from cuvis_ai_schemas.pipeline import PipelineMetadata
from cuvis_ai_schemas.training import (
    TrainingConfig,
    TrainRunConfig,
)
from loguru import logger
from omegaconf import DictConfig

from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.deciders.binary_decider import BinaryDecider
from cuvis_ai.node.anomaly_visualization import AnomalyMask
from cuvis_ai.node.conversion import ScoreToLogit
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.metrics import AnomalyDetectionMetrics, AnomalyPixelStatisticsMetric
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.normalization import MinMaxNormalizer


@hydra.main(
    config_path="../configs/", config_name="trainrun/default_statistical", version_base=None
)
def main(cfg: DictConfig) -> None:
    """Statistical RX Anomaly Detection with trainrun config saving."""

    logger.info("=== Statistical RX Anomaly Detection ===")

    output_dir = Path(cfg.output_dir)

    # Stage 1: Setup datamodule
    datamodule = SingleCu3sDataModule(**cfg.data)
    datamodule.setup(stage="fit")

    # Stage 2: Build graph
    pipeline = CuvisPipeline("RX_Statistical")

    data_node = LentilsAnomalyDataNode(
        normal_class_ids=[0, 1],
        # {0: 'Unlabeled', 1: 'Lentils_black', 2: 'Lentils_brown', 3: 'Stone', 4: 'Background'}
    )
    normalizer_node = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
    rx_node = RXGlobal(num_channels=61, eps=1.0e-6)
    logit_head = ScoreToLogit(init_scale=1.0, init_bias=0.0)
    decider_node = BinaryDecider(threshold=0.5)
    metrics_anomaly = AnomalyDetectionMetrics(name="metrics_anomaly")
    sample_metrics = AnomalyPixelStatisticsMetric(name="sample_metrics")
    viz_mask = AnomalyMask(name="mask", channel=30, up_to=5)

    tensorboard_node = TensorBoardMonitorNode(
        output_dir=str(output_dir / ".." / "tensorboard"),
        run_name=pipeline.name,
    )

    # Stage 3: Connect graph
    pipeline.connect(
        # Processing flow
        (data_node.outputs.cube, normalizer_node.data),
        (normalizer_node.normalized, rx_node.data),
        (rx_node.scores, logit_head.scores),
        (logit_head.logits, decider_node.logits),
        # Metric flow
        (decider_node.decisions, metrics_anomaly.decisions),
        (data_node.outputs.mask, metrics_anomaly.targets),
        (metrics_anomaly.metrics, tensorboard_node.metrics),
        # Sample custom metrics flow
        (decider_node.decisions, sample_metrics.decisions),
        (sample_metrics.metrics, tensorboard_node.metrics),
        # Visualization flow
        (decider_node.decisions, viz_mask.decisions),
        (data_node.outputs.mask, viz_mask.mask),
        (data_node.outputs.cube, viz_mask.cube),
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
        wrap_markdown=True,  # default; set False for raw Mermaid text
        show_execution_stage=True,
    )

    # Stage 4: Statistical initialization
    stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    stat_trainer.fit()

    # Identify metric and loss nodes from pipeline
    loss_nodes_list = [
        node
        for node in pipeline.nodes()
        if hasattr(node, "__class__") and "Loss" in node.__class__.__name__
    ]
    metric_nodes_list = [
        node
        for node in pipeline.nodes()
        if hasattr(node, "__class__") and "Metric" in node.__class__.__name__
    ]

    loss_node_names = [node.name for node in loss_nodes_list]
    metric_node_names = [node.name for node in metric_nodes_list]
    logger.info(f"Metric nodes: {metric_node_names}")
    logger.info(f"Loss nodes: {loss_node_names}")

    # Stage 5: Run validation
    logger.info("Running validation evaluation...")
    stat_trainer.validate()

    # Stage 6: Run test
    logger.info("Running test evaluation...")
    stat_trainer.test()

    # Stage 7: Save trained pipeline and experiment config
    results_dir = output_dir / "trained_models"

    pipeline_output_path = results_dir / f"{pipeline.name}.yaml"
    logger.info(f"Saving trained pipeline to: {pipeline_output_path}")

    pipeline.save_to_file(
        str(pipeline_output_path),
        metadata=PipelineMetadata(
            name=pipeline.name,
            description=f"Trained model from {pipeline.name} experiment (statistical training)",
            tags=["statistical", "rx"],
            author="cuvis.ai",
        ),
    )
    logger.info(f"  Created: {pipeline_output_path}")
    logger.info(f"  Weights: {pipeline_output_path.with_suffix('.pt')}")

    # Create and save complete trainrun config for reproducibility
    # Get pipeline config from serialize (contains full structure inline)
    pipeline_config = pipeline.serialize()

    trainrun_config = TrainRunConfig(
        name=cfg.name,
        pipeline=pipeline_config,
        data=cfg.data,
        training=TrainingConfig(seed=42),
        output_dir=str(output_dir),
        loss_nodes=loss_node_names,
        metric_nodes=metric_node_names,
        freeze_nodes=[],  # All nodes remain frozen in statistical training
        unfreeze_nodes=[],  # No nodes unfrozen in statistical training
    )

    trainrun_output_path = results_dir / f"{cfg.name}_trainrun.yaml"
    logger.info(f"Saving trainrun config to: {trainrun_output_path}")
    trainrun_config.save_to_file(str(trainrun_output_path))

    # Stage 8: Report results
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
