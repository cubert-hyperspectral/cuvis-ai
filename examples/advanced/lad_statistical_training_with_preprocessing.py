"""Statistical Initialization - LAD with Bandpass and Per-Pixel Unit Norm Preprocessing.

This example extends LAD statistical training by adding preprocessing nodes:
- BandpassByWavelength: Filters channels by wavelength range
- PerPixelUnitNorm: Normalizes each pixel's spectrum to unit L2 norm

The pipeline: Data → MinMax → Bandpass → PerPixelUnitNorm → LAD → Metrics/Viz
"""

from pathlib import Path
from typing import Any

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig
from torch import Tensor

from cuvis_ai.anomaly.lad_detector import LADGlobal
from cuvis_ai.data.lentils_anomaly import SingleCu3sDataModule
from cuvis_ai.deciders.binary_decider import QuantileBinaryDecider
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.node import Node
from cuvis_ai.node.normalization import MinMaxNormalizer, PerPixelUnitNorm
from cuvis_ai.node.preprocessors import BandpassByWavelength
from cuvis_ai.node.visualizations import AnomalyMask, ScoreHeatmapVisualizer
from cuvis_ai.pipeline.pipeline import CuvisPipeline
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.training import StatisticalTrainer
from cuvis_ai.training.config import (
    ExperimentConfig,
    PipelineMetadata,
    TrainingConfig,
)
from cuvis_ai.utils.types import Context, ExecutionStage, Metric


class SampleCustomMetrics(Node):
    """Compute anomaly pixel statistics from binary decisions."""

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary anomaly decisions [B, H, W, 1]",
        ),
    }

    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects")}

    def __init__(self, **kwargs) -> None:
        super().__init__(execution_stages={ExecutionStage.VAL, ExecutionStage.TEST}, **kwargs)

    def forward(self, decisions: Tensor, context: Context) -> dict[str, Any]:
        total_pixels = decisions.numel()
        anomalous_pixels = int(decisions.sum().item())
        anomaly_percentage = (anomalous_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

        metrics = [
            Metric(
                name="anomaly/total_pixels",
                value=float(total_pixels),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="anomaly/anomalous_pixels",
                value=float(anomalous_pixels),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="anomaly/anomaly_percentage",
                value=anomaly_percentage,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
        ]

        return {"metrics": metrics}


@hydra.main(config_path="../../configs/", config_name="experiment/default", version_base=None)
def main(cfg: DictConfig) -> None:
    """LAD Statistical with Preprocessing and experiment config saving."""

    logger.info("=== LAD with Bandpass + Per-Pixel Unit Norm Preprocessing ===")

    output_dir = Path(cfg.output_dir)

    # Stage 1: Setup datamodule
    datamodule = SingleCu3sDataModule(**cfg.data)
    datamodule.setup(stage="fit")

    wavelengths = datamodule.train_ds.wavelengths
    logger.info(f"Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")

    # Stage 2: Build graph
    pipeline = CuvisPipeline("LAD_Bandpass_UnitNorm")

    data_node = LentilsAnomalyDataNode(
        normal_class_ids=[0, 1],
        # {0: 'Unlabeled', 1: 'Lentils_black', 2: 'Lentils_brown', 3: 'Stone', 4: 'Background'}
    )

    # Preprocessing nodes
    normalizer_node = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
    bandpass_node = BandpassByWavelength(
        min_wavelength_nm=700.0,
        max_wavelength_nm=None,
        wavelengths=wavelengths,
    )
    unit_norm_node = PerPixelUnitNorm(eps=1.0e-8)

    lad_node = LADGlobal(
        num_channels=20, eps=1.0e-8, normalize_laplacian=True, use_numpy_laplacian=True
    )
    decider_node = QuantileBinaryDecider(quantile=0.99)
    metrics_anomaly = AnomalyDetectionMetrics(name="metrics_anomaly")
    sample_metrics = SampleCustomMetrics(name="sample_metrics")
    viz_mask = AnomalyMask(name="mask", channel=30, up_to=5)
    score_viz = ScoreHeatmapVisualizer(name="score_heatmap", normalize_scores=True, up_to=5)

    tensorboard_node = TensorBoardMonitorNode(
        output_dir=str(output_dir / ".." / "tensorboard"),
        run_name=pipeline.name,
    )

    # Stage 3: Connect graph

    pipeline.connect(
        # Processing flow with preprocessing
        (data_node.outputs.cube, normalizer_node.data),
        (normalizer_node.normalized, bandpass_node.data),
        (data_node.outputs.wavelengths, bandpass_node.wavelengths),
        (bandpass_node.filtered, unit_norm_node.data),
        (unit_norm_node.normalized, lad_node.data),
        # Decision flow
        (lad_node.scores, decider_node.logits),
        # Metric flow
        (decider_node.decisions, metrics_anomaly.decisions),
        (data_node.outputs.mask, metrics_anomaly.targets),
        (lad_node.scores, metrics_anomaly.logits),
        (metrics_anomaly.metrics, tensorboard_node.metrics),
        # Sample custom metrics flow
        (decider_node.decisions, sample_metrics.decisions),
        (sample_metrics.metrics, tensorboard_node.metrics),
        # Visualization flow
        (lad_node.scores, score_viz.scores),
        (score_viz.artifacts, tensorboard_node.artifacts),
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
        wrap_markdown=True,
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
            description=f"Trained model from {pipeline.name} experiment (statistical training with preprocessing)",
            tags=["statistical", "lad", "preprocessing", "bandpass", "unit_norm"],
            author="cuvis.ai",
        ),
    )
    logger.info(f"  Created: {pipeline_output_path}")
    logger.info(f"  Weights: {pipeline_output_path.with_suffix('.pt')}")

    # Create and save complete experiment config for reproducibility
    pipeline_config = pipeline.serialize()

    experiment_config = ExperimentConfig(
        name=cfg.name,
        pipeline=pipeline_config,
        data=cfg.data,
        training=TrainingConfig(seed=42),
        output_dir=str(output_dir),
        loss_nodes=loss_node_names,
        metric_nodes=metric_node_names,
        freeze_nodes=[],
        unfreeze_nodes=[],
    )

    experiment_output_path = results_dir / f"{cfg.name}_experiment.yaml"
    logger.info(f"Saving experiment config to: {experiment_output_path}")
    experiment_config.save_to_file(str(experiment_output_path))

    # Stage 8: Report results
    logger.info("=== Training Complete ===")
    logger.info(f"Trained pipeline saved: {pipeline_output_path}")
    logger.info(f"Experiment config saved: {experiment_output_path}")
    logger.info(f"TensorBoard logs: {tensorboard_node.output_dir}")
    logger.info("To restore this experiment:")
    logger.info(
        f"  uv run python examples/serialization/restore_experiment.py --experiment-path {experiment_output_path}"
    )
    logger.info(f"View logs: uv run tensorboard --logdir={output_dir}")


if __name__ == "__main__":
    main()
