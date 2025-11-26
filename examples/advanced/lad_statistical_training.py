"""Statistical Initialization Only - LAD Baseline (Reflectance/Raw toggle).

This example mirrors `examples_torch/statistical_training.py`, but swaps the RX detector
for the newly ported LAD detector. The rest of the pipeline—normalization, metrics,
visualizations, TensorBoard logging—remains unchanged so we can compare behavior apples-to-apples.

Use the Hydra override `mode=Reflectance` or `mode=Raw` to switch data modes:

    uv run python examples_torch/lad_statistical_training.py mode=Raw
"""

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
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.node.visualizations import AnomalyMask, ScoreHeatmapVisualizer
from cuvis_ai.pipeline.canvas import CuvisCanvas
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.training import StatisticalTrainer
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

    def serialize(self, serial_dir: str) -> dict:
        return {**self.hparams}

    def load(self, params: dict, serial_dir: str) -> None:
        pass


@hydra.main(version_base=None, config_path="../cuvis_ai/conf", config_name="general")
def main(cfg: DictConfig) -> None:
    dataset_mode = str(getattr(cfg, "mode", "Raw"))
    normalize_to_unit = bool(getattr(cfg, "normalize_to_unit", False))
    data_dir = str(getattr(cfg.data, "data_dir", "../data/Lentils"))
    logger.info("=== Statistical Initialization for LAD Anomaly Detection ===")
    logger.info(f"Dataset mode: {dataset_mode} | normalize_to_unit={normalize_to_unit}")

    datamodule = SingleCu3sDataModule(
        data_dir=data_dir,
        dataset_name="Lentils",
        batch_size=4,
        train_ids=[0],
        val_ids=[3, 4, 6],
        test_ids=[1, 5],
        processing_mode=dataset_mode,
    )
    datamodule.setup(stage="fit")

    wavelengths = datamodule.train_ds.wavelengths
    logger.info(f"Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")

    canvas = CuvisCanvas(f"LAD_Statistical_{dataset_mode}")

    data_node = LentilsAnomalyDataNode(
        wavelengths=wavelengths,
        normal_class_ids=[0, 1],
    )
    normalizer_node = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
    lad_node = LADGlobal(eps=1.0e-8, normalize_laplacian=True, use_numpy_laplacian=True)
    decider_node = QuantileBinaryDecider(quantile=0.995)
    metrics_anomaly = AnomalyDetectionMetrics(name="metrics_anomaly")
    sample_metrics = SampleCustomMetrics(name="sample_metrics")
    viz_mask = AnomalyMask(channel=30, up_to=5)
    score_viz = ScoreHeatmapVisualizer(normalize_scores=True, up_to=5)
    tensorboard_node = TensorBoardMonitorNode(
        run_name=f"lad_statistical_quantile_{dataset_mode.lower()}",
        output_dir="./outputs/",
    )

    # Core processing: optionally normalize to [0, 1] before LAD
    if normalize_to_unit:
        canvas.connect(
            (data_node.outputs.cube, normalizer_node.data),
            (normalizer_node.normalized, lad_node.data),
        )
    else:
        canvas.connect(
            (data_node.outputs.cube, lad_node.data),
        )

    canvas.connect(
        # LAD scores → quantile threshold
        (lad_node.scores, decider_node.logits),
        # Visualizations consume raw LAD scores (heatmaps) and cubes/masks (overlay)
        (lad_node.scores, score_viz.scores),
        (score_viz.artifacts, tensorboard_node.artifacts),
        # Metrics consume binary decisions + ground-truth mask (no AP when logits absent)
        (decider_node.decisions, metrics_anomaly.decisions),
        (data_node.outputs.mask, metrics_anomaly.targets),
        (metrics_anomaly.metrics, tensorboard_node.metrics),
        # Custom per-batch statistics
        (decider_node.decisions, sample_metrics.decisions),
        (sample_metrics.metrics, tensorboard_node.metrics),
        # Mask visualization overlays (decisions + GT + cube)
        (decider_node.decisions, viz_mask.decisions),
        (data_node.outputs.mask, viz_mask.mask),
        (data_node.outputs.cube, viz_mask.cube),
        (viz_mask.artifacts, tensorboard_node.artifacts),
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
        wrap_markdown=True,
        show_execution_stage=True,
    )

    stat_trainer = StatisticalTrainer(canvas=canvas, datamodule=datamodule)
    stat_trainer.fit()

    logger.info("Running validation...")
    stat_trainer.validate()

    logger.info("Running test...")
    stat_trainer.test()

    logger.info(
        "Done! View logs via: uv run tensorboard --logdir=./outputs/lad_statistical_%s",
        dataset_mode.lower(),
    )


if __name__ == "__main__":
    main()
