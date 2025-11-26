"""Statistical Initialization - LAD with Bandpass and Per-Pixel Unit Norm Preprocessing.

This example extends `lad_statistical_training.py` by adding preprocessing nodes:
- BandpassByWavelength: Filters channels by wavelength range
- PerPixelUnitNorm: Normalizes each pixel's spectrum to unit L2 norm

The pipeline: Data → Bandpass → PerPixelUnitNorm → LAD → Metrics/Viz

Use Hydra overrides to configure:
- mode: "Reflectance" or "Raw" (data loading mode)
- normalize_to_unit: true/false (dataset-level normalization)
- min_wavelength_nm: Minimum wavelength for bandpass (default: 500.0)
- max_wavelength_nm: Maximum wavelength for bandpass (default: 900.0, or None for no upper bound)

Example:
    uv run python examples_torch/lad_statistical_training_with_preprocessing.py mode=Raw min_wavelength_nm=600.0 max_wavelength_nm=800.0
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
from cuvis_ai.node.normalization import MinMaxNormalizer, PerPixelUnitNorm
from cuvis_ai.node.preprocessors import BandpassByWavelength
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
    min_wavelength_nm = float(getattr(cfg, "min_wavelength_nm", 700.0))
    max_wavelength_nm = getattr(cfg, "max_wavelength_nm", None)
    if max_wavelength_nm is not None:
        max_wavelength_nm = float(max_wavelength_nm)

    logger.info("=== LAD with Bandpass + Per-Pixel Unit Norm Preprocessing ===")
    logger.info(f"Dataset mode: {dataset_mode} | normalize_to_unit={normalize_to_unit}")
    max_wl_str = f"{max_wavelength_nm:.1f}" if max_wavelength_nm is not None else "inf"
    logger.info(f"Bandpass range: {min_wavelength_nm:.1f} - {max_wl_str} nm")

    data_dir = str(getattr(cfg.data, "data_dir", "../data/Lentils"))
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
    logger.info(f"Full wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")

    canvas = CuvisCanvas(f"LAD_Bandpass_UnitNorm_{dataset_mode}")

    # Data entry node
    data_node = LentilsAnomalyDataNode(
        wavelengths=wavelengths,
        normal_class_ids=[0, 1],
    )

    # Preprocessing nodes
    unit_range_node = MinMaxNormalizer(eps=1.0e-6, use_running_stats=False)
    bandpass_node = BandpassByWavelength(
        min_wavelength_nm=min_wavelength_nm,
        max_wavelength_nm=max_wavelength_nm,
        wavelengths=wavelengths,  # Cache wavelengths for filtering
    )
    unit_norm_node = PerPixelUnitNorm(eps=1e-8)

    # Detection node
    lad_node = LADGlobal(eps=1.0e-8, normalize_laplacian=True, use_numpy_laplacian=True)

    # Decision and metrics nodes
    decider_node = QuantileBinaryDecider(quantile=0.99)
    metrics_anomaly = AnomalyDetectionMetrics(name="metrics_anomaly")

    # Visualization nodes
    viz_mask = AnomalyMask(channel=30, up_to=5)
    score_viz = ScoreHeatmapVisualizer(normalize_scores=True, up_to=5)

    # Monitoring node
    tensorboard_node = TensorBoardMonitorNode(
        run_name=f"lad_bandpass_unitnorm_{dataset_mode.lower()}",
        output_dir="./outputs/",
    )

    # Connect the pipeline:
    # Data → (optional MinMax) → Bandpass → PerPixelUnitNorm → LAD → Decision/Metrics/Viz
    if normalize_to_unit:
        canvas.connect(
            (data_node.outputs.cube, unit_range_node.data),
            (unit_range_node.normalized, bandpass_node.data),
        )
    else:
        canvas.connect(
            (data_node.outputs.cube, bandpass_node.data),
        )

    canvas.connect(
        # Preprocessing chain after bandpass
        (data_node.outputs.wavelengths, bandpass_node.wavelengths),
        (bandpass_node.filtered, unit_norm_node.data),
        # Detection chain
        (unit_norm_node.normalized, lad_node.data),
        (lad_node.scores, decider_node.logits),
        # Metrics
        (decider_node.decisions, metrics_anomaly.decisions),
        (data_node.outputs.mask, metrics_anomaly.targets),
        (lad_node.scores, metrics_anomaly.logits),
        (metrics_anomaly.metrics, tensorboard_node.metrics),
        # Visualizations
        (lad_node.scores, score_viz.scores),
        (score_viz.artifacts, tensorboard_node.artifacts),
        (decider_node.decisions, viz_mask.decisions),
        (data_node.outputs.mask, viz_mask.mask),
        (data_node.outputs.cube, viz_mask.cube),
        (lad_node.scores, viz_mask.scores),
        (viz_mask.artifacts, tensorboard_node.artifacts),
    )

    # Visualize the canvas
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

    # Train statistical nodes (LADGlobal)
    stat_trainer = StatisticalTrainer(canvas=canvas, datamodule=datamodule)
    stat_trainer.fit()

    logger.info("Running validation...")
    stat_trainer.validate()

    logger.info("Running test...")
    stat_trainer.test()

    logger.info(
        "Done! View logs via: uv run tensorboard --logdir=./outputs/lad_bandpass_unitnorm_%s",
        dataset_mode.lower(),
    )


if __name__ == "__main__":
    main()
