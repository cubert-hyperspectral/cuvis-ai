"""Gradient-based DeepSVDD training demo using the port-based pipeline."""

from __future__ import annotations

import hydra
from loguru import logger
from omegaconf import DictConfig

from cuvis_ai.anomaly.deep_svdd import DeepSVDDCenterTracker, DeepSVDDEncoder, DeepSVDDScores
from cuvis_ai.data.lentils_anomaly import LentilsAnomaly
from cuvis_ai.deciders.binary_decider import QuantileBinaryDecider
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.losses import DeepSVDDSoftBoundaryLoss
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.normalization import PerPixelUnitNorm
from cuvis_ai.node.preprocessors import BandpassByWavelength
from cuvis_ai.node.visualizations import AnomalyMask, ScoreHeatmapVisualizer
from cuvis_ai.pipeline.canvas import CuvisCanvas
from cuvis_ai.training import GradientTrainer, StatisticalTrainer
from cuvis_ai.training.config import OptimizerConfig, TrainerConfig


@hydra.main(version_base=None, config_path="../cuvis_ai/conf", config_name="general")
def main(cfg: DictConfig) -> None:
    dataset_mode = str(getattr(cfg, "mode", "Raw"))
    normalize_to_unit = bool(getattr(cfg, "normalize_to_unit", False))
    min_wavelength_nm = float(getattr(cfg, "min_wavelength_nm", 700.0))
    max_wavelength_nm = getattr(cfg, "max_wavelength_nm", None)
    if max_wavelength_nm is not None:
        max_wavelength_nm = float(max_wavelength_nm)

    logger.info("=== DeepSVDD gradient training demo ===")
    logger.info("Dataset mode: %s | normalize_to_unit=%s", dataset_mode, normalize_to_unit)
    max_wl_str = f"{max_wavelength_nm:.1f}" if max_wavelength_nm is not None else "inf"
    logger.info("Bandpass range: %.1f - %s nm", min_wavelength_nm, max_wl_str)

    datamodule = LentilsAnomaly(
        data_dir="C:/Users/anish.raj/projects/gitlab_cuvis_ai_3/cuvis.ai/data/Lentils",
        batch_size=2,
        train_ids=[0],
        val_ids=[3, 4, 6],
        test_ids=[1, 5],
        mode=dataset_mode,
        normalize_to_unit=normalize_to_unit,
    )
    datamodule.setup(stage="fit")

    wavelengths = datamodule.train_ds.wavelengths
    canvas = CuvisCanvas(f"DeepSVDD_Gradient_{dataset_mode}")

    data_node = LentilsAnomalyDataNode(wavelengths=wavelengths, normal_class_ids=[0, 1])
    bandpass_node = BandpassByWavelength(
        min_wavelength_nm=min_wavelength_nm,
        max_wavelength_nm=max_wavelength_nm,
        wavelengths=wavelengths,
    )
    unit_norm_node = PerPixelUnitNorm(eps=1e-8)

    encoder = DeepSVDDEncoder(rep_dim=32, hidden=128, sample_n=200_000, seed=0)
    center_tracker = DeepSVDDCenterTracker(alpha=0.1)
    loss_node = DeepSVDDSoftBoundaryLoss(name="deepsvdd_loss", nu=0.05)
    score_node = DeepSVDDScores()

    decider_node = QuantileBinaryDecider(quantile=0.995)
    metrics_node = AnomalyDetectionMetrics(name="metrics_anomaly")
    viz_mask = AnomalyMask(channel=30, up_to=5)
    score_viz = ScoreHeatmapVisualizer(normalize_scores=True, up_to=5)

    tensorboard_node = TensorBoardMonitorNode(
        run_name=f"deep_svdd_gradient_{dataset_mode.lower()}",
        output_dir="./outputs/",
    )

    canvas.connect(
        # Preprocessing chain
        (data_node.outputs.cube, bandpass_node.data),
        (data_node.outputs.wavelengths, bandpass_node.wavelengths),
        (bandpass_node.filtered, unit_norm_node.data),
        (unit_norm_node.normalized, encoder.data),
        # Encoder outputs routed to loss + score computation
        (encoder.embeddings, center_tracker.embeddings),
        (encoder.embeddings, loss_node.embeddings),
        (encoder.embeddings, score_node.embeddings),
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

    logger.info("Visualizing canvas...")
    canvas.visualize(
        format="render_graphviz",
        output_path=f"outputs/canvases/{canvas.name}.png",
        show_execution_stage=True,
    )

    logger.info("Phase 1: statistical fit of DeepSVDD encoder")
    stat_trainer = StatisticalTrainer(canvas=canvas, datamodule=datamodule)
    stat_trainer.fit()

    logger.info("Phase 2: unfreezing encoder for gradient optimization")
    encoder.unfreeze()

    trainer_config = TrainerConfig(max_epochs=70, accelerator="auto", enable_progress_bar=True)
    optimizer_config = OptimizerConfig(name="adam", lr=1e-3)

    logger.info("Phase 3: gradient training with DeepSVDDSoftBoundaryLoss")
    grad_trainer = GradientTrainer(
        canvas=canvas,
        datamodule=datamodule,
        loss_nodes=[loss_node],
        metric_nodes=[metrics_node],
        trainer_config=trainer_config,
        optimizer_config=optimizer_config,
        monitors=[tensorboard_node],
    )
    grad_trainer.fit()

    logger.info("Validating trained DeepSVDD model...")
    grad_trainer.validate(ckpt_path=None)

    logger.info("Testing trained DeepSVDD model...")
    grad_trainer.test(ckpt_path=None)

    logger.info(
        "Finished! Launch TensorBoard with: uv run tensorboard --logdir=./outputs/deep_svdd_gradient_%s",
        dataset_mode.lower(),
    )


if __name__ == "__main__":
    main()

