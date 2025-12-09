"""Supervised CIR AdaCLIP example (windowed mRMR over NIR/Red/Green bands).

This script mirrors the style of:
  - examples/lad_statistical_training.py
  - examples/deep_svdd_gradient_training.py
  - examples/adaclip/statistical_baseline.py

It:
  * Builds a CuvisCanvas explicitly.
  * Uses LentilsAnomalyDataNode → SupervisedCIRBandSelector → AdaCLIPDetector.
  * Runs a statistical fit phase to learn supervised band scores (Fisher + AUC + MI).
  * Adds a quantile-based decider, generic anomaly metrics, and visualizations.
  * Logs everything via TensorBoardMonitorNode.

Override Hydra params like data_root, train_ids, test_ids, or model_name, e.g.:

    uv run python examples/adaclip/statistical_supervised_cir.py \\
        data_root=../data/Lentils \\
        model_name=ViT-L-14-336 \\
        weight_name=pretrained_all
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from cuvis_ai.anomaly.adaclip import download_weights, list_available_weights
from cuvis_ai.data.lentils_anomaly import SingleCu3sDataModule
from cuvis_ai.deciders.binary_decider import QuantileBinaryDecider
from cuvis_ai.node import (
    AdaCLIPDetector,
    SupervisedCIRBandSelector,
)
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.visualizations import AnomalyMask, ScoreHeatmapVisualizer
from cuvis_ai.pipeline.pipeline import CuvisPipeline
from cuvis_ai.training import StatisticalTrainer

# Ensure project root on sys.path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "Lentils"
DEFAULT_TRAIN_IDS = [0, 2]
DEFAULT_VAL_IDS: list[int] = []
DEFAULT_TEST_IDS = [1, 3, 5]
DEFAULT_MODEL_NAME = "ViT-L-14-336"
DEFAULT_WEIGHT_NAME = "pretrained_all"
DEFAULT_PROMPT = "normal: lentils, anomaly: stones"
DEFAULT_EXPERIMENT_NAME = "statistical_adaclip_supervised_cir"
DEFAULT_MONITOR_ROOT = PROJECT_ROOT / "outputs" / "tensorboard"
DEFAULT_QUANTILE = 0.995
DEFAULT_MASK_CHANNEL = 0
DEFAULT_SUPERVISED_SCORE_WEIGHTS = (1.0, 1.0, 1.0)
DEFAULT_SUPERVISED_LAMBDA = 0.5
DEFAULT_SUPERVISED_CIR_WINDOWS = ((840.0, 910.0), (650.0, 720.0), (500.0, 570.0))
AVAILABLE_BACKBONES = [
    "ViT-L-14-336",
    "ViT-L-14",
    "ViT-B-16",
    "ViT-B-32",
    "ViT-H-14",
]


def _expand_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser()


@hydra.main(version_base=None, config_path="../../cuvis_ai/conf", config_name="general")
def main(cfg: DictConfig) -> None:
    # ----------------------------
    # Resolve configuration
    # ----------------------------
    data_root = _expand_path(getattr(cfg, "data_root", DEFAULT_DATA_ROOT))
    train_ids = list(getattr(cfg, "train_ids", DEFAULT_TRAIN_IDS))
    val_ids = list(getattr(cfg, "val_ids", DEFAULT_VAL_IDS))
    test_ids = list(getattr(cfg, "test_ids", DEFAULT_TEST_IDS))

    model_name = str(getattr(cfg, "model_name", DEFAULT_MODEL_NAME))
    weight_name = str(getattr(cfg, "weight_name", DEFAULT_WEIGHT_NAME))
    prompt_text = str(getattr(cfg, "prompt", DEFAULT_PROMPT))
    experiment_name = str(getattr(cfg, "experiment_name", DEFAULT_EXPERIMENT_NAME))
    monitor_root = _expand_path(getattr(cfg, "monitor_root", DEFAULT_MONITOR_ROOT))

    quantile = float(getattr(cfg, "quantile", DEFAULT_QUANTILE))
    mask_channel = int(getattr(cfg, "mask_viz_channel", DEFAULT_MASK_CHANNEL))
    gaussian_sigma = float(getattr(cfg, "gaussian_sigma", 4.0))
    target_class_id = int(getattr(cfg, "target_class_id", 3))

    # Supervised CIR windows and scoring config
    windows = tuple(getattr(cfg, "supervised_cir_windows", DEFAULT_SUPERVISED_CIR_WINDOWS))
    score_weights = tuple(
        getattr(cfg, "supervised_score_weights", DEFAULT_SUPERVISED_SCORE_WEIGHTS)
    )
    lambda_penalty = float(getattr(cfg, "supervised_lambda_penalty", DEFAULT_SUPERVISED_LAMBDA))

    if model_name not in AVAILABLE_BACKBONES:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {', '.join(AVAILABLE_BACKBONES)}"
        )

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_root}. Set data_root or download the Lentils dataset."
        )

    logger.info("=== AdaCLIP supervised CIR example ===")
    logger.info("Data root: {}", data_root)
    logger.info("Splits: train={}, val={}, test={}", train_ids, val_ids, test_ids)
    logger.info("Model: {} | Weights: {}", model_name, weight_name)
    logger.info("Prompt: {}", prompt_text)
    logger.info("Target anomaly class_id: {}", target_class_id)
    logger.info("Supervised CIR windows: {}", windows)
    logger.info("Supervised score weights: {}", score_weights)
    logger.info("Supervised lambda_penalty: {}", lambda_penalty)

    # ----------------------------
    # Data & weights
    # ----------------------------
    datamodule = SingleCu3sDataModule(
        data_dir=str(data_root),
        dataset_name="Lentils",
        batch_size=3,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        processing_mode="Reflectance",
        normalize_to_unit=False,
    )
    datamodule.setup(stage=None)

    wavelengths = datamodule.train_ds.wavelengths
    logger.info("Wavelength range: {:.1f}-{:.1f} nm", wavelengths.min(), wavelengths.max())

    logger.info("Available AdaCLIP weights: {}", list_available_weights())
    download_weights(weight_name)

    # ----------------------------
    # Build pipeline
    # ----------------------------
    canvas_name = f"{experiment_name}_{model_name}_{Path(weight_name).stem}".replace("-", "_")
    pipeline = CuvisPipeline(canvas_name)

    data_node = LentilsAnomalyDataNode(
        wavelengths=wavelengths,
        normal_class_ids=[0, 1],
    )
    band_selector = SupervisedCIRBandSelector(
        windows=windows,
        score_weights=score_weights,
        lambda_penalty=lambda_penalty,
    )
    adaclip = AdaCLIPDetector(
        weight_name=weight_name,
        backbone=model_name,
        prompt_text=prompt_text,
        gaussian_sigma=gaussian_sigma,
    )

    decider = QuantileBinaryDecider(quantile=quantile)
    standard_metrics = AnomalyDetectionMetrics(name="detection_metrics")
    score_viz = ScoreHeatmapVisualizer(normalize_scores=True, up_to=3)
    mask_viz = AnomalyMask(channel=mask_channel, up_to=3)
    monitor = TensorBoardMonitorNode(
        run_name=canvas_name,
        output_dir=str(monitor_root),
    )

    # Wiring: cube → band selector → AdaCLIP → decider → metrics + viz + TB
    pipeline.connect(
        # hyperspectral → supervised CIR RGB
        (data_node.outputs.cube, band_selector.inputs.cube),
        (data_node.outputs.wavelengths, band_selector.inputs.wavelengths),
        (data_node.outputs.mask, band_selector.inputs.mask),
        # RGB → AdaCLIP
        (band_selector.outputs.rgb_image, adaclip.inputs.rgb_image),
        # AdaCLIP scores → decider + visualizations
        (adaclip.outputs.scores, decider.inputs.logits),
        (adaclip.outputs.scores, score_viz.inputs.scores),
        (adaclip.outputs.scores, mask_viz.inputs.scores),
        # decisions + GT for metrics + overlay
        (decider.outputs.decisions, standard_metrics.inputs.decisions),
        (data_node.outputs.mask, standard_metrics.inputs.targets),
        (decider.outputs.decisions, mask_viz.inputs.decisions),
        (data_node.outputs.mask, mask_viz.inputs.mask),
        (data_node.outputs.cube, mask_viz.inputs.cube),
        # send metrics + artifacts to TensorBoard
        (standard_metrics.outputs.metrics, monitor.inputs.metrics),
        (score_viz.outputs.artifacts, monitor.inputs.artifacts),
        (mask_viz.outputs.artifacts, monitor.inputs.artifacts),
    )

    # ----------------------------
    # Visualize and run
    # ----------------------------
    pipeline.visualize(
        format="render_graphviz",
        output_path=f"outputs/canvases/{pipeline.name}.png",
        show_execution_stage=True,
    )

    trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)

    # Supervised band selector requires an initial statistical fit
    if getattr(band_selector, "requires_initial_fit", False):
        logger.info("Running statistical fit for supervised CIR band selector...")
        trainer.fit()

    if val_ids:
        logger.info("Running validation...")
        trainer.validate()
    else:
        logger.info("Skipping validation (no val_ids provided)")

    logger.info("Running test...")
    trainer.test()

    logger.info(f"Done. View TensorBoard with:\n  uv run tensorboard --logdir={monitor_root}")


if __name__ == "__main__":
    main()
