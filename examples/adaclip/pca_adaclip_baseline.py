"""PCA baseline for AdaClip anomaly detection.

This script demonstrates using PCA (Principal Component Analysis) as a baseline
for hyperspectral data reduction (61 channels → 3 channels) before AdaClip.

The pipeline:
  1. HSI cube (61 channels) → Normalizer → PCA (3 components) → PCA Normalizer → RGB-like (3 channels)
  2. RGB-like → AdaClip (frozen) → Anomaly scores
  3. Scores → Metrics (for evaluation)
  4. No gradient training - PCA is frozen after statistical initialization

This serves as a baseline to compare against learnable methods (DRCNN mixer, MLBS selector).
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig

from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import StatisticalTrainer
from cuvis_ai_core.training.config import (
    PipelineMetadata,
    TrainingConfig,
    TrainRunConfig,
)

from cuvis_ai.deciders.binary_decider import QuantileBinaryDecider
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.drcnn_tensorboard_viz import DRCNNTensorBoardViz
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.node.visualizations import AnomalyMask, ScoreHeatmapVisualizer


@hydra.main(
    config_path="../../configs/", config_name="trainrun/pca_adaclip_baseline", version_base=None
)
def main(cfg: DictConfig) -> None:
    """PCA baseline + AdaClip evaluation (no gradient training)."""

    logger.info("=== PCA Baseline + AdaClip Evaluation ===")

    # Load AdaCLIP plugin from GitLab repository
    logger.info("Loading AdaCLIP plugin from GitLab repository...")
    try:
        # Create a NodeRegistry instance for plugin loading
        registry = NodeRegistry()
        registry.load_plugin(
            name="adaclip",
            config={
                "repo": "git@gitlab.cubert.local:cubert/cuvis-ai-adaclip.git",
                "ref": "v0.1.0",  # Tagged release for production stability
                "provides": ["cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector"]
            }
        )
        
        # For local development, comment out above and use:
        # registry.load_plugin(
        #     name="adaclip",
        #     config={
        #         "path": r"D:\code-repos\cuvis-ai-adaclip",
        #         "provides": ["cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector"]
        #     }
        # )
        
        # Get the AdaCLIPDetector class from the registry
        AdaCLIPDetector = NodeRegistry.get("cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector")
        logger.info("✓ AdaCLIP plugin loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load AdaCLIP plugin: {e}")
        raise

    output_dir = Path(cfg.output_dir)

    # Stage 1: Setup datamodule
    datamodule = SingleCu3sDataModule(**cfg.data)
    datamodule.setup(stage="fit")

    # Access node parameters from cfg.pipeline if available, otherwise from cfg
    pipeline_cfg = cfg.pipeline if hasattr(cfg, "pipeline") and cfg.pipeline else cfg

    # Infer input channels from dataset
    train_loader = datamodule.train_dataloader()
    first_batch = next(iter(train_loader))
    input_channels = first_batch["cube"].shape[-1]  # [B, H, W, C] -> C

    logger.info(f"Input channels: {input_channels}")
    logger.info(
        f"Wavelengths: min {datamodule.train_ds.wavelengths_nm.min()} nm, "
        f"max {datamodule.train_ds.wavelengths_nm.max()} nm"
    )

    # DEBUG: Print first batch info
    logger.info("=== DEBUG: First Batch Info ===")
    cube_float = (
        first_batch["cube"].float()
        if first_batch["cube"].dtype != torch.float32
        else first_batch["cube"]
    )
    logger.info(
        f"  Cube shape: {first_batch['cube'].shape}, dtype={first_batch['cube'].dtype}, "
        f"min={cube_float.min().item():.4f}, "
        f"max={cube_float.max().item():.4f}"
    )
    if "mask" in first_batch:
        logger.info(
            f"  Mask shape: {first_batch['mask'].shape}, "
            f"dtype={first_batch['mask'].dtype}, "
            f"unique_values={torch.unique(first_batch['mask'])}"
        )

    # Stage 2: Build pipeline
    pipeline = CuvisPipeline("PCA_AdaClip_Baseline")

    # Data entry node
    data_node = LentilsAnomalyDataNode(normal_class_ids=pipeline_cfg.data_node.normal_class_ids)

    # Normalize HSI data (input normalization)
    normalizer = MinMaxNormalizer(
        eps=pipeline_cfg.normalizer.get("eps", 1e-6),
        use_running_stats=pipeline_cfg.normalizer.get("use_running_stats", True),
        name="hsi_normalizer",
    )

    # PCA: 61 → 3 channels (frozen, statistical initialization only)
    pca = TrainablePCA(
        n_components=3,  # 3 components for RGB compatibility
        whiten=False,  # Don't whiten - we'll normalize separately to [0,1]
        init_method="svd",  # Use SVD for initialization
        eps=pipeline_cfg.get("pca", {}).get("eps", 1e-6),
        name="pca_baseline",
    )

    # Normalizer for PCA output (to get [0,1] range for AdaClip)
    # Use per-image, per-channel normalization like the mixer does
    pca_normalizer = MinMaxNormalizer(
        eps=pipeline_cfg.normalizer.get("eps", 1e-6),
        use_running_stats=False,  # Per-image normalization (not running stats)
        name="pca_output_normalizer",
    )

    # AdaClip detector (FROZEN)
    adaclip = AdaCLIPDetector(
        weight_name=pipeline_cfg.adaclip.get("weight_name", "pretrained_all"),
        backbone=pipeline_cfg.adaclip.get("backbone", "ViT-L-14-336"),
        prompt_text=pipeline_cfg.adaclip.get("prompt_text", "normal: lentils, anomaly: stones"),
        image_size=pipeline_cfg.adaclip.get("image_size", 518),
        prompting_depth=pipeline_cfg.adaclip.get("prompting_depth", 4),
        prompting_length=pipeline_cfg.adaclip.get("prompting_length", 5),
        gaussian_sigma=pipeline_cfg.adaclip.get("gaussian_sigma", 4.0),
        use_half_precision=pipeline_cfg.adaclip.get("use_half_precision", True),
        enable_warmup=pipeline_cfg.adaclip.get("enable_warmup", True),
        enable_gradients=False,  # No gradients needed for baseline
        name="adaclip",
    )

    # Decider for inference/visualization
    decider = QuantileBinaryDecider(
        quantile=pipeline_cfg.decider.get("quantile", 0.995),
        name="decider",
    )

    # Metrics
    metrics_node = AnomalyDetectionMetrics(name="metrics_anomaly")

    # Visualizations
    viz_mask = AnomalyMask(
        name="mask",
        channel=pipeline_cfg.viz.get("mask_channel", 30),
        up_to=pipeline_cfg.viz.get("up_to", 5),
    )
    score_viz = ScoreHeatmapVisualizer(
        name="score_heatmap",
        normalize_scores=True,
        up_to=pipeline_cfg.viz.get("up_to", 5),
    )

    # TensorBoard visualization for PCA-AdaClip pipeline
    # This creates image artifacts for: HSI input, PCA output, normalized PCA (AdaClip input), masks, scores
    drcnn_tb_viz = DRCNNTensorBoardViz(
        hsi_channels=[0, 20, 40],  # Channels for false-color RGB visualization
        max_samples=4,  # Log up to 4 samples per batch
        log_every_n_batches=1,  # Log every batch
        name="pca_tensorboard_viz",
    )

    # Monitoring
    tensorboard_node = TensorBoardMonitorNode(
        output_dir=str(output_dir / "tensorboard"),
        run_name=pipeline.name,
    )

    # Stage 3: Connect the pipeline
    pipeline.connect(
        # Data flow: HSI → Normalizer → PCA → PCA Normalizer → AdaClip
        (data_node.outputs.cube, normalizer.data),
        (normalizer.normalized, pca.data),
        (pca.projected, pca_normalizer.data),  # Normalize PCA output to [0,1]
        (pca_normalizer.normalized, adaclip.rgb_image),  # Connect to AdaClip
        # Inference/visualization flow: Scores → Decider → Metrics/Viz
        (adaclip.scores, decider.logits),
        (adaclip.scores, metrics_node.logits),
        (adaclip.scores, score_viz.scores),
        (adaclip.scores, viz_mask.scores),
        # Metrics and visualization inputs
        (decider.decisions, metrics_node.decisions),
        (data_node.outputs.mask, metrics_node.targets),
        (decider.decisions, viz_mask.decisions),
        (data_node.outputs.mask, viz_mask.mask),
        (data_node.outputs.cube, viz_mask.cube),
        # PCA TensorBoard visualization
        # Note: We visualize PCA output (before normalization) and normalized PCA (AdaClip input)
        (data_node.outputs.cube, drcnn_tb_viz.hsi_cube),
        (
            pca_normalizer.normalized,
            drcnn_tb_viz.mixer_output,
        ),  # Reuse same port name for normalized PCA
        (data_node.outputs.mask, drcnn_tb_viz.ground_truth_mask),
        (adaclip.scores, drcnn_tb_viz.adaclip_scores),
        # Monitoring
        (metrics_node.metrics, tensorboard_node.metrics),
        (score_viz.artifacts, tensorboard_node.artifacts),
        (viz_mask.artifacts, tensorboard_node.artifacts),
        (drcnn_tb_viz.artifacts, tensorboard_node.artifacts),
    )

    # Visualize pipeline
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

    # Stage 4: Statistical initialization (required for PCA and normalizer)
    logger.info("Phase 1: Statistical initialization of PCA and normalizer...")
    stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    stat_trainer.fit()

    # DEBUG: Check PCA components and explained variance
    logger.info("=== DEBUG: PCA Components ===")
    with torch.no_grad():
        # Get a sample batch to check PCA output
        sample_batch = next(iter(train_loader))
        sample_cube = sample_batch["cube"].float()

        # Run through normalizer (already initialized by StatisticalTrainer)
        normalizer_output = normalizer.forward(data=sample_cube)

        # Run through PCA
        pca_output = pca.forward(data=normalizer_output["normalized"])

        logger.info(f"  PCA projected shape: {pca_output['projected'].shape}")
        logger.info(
            f"  PCA projected range: min={pca_output['projected'].min().item():.4f}, "
            f"max={pca_output['projected'].max().item():.4f}, "
            f"mean={pca_output['projected'].mean().item():.4f}"
        )

        if "explained_variance_ratio" in pca_output:
            ev_ratio = pca_output["explained_variance_ratio"]
            logger.info(f"  Explained variance ratio: {ev_ratio.tolist()}")
            logger.info(f"  Total explained variance: {ev_ratio.sum().item():.4f}")

        # Run through PCA normalizer
        pca_norm_output = pca_normalizer.forward(data=pca_output["projected"])
        logger.info(
            f"  Normalized PCA (AdaClip input) range: "
            f"min={pca_norm_output['normalized'].min().item():.4f}, "
            f"max={pca_norm_output['normalized'].max().item():.4f}, "
            f"mean={pca_norm_output['normalized'].mean().item():.4f}"
        )

    # Stage 5: No gradient training - PCA is frozen baseline
    logger.info("Phase 2: PCA baseline - no gradient training (frozen)")
    logger.info("  PCA components are fixed after statistical initialization")
    logger.info("  This serves as a baseline to compare against learnable methods")

    # Stage 6: Run evaluation (validation and test)
    # Use StatisticalTrainer for evaluation (no gradient training needed for baseline)
    # This is the same pattern as LAD statistical training examples
    logger.info("Running validation evaluation...")
    stat_trainer.validate()

    logger.info("Running test evaluation...")
    stat_trainer.test()

    # Stage 7: Save pipeline and experiment config
    results_dir = output_dir / "trained_models"
    results_dir.mkdir(parents=True, exist_ok=True)

    pipeline_output_path = results_dir / f"{pipeline.name}.yaml"
    logger.info(f"Saving baseline pipeline to: {pipeline_output_path}")

    # Get explained variance info for metadata
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        sample_cube = sample_batch["cube"].float()
        normalizer_output = normalizer.forward(data=sample_cube)
        pca_output = pca.forward(data=normalizer_output["normalized"])
        ev_ratio = (
            pca_output.get("explained_variance_ratio", torch.zeros(3))
            if "explained_variance_ratio" in pca_output
            else torch.zeros(3)
        )

    pipeline.save_to_file(
        str(pipeline_output_path),
        metadata=PipelineMetadata(
            name=pipeline.name,
            description=(
                f"PCA baseline + AdaClip model from {pipeline.name} trainrun. "
                f"PCA reduces {input_channels} channels to 3 for AdaClip compatibility. "
                f"Explained variance ratio: {ev_ratio.tolist()}"
            ),
            tags=["baseline", "pca", "adaclip", "anomaly_detection", "statistical"],
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
        training=TrainingConfig(seed=42),  # Minimal training config (no actual training)
        output_dir=str(output_dir),
        loss_nodes=[],  # No loss nodes for baseline
        metric_nodes=[metrics_node.name],
        freeze_nodes=[],  # All nodes remain frozen
        unfreeze_nodes=[],  # Nothing to unfreeze
    )

    trainrun_output_path = results_dir / f"{cfg.name}_trainrun.yaml"
    logger.info(f"Saving trainrun config to: {trainrun_output_path}")
    trainrun_config.save_to_file(str(trainrun_output_path))

    # Stage 8: Report results
    logger.info("=== Baseline Evaluation Complete ===")
    logger.info(f"Baseline pipeline saved: {pipeline_output_path}")
    logger.info(f"TrainRun config saved: {trainrun_output_path}")
    logger.info(f"TensorBoard logs: {tensorboard_node.output_dir}")
    logger.info(f"View TensorBoard: uv run tensorboard --logdir={output_dir / 'tensorboard'}")
    logger.info("TensorBoard will show:")
    logger.info("  - HSI input images (false-color RGB)")
    logger.info("  - PCA output (3-channel projection)")
    logger.info("  - Normalized PCA (what AdaClip sees as input)")
    logger.info("  - Ground truth masks")
    logger.info("  - AdaClip anomaly score heatmaps")
    logger.info(f"Explained variance ratio: {ev_ratio.tolist()}")
    logger.info(f"Total explained variance: {ev_ratio.sum().item():.4f}")


if __name__ == "__main__":
    main()
