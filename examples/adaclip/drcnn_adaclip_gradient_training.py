"""Gradient-based DRCNN channel mixer training with AdaClip.

This script demonstrates training a learnable channel mixer (DRCNN-style) to reduce
hyperspectral data (61 channels) to 3 channels optimized for AdaClip anomaly detection.

The pipeline:
  1. HSI cube (61 channels) → LearnableChannelMixer → RGB-like (3 channels)
  2. RGB-like → AdaClip (frozen) → Anomaly scores
  3. Scores → IoU Loss (direct, no thresholding to preserve gradients)
  4. Backpropagation updates only the mixer weights

Based on:
  - Zeegers et al., "Task-Driven Learned Hyperspectral Data Reduction Using
    End-to-End Supervised Deep Learning," J. Imaging 6(12):132, 2020.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import GradientTrainer, StatisticalTrainer
from cuvis_ai_core.training.config import (
    CallbacksConfig,
    ModelCheckpointConfig,
    PipelineMetadata,
    SchedulerConfig,
    TrainingConfig,
    TrainRunConfig,
)

from cuvis_ai.deciders.binary_decider import QuantileBinaryDecider
from cuvis_ai.node.channel_mixer import LearnableChannelMixer
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.drcnn_tensorboard_viz import DRCNNTensorBoardViz
from cuvis_ai.node.losses import IoULoss
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.node.visualizations import AnomalyMask, ScoreHeatmapVisualizer


@hydra.main(config_path="../../configs/", config_name="trainrun/drcnn_adaclip", version_base=None)
def main(cfg: DictConfig) -> None:
    """DRCNN channel mixer + AdaClip training with gradient optimization."""

    logger.info("=== DRCNN Channel Mixer + AdaClip Gradient Training ===")

    # Load AdaCLIP plugin from local development clone
    logger.info("Loading AdaCLIP plugin from local repository...")
    try:
        # Create a NodeRegistry instance for plugin loading
        registry = NodeRegistry()
        registry.load_plugin(
            name="adaclip",
            config={
                "path": r"D:\code-repos\cuvis-ai-adaclip",
                "provides": ["cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector"]
            }
        )
        
        # Get the AdaCLIPDetector class from the registry
        AdaCLIPDetector = NodeRegistry.get("cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector")
        logger.info("✓ AdaCLIP plugin loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load AdaCLIP plugin: {e}")
        logger.error(
            "Make sure the cuvis-ai-adaclip repository is cloned at:\n"
            "  D:\\code-repos\\cuvis-ai-adaclip\n"
            "Or update the path in this script to match your local setup."
        )
        raise

    output_dir = Path(cfg.output_dir)

    # Stage 1: Setup datamodule
    datamodule = SingleCu3sDataModule(**cfg.data)
    datamodule.setup(stage="fit")

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
        raw_mask = first_batch["mask"]
        unique_classes = torch.unique(raw_mask).cpu().tolist()
        logger.info(
            f"  Mask shape: {raw_mask.shape}, "
            f"dtype={raw_mask.dtype}, "
            f"unique_values={unique_classes}"
        )
        # DEBUG: Count pixels per class
        for class_id in unique_classes:
            count = (raw_mask == class_id).sum().item()
            total = raw_mask.numel()
            pct = 100.0 * count / total if total > 0 else 0.0
            class_name = {
                0: "Unlabeled",
                1: "Lentils_black",
                2: "Lentils_brown",
                3: "Stone",
                4: "Background",
            }.get(class_id, f"Class_{class_id}")
            logger.info(f"    Class {class_id} ({class_name}): {count} pixels ({pct:.2f}%)")

    # Stage 2: Build pipeline
    pipeline = CuvisPipeline("DRCNN_AdaClip_Gradient")

    # Data entry node (hardcoded from YAML pipeline.data_node)
    # Use anomaly_class_ids=[3] to only treat Stone (class 3) as anomaly for IoU loss
    data_node = LentilsAnomalyDataNode(
        normal_class_ids=[0, 1],
        anomaly_class_ids=[3],  # Only class 3 (Stone) is treated as anomaly
    )

    # Optional: Normalize HSI data (hardcoded from YAML pipeline.normalizer)
    normalizer = MinMaxNormalizer(
        eps=1e-6,
        use_running_stats=True,
    )

    # DRCNN-style channel mixer: 61 → 3 channels (supports multi-layer reduction)
    # Hardcoded from YAML pipeline.mixer
    mixer = LearnableChannelMixer(
        input_channels=input_channels,
        output_channels=3,  # RGB compatibility
        leaky_relu_negative_slope=0.1,  # Increased from 0.01 to be less aggressive
        use_bias=True,
        use_activation=True,
        normalize_output=True,  # Per-image, per-channel min-max normalization to [0, 1] for AdaClip compatibility
        init_method="pca",  # Options: "xavier", "kaiming", "pca", "zeros"
        eps=1e-6,
        reduction_scheme=[
            61,
            16,
            8,
            3,
        ],  # Multi-layer gradual reduction (matches DRCNN paper style)
        name="channel_mixer",
    )

    # AdaClip detector (FROZEN during training)
    # Hardcoded from YAML pipeline.adaclip
    adaclip = AdaCLIPDetector(
        weight_name="pretrained_all",
        backbone="ViT-L-14-336",
        prompt_text="",  # "normal: lentils, anomaly: stones",
        image_size=518,
        prompting_depth=4,
        prompting_length=5,
        gaussian_sigma=4.0,
        use_half_precision=False,
        enable_warmup=False,
        enable_gradients=True,  # CRITICAL: Allow gradients to flow through (but weights stay frozen)
        name="adaclip",
    )

    # IoU loss (differentiable, works on continuous scores)
    # Hardcoded from YAML pipeline.loss
    iou_loss = IoULoss(
        weight=1.0,
        smooth=1e-6,
        normalize_method="minmax",  # Changed from "sigmoid" to "minmax" to preserve dynamic range
        name="iou_loss",
    )

    # Decider for inference/visualization (NOT used in training path)
    # Hardcoded from YAML pipeline.decider
    decider = QuantileBinaryDecider(
        quantile=0.995,
        name="decider",
    )

    # Metrics
    metrics_node = AnomalyDetectionMetrics(name="metrics_anomaly")

    # Visualizations
    # Hardcoded from YAML pipeline.viz
    viz_mask = AnomalyMask(
        name="mask",
        channel=30,
        up_to=5,
    )
    score_viz = ScoreHeatmapVisualizer(
        name="score_heatmap",
        normalize_scores=True,
        up_to=5,
    )

    # TensorBoard visualization for DRCNN-AdaClip pipeline
    # This creates image artifacts for: HSI input, mixer output (AdaClip input), masks, scores
    drcnn_tb_viz = DRCNNTensorBoardViz(
        hsi_channels=[0, 20, 40],  # Channels for false-color RGB visualization
        max_samples=4,  # Log up to 4 samples per batch
        log_every_n_batches=1,  # Log every batch (set to higher value to reduce TensorBoard size)
        name="drcnn_tensorboard_viz",
    )

    # Monitoring
    tensorboard_node = TensorBoardMonitorNode(
        output_dir=str(
            output_dir / "tensorboard"
        ),  # Changed to output_dir/tensorboard for better organization
        run_name=pipeline.name,
    )

    # Stage 3: Connect the pipeline
    pipeline.connect(
        # Data flow: HSI → Normalizer → Mixer → AdaClip
        (data_node.outputs.cube, normalizer.data),
        (normalizer.normalized, mixer.data),
        (mixer.rgb, adaclip.rgb_image),
        # Loss flow: AdaClip scores → IoU Loss (direct, no decider!)
        (adaclip.scores, iou_loss.predictions),
        (data_node.outputs.mask, iou_loss.targets),
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
        # DRCNN TensorBoard visualization
        (data_node.outputs.cube, drcnn_tb_viz.hsi_cube),
        (mixer.rgb, drcnn_tb_viz.mixer_output),
        (data_node.outputs.mask, drcnn_tb_viz.ground_truth_mask),
        (adaclip.scores, drcnn_tb_viz.adaclip_scores),
        # Monitoring
        (metrics_node.metrics, tensorboard_node.metrics),
        (score_viz.artifacts, tensorboard_node.artifacts),
        (viz_mask.artifacts, tensorboard_node.artifacts),
        (drcnn_tb_viz.artifacts, tensorboard_node.artifacts),  # Add DRCNN-specific visualizations
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

    # Stage 4: Configure training
    training_cfg = TrainingConfig.from_dict(OmegaConf.to_container(cfg.training, resolve=True))  # type: ignore[arg-type]

    # Programmatically add extra callbacks if needed
    if training_cfg.trainer.callbacks is None:
        training_cfg.trainer.callbacks = CallbacksConfig()

    # Early stopping removed per user request

    # Configure checkpointing
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

    # Stage 5: Statistical initialization (optional, if using PCA init)
    if mixer.requires_initial_fit:
        logger.info("Phase 1: Statistical initialization of channel mixer (PCA)...")
        stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
        stat_trainer.fit()
    else:
        logger.info("Phase 1: Skipping statistical initialization (using weight init)")

    # Stage 6: Unfreeze mixer for gradient training
    logger.info("Phase 2: Unfreezing channel mixer for gradient training...")
    unfreeze_node_names = list(cfg.unfreeze_nodes) if "unfreeze_nodes" in cfg else [mixer.name]
    pipeline.unfreeze_nodes_by_name(unfreeze_node_names)
    logger.info(f"Unfrozen nodes: {unfreeze_node_names}")
    logger.info("AdaClip remains frozen (enable_gradients=True allows gradient flow)")

    # DEBUG: Check mixer parameters
    logger.info("=== DEBUG: Mixer Parameters (Before Training) ===")
    initial_weights = {}
    for name, param in mixer.named_parameters():
        initial_weights[name] = param.data.clone()
        logger.info(
            f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}, "
            f"min={param.min().item():.4f}, max={param.max().item():.4f}, "
            f"mean={param.mean().item():.4f}, std={param.std().item():.4f}"
        )

    # Stage 7: Gradient training
    logger.info("Phase 3: Gradient training with IoU loss...")
    grad_trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=datamodule,
        loss_nodes=[iou_loss],
        metric_nodes=[metrics_node],
        trainer_config=training_cfg.trainer,
        optimizer_config=training_cfg.optimizer,
        monitors=[tensorboard_node],
    )
    grad_trainer.fit()

    # DEBUG: Check if weights actually changed
    logger.info("=== DEBUG: Mixer Parameters (After Training) ===")
    for name, param in mixer.named_parameters():
        if name in initial_weights:
            weight_diff = (param.data - initial_weights[name]).abs()
            logger.info(
                f"  {name}: min={param.min().item():.4f}, max={param.max().item():.4f}, "
                f"mean={param.mean().item():.4f}, std={param.std().item():.4f}"
            )
            logger.info(
                f"    Weight change: max_diff={weight_diff.max().item():.6f}, "
                f"mean_diff={weight_diff.mean().item():.6f}"
            )

    logger.info("Running validation evaluation with last checkpoint...")
    val_results = grad_trainer.validate(ckpt_path="last")
    logger.info(f"Validation results: {val_results}")

    # Identify metric and loss nodes
    loss_node_names = [iou_loss.name]
    metric_node_names = [metrics_node.name]
    logger.info(f"Loss nodes: {loss_node_names}")
    logger.info(f"Metric nodes: {metric_node_names}")

    # Stage 8: Evaluate on test set
    logger.info("Running test evaluation with last checkpoint...")
    test_results = grad_trainer.test(ckpt_path="last")
    logger.info(f"Test results: {test_results}")

    # Stage 9: Save trained pipeline and experiment config
    results_dir = output_dir / "trained_models"
    results_dir.mkdir(parents=True, exist_ok=True)

    pipeline_output_path = results_dir / f"{pipeline.name}.yaml"
    logger.info(f"Saving trained pipeline to: {pipeline_output_path}")

    pipeline.save_to_file(
        str(pipeline_output_path),
        metadata=PipelineMetadata(
            name=pipeline.name,
            description=(
                f"Trained DRCNN channel mixer + AdaClip model from {pipeline.name} trainrun. "
                f"Mixer reduces {input_channels} channels to 3 for AdaClip compatibility."
            ),
            tags=["gradient", "drcnn", "adaclip", "channel_mixer", "anomaly_detection"],
            author="cuvis.ai",
        ),
    )
    logger.info(f"  Created: {pipeline_output_path}")
    logger.info(f"  Weights: {pipeline_output_path.with_suffix('.pt')}")

    # Create and save complete trainrun config for reproducibility
    pipeline_config = pipeline.serialize()

    # Convert cfg.data to a serializable dict (fixes YAML serialization issue with lists)
    data_dict = OmegaConf.to_container(cfg.data, resolve=True)  # type: ignore[arg-type]

    trainrun_config = TrainRunConfig(
        name=cfg.name,
        pipeline=pipeline_config,
        data=data_dict,  # Use converted dict instead of OmegaConf object
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

    # Stage 10: Report results
    logger.info("=== Training Complete ===")
    logger.info(f"Trained pipeline saved: {pipeline_output_path}")
    logger.info(f"TrainRun config saved: {trainrun_output_path}")
    logger.info(f"TensorBoard logs: {tensorboard_node.output_dir}")
    logger.info("To restore this trainrun:")
    logger.info(
        f"  uv run python examples/serialization/restore_trainrun.py --trainrun-path {trainrun_output_path}"
    )
    logger.info(f"View TensorBoard: uv run tensorboard --logdir={output_dir / 'tensorboard'}")
    logger.info("TensorBoard will show:")
    logger.info("  - HSI input images (false-color RGB)")
    logger.info("  - Mixer output (what AdaClip sees as input)")
    logger.info("  - Ground truth masks")
    logger.info("  - AdaClip anomaly score heatmaps")


if __name__ == "__main__":
    main()
