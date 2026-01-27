"""Gradient-based Concrete band selector training with AdaClip.

This script mirrors the DRCNN and MLBS AdaClip setups but uses a
Concrete/Gumbel-Softmax band selector to learn 3 discrete bands that
are optimal for AdaClip-based anomaly detection.

Pipeline:
  1. HSI cube (61 channels) → ConcreteBandSelector → RGB-like (3 channels)
  2. RGB-like → AdaClip (frozen) → Anomaly scores
  3. Scores → IoU or BCE loss (+ distinctness loss on selector weights)
  4. Backpropagation updates only the selector logits
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import GradientTrainer
from cuvis_ai_core.training.config import (
    CallbacksConfig,
    ModelCheckpointConfig,
    PipelineMetadata,
    SchedulerConfig,
    TrainingConfig,
    TrainRunConfig,
)
from cuvis_ai_core.utils.node_registry import NodeRegistry
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cuvis_ai.deciders.two_stage_decider import TwoStageBinaryDecider
from cuvis_ai.node.concrete_selector import ConcreteBandSelector
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.drcnn_tensorboard_viz import DRCNNTensorBoardViz
from cuvis_ai.node.losses import AnomalyBCEWithLogits, DistinctnessLoss, IoULoss
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.node.visualizations import AnomalyMask, ScoreHeatmapVisualizer


@hydra.main(
    config_path="../../configs/",
    config_name="trainrun/concrete_adaclip",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Concrete band selector + AdaClip training with gradient optimization."""

    logger.info("=== Concrete Band Selector + AdaClip Gradient Training ===")

    # Load AdaCLIP plugin from GitLab repository
    logger.info("Loading AdaCLIP plugin from GitLab repository...")

    # Create a NodeRegistry instance for plugin loading
    registry = NodeRegistry()
    registry.load_plugin(
        name="adaclip",
        config={
            "repo": "git@github.com:cubert-hyperspectral/cuvis-ai-adaclip.git",
            "ref": "v0.1.0",  # Tagged release for production stability
            "provides": ["cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector"],
        },
    )

    # For local development, comment out above and use:
    # registry.load_plugin(
    #     name="adaclip",
    #     config={
    #         "path": r"D:\code-repos\cuvis-ai-adaclip",
    #         "provides": ["cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector"]
    #     }
    # )

    # Get the AdaCLIPDetector class from the registry (class method works for both built-ins and plugins)
    AdaCLIPDetector = NodeRegistry.get("cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector")
    logger.info("✓ AdaCLIP plugin loaded successfully")

    output_dir = Path(cfg.output_dir)

    # Stage 1: Setup datamodule
    datamodule = SingleCu3sDataModule(**cfg.data)
    datamodule.setup(stage="fit")

    # Access node parameters from cfg.pipeline if present, otherwise from cfg
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
    pipeline = CuvisPipeline("Concrete_AdaClip_Gradient")

    # Data entry node
    # Use anomaly_class_ids=[3] to only treat Stone (class 3) as anomaly for IoU loss
    data_node = LentilsAnomalyDataNode(
        normal_class_ids=pipeline_cfg.get("data_node", {}).get("normal_class_ids", [0, 1]),
        anomaly_class_ids=[3],  # Only class 3 (Stone) is treated as anomaly
    )

    # Optional: Normalize HSI data
    normalizer_cfg = pipeline_cfg.get("normalizer", {})
    normalizer = MinMaxNormalizer(
        eps=normalizer_cfg.get("eps", 1e-6),
        use_running_stats=normalizer_cfg.get("use_running_stats", True),
    )

    # Concrete band selector: 61 → 3 channels (categorical one-of-T per channel)
    selector_cfg = pipeline_cfg.get("selector", {})
    selector = ConcreteBandSelector(
        input_channels=input_channels,
        output_channels=3,
        tau_start=selector_cfg.get("tau_start", 10.0),
        tau_end=selector_cfg.get("tau_end", 0.1),
        max_epochs=cfg.training.trainer.max_epochs,
        use_hard_inference=selector_cfg.get("use_hard_inference", True),
        eps=selector_cfg.get("eps", 1e-6),
        name="concrete_selector",
    )

    debug_cfg = cfg.get("debug", {})
    if debug_cfg and debug_cfg.get("save_intermediates", False):
        # Placeholder for potential future tensor saving; disabled in this version.
        pass

    # AdaClip detector (FROZEN during training)
    adaclip_cfg = pipeline_cfg.get("adaclip", {})
    adaclip = AdaCLIPDetector(
        weight_name=adaclip_cfg.get("weight_name", "pretrained_all"),
        backbone=adaclip_cfg.get("backbone", "ViT-L-14-336"),
        prompt_text=adaclip_cfg.get("prompt_text", ""),  # "normal: lentils, anomaly: stones"),
        image_size=adaclip_cfg.get("image_size", 518),
        prompting_depth=adaclip_cfg.get("prompting_depth", 4),
        prompting_length=adaclip_cfg.get("prompting_length", 5),
        gaussian_sigma=adaclip_cfg.get("gaussian_sigma", 4.0),
        use_half_precision=adaclip_cfg.get("use_half_precision", False),
        enable_warmup=adaclip_cfg.get("enable_warmup", False),
        enable_gradients=True,  # allow gradients to flow, but keep weights frozen
        name="adaclip",
    )
    if debug_cfg and debug_cfg.get("save_intermediates", False):
        # Debug saving hooks for AdaClip could go here if needed.
        pass
    else:
        logger.info("AdaClip debug saving disabled")

    # Choose supervising loss: IoU (default) or BCE-with-logits
    loss_nodes_config = cfg.get("loss_nodes", [])
    use_bce_loss = "bce_loss" in loss_nodes_config

    loss_cfg = pipeline_cfg.get("loss", {})
    if use_bce_loss:
        logger.info("Using AnomalyBCEWithLogits loss per configuration")
        loss_node = AnomalyBCEWithLogits(
            weight=loss_cfg.get("weight", 1.0),
            pos_weight=loss_cfg.get("pos_weight", None),
            reduction=loss_cfg.get("reduction", "mean"),
            name="bce_loss",
        )
    else:
        logger.info("Using IoU loss per configuration")
        loss_node = IoULoss(
            weight=loss_cfg.get("weight", 1.0),
            smooth=loss_cfg.get("smooth", 1e-6),
            normalize_method=loss_cfg.get("normalize_method", "minmax"),
            name="iou_loss",
        )

    # Distinctness loss: repulsion between selector channels to avoid collapse
    distinctness_cfg = pipeline_cfg.get("distinctness_loss", {})
    distinctness_loss = DistinctnessLoss(
        weight=distinctness_cfg.get("weight", 0.1),
        name="distinctness_loss",
    )

    # Decider for inference/visualization (NOT used in training path)
    decider_cfg = pipeline_cfg.get("decider", {})
    decider = TwoStageBinaryDecider(
        image_threshold=decider_cfg.get("image_threshold", 0.20),
        top_k_fraction=decider_cfg.get("top_k_fraction", 0.001),
        quantile=decider_cfg.get("quantile", 0.995),
        name="decider",
    )

    # Metrics
    metrics_node = AnomalyDetectionMetrics(name="metrics_anomaly")

    # Visualizations
    viz_cfg = pipeline_cfg.get("viz", {})
    viz_mask = AnomalyMask(
        name="mask",
        channel=viz_cfg.get("mask_channel", 30),
        up_to=viz_cfg.get("up_to", 5),
    )
    score_viz = ScoreHeatmapVisualizer(
        name="score_heatmap",
        normalize_scores=True,
        up_to=viz_cfg.get("up_to", 5),
    )

    # TensorBoard visualization for Concrete-AdaClip pipeline
    drcnn_tb_viz = DRCNNTensorBoardViz(
        hsi_channels=[0, 20, 40],
        max_samples=4,
        log_every_n_batches=1,
        name="concrete_tensorboard_viz",
    )

    # Monitoring
    tensorboard_node = TensorBoardMonitorNode(
        output_dir=str(output_dir / "tensorboard"),
        run_name=pipeline.name,
    )

    # Stage 3: Connect the pipeline
    pipeline.connect(
        # Data flow: HSI → Normalizer → Concrete Selector → AdaClip
        (data_node.outputs.cube, normalizer.data),
        (normalizer.normalized, selector.data),
        (selector.rgb, adaclip.rgb_image),
        # Loss flow: AdaClip scores → main loss
        (adaclip.scores, loss_node.predictions),
        (data_node.outputs.mask, loss_node.targets),
        # Distinctness loss on selector weights
        (selector.selection_weights, distinctness_loss.selection_weights),
        # Inference/visualization: Scores → Decider/Metrics/Viz
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
        # TensorBoard visualization
        (data_node.outputs.cube, drcnn_tb_viz.hsi_cube),
        (selector.rgb, drcnn_tb_viz.mixer_output),
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

    # Stage 4: Configure training
    training_cfg = TrainingConfig.from_dict(
        OmegaConf.to_container(cfg.training, resolve=True)  # type: ignore[arg-type]
    )

    if training_cfg.trainer.callbacks is None:
        training_cfg.trainer.callbacks = CallbacksConfig()

    training_cfg.trainer.callbacks.checkpoint = ModelCheckpointConfig(
        dirpath=str(output_dir / "checkpoints"),
        monitor="metrics_anomaly/iou",
        mode="max",
        save_top_k=3,
        save_last=True,
        filename="{epoch:02d}",
        verbose=True,
    )

    if training_cfg.scheduler is None:
        training_cfg.scheduler = SchedulerConfig(
            name="reduce_on_plateau",
            monitor="metrics_anomaly/iou",
            mode="max",
            factor=0.5,
            patience=5,
        )

    # Stage 5: Concrete selector uses weight initialization only
    logger.info("Phase 1: Skipping statistical initialization (Concrete selector uses weight init)")

    # Stage 6: Unfreeze selector for gradient training
    logger.info("Phase 2: Unfreezing Concrete selector for gradient training...")
    unfreeze_node_names = list(cfg.unfreeze_nodes) if "unfreeze_nodes" in cfg else [selector.name]
    pipeline.unfreeze_nodes_by_name(unfreeze_node_names)
    logger.info(f"Unfrozen nodes: {unfreeze_node_names}")
    logger.info("AdaClip remains frozen (enable_gradients=True allows gradient flow)")

    # DEBUG: Check selector parameters before training
    logger.info("=== DEBUG: Concrete Selector Parameters (Before Training) ===")
    initial_weights = {}
    for name, param in selector.named_parameters():
        initial_weights[name] = param.data.clone()
        logger.info(
            f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}, "
            f"min={param.min().item():.4f}, max={param.max().item():.4f}, "
            f"mean={param.mean().item():.4f}, std={param.std().item():.4f}"
        )

    selector.eval()
    with torch.no_grad():
        S_initial = selector.get_selection_weights(deterministic=True)
        bands_initial = selector.get_selected_bands()
        tau_initial = selector._get_tau(epoch=0)
    logger.info("=== DEBUG: Initial Concrete Selection Weights ===")
    logger.info(f"  S.shape: {S_initial.shape}")
    logger.info(f"  S per channel sum: {S_initial.sum(dim=-1).tolist()}")
    logger.info(f"  S per channel max: {S_initial.max(dim=-1)[0].tolist()}")
    logger.info(f"  Initial selected bands (argmax): {bands_initial.tolist()}")
    logger.info(f"  Initial temperature (epoch 0): {tau_initial:.4f}")

    # Print top-3 bands per channel for inspection
    for c in range(selector.output_channels):
        top3 = torch.topk(S_initial[c], k=min(3, selector.input_channels))
        logger.info(
            f"  Channel {c} top-3 bands: {top3.indices.tolist()} (weights: {top3.values.tolist()})"
        )

    # Test forward pass to verify selector works
    logger.info("=== DEBUG: Testing Concrete Selector Forward Pass ===")
    selector.train()
    test_cube = first_batch["cube"][:1].float()  # Single sample [1, H, W, C], convert to float
    test_normalized = normalizer.forward(data=test_cube)["normalized"]

    from cuvis_ai_core.utils.types import Context, ExecutionStage

    test_context = Context(stage=ExecutionStage.TRAIN, epoch=0, batch_idx=0, global_step=0)

    with torch.no_grad():
        test_output = selector.forward(data=test_normalized, context=test_context)
        test_rgb = test_output["rgb"]
        test_weights = test_output["selection_weights"]
        test_tau = selector._get_tau(epoch=0)

    logger.info(f"  Test input shape: {test_normalized.shape}")
    logger.info(f"  Test RGB output shape: {test_rgb.shape}")
    logger.info(f"  Test RGB range: [{test_rgb.min().item():.4f}, {test_rgb.max().item():.4f}]")
    logger.info(f"  Test selection_weights shape: {test_weights.shape}")
    logger.info(f"  Test selection_weights sum per channel: {test_weights.sum(dim=-1).tolist()}")
    logger.info(f"  Test temperature (epoch 0): {test_tau:.4f}")
    logger.info("  ✅ Forward pass successful!")

    selector.train()

    # Stage 7: Gradient training
    logger.info("Phase 3: Gradient training with main loss + distinctness penalty...")
    grad_trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=datamodule,
        loss_nodes=[loss_node, distinctness_loss],
        metric_nodes=[metrics_node],
        trainer_config=training_cfg.trainer,
        optimizer_config=training_cfg.optimizer,
        monitors=[tensorboard_node],
    )
    grad_trainer.fit()

    # DEBUG: Check if weights actually changed
    logger.info("=== DEBUG: Concrete Selector Parameters (After Training) ===")
    for name, param in selector.named_parameters():
        if name in initial_weights:
            diff = (param.data - initial_weights[name]).abs()
            logger.info(
                f"  {name}: min={param.min().item():.4f}, max={param.max().item():.4f}, "
                f"mean={param.mean().item():.4f}, std={param.std().item():.4f}"
            )
            logger.info(
                f"    Weight change: max_diff={diff.max().item():.6f}, "
                f"mean_diff={diff.mean().item():.6f}"
            )

    selector.eval()
    with torch.no_grad():
        S_final = selector.get_selection_weights(deterministic=True)
        bands_final = selector.get_selected_bands()

    logger.info("=== DEBUG: Final Concrete Selection Weights ===")
    logger.info(f"  S.shape: {S_final.shape}")
    logger.info(f"  S per channel max: {S_final.max(dim=-1)[0].tolist()}")
    logger.info(f"  Final selected bands (argmax): {bands_final.tolist()}")

    unique_bands = torch.unique(bands_final).numel()
    if unique_bands < selector.output_channels:
        logger.warning(
            f"⚠️  WARNING: Only {unique_bands} unique bands selected out of "
            f"{selector.output_channels} channels!"
        )
    else:
        logger.info(f"✅ All {selector.output_channels} channels selected different bands")

    logger.info("Running validation evaluation with last checkpoint...")
    val_results = grad_trainer.validate(ckpt_path="last")
    logger.info(f"Validation results: {val_results}")

    loss_node_names = [loss_node.name, distinctness_loss.name]
    metric_node_names = [metrics_node.name]
    logger.info(f"Loss nodes: {loss_node_names}")
    logger.info(f"Metric nodes: {metric_node_names}")

    logger.info("Running test evaluation with last checkpoint...")
    test_results = grad_trainer.test(ckpt_path="last")
    logger.info(f"Test results: {test_results}")

    # Stage 8: Save trained pipeline and experiment config
    results_dir = output_dir / "trained_models"
    results_dir.mkdir(parents=True, exist_ok=True)

    pipeline_output_path = results_dir / f"{pipeline.name}.yaml"
    logger.info(f"Saving trained pipeline to: {pipeline_output_path}")

    pipeline.save_to_file(
        str(pipeline_output_path),
        metadata=PipelineMetadata(
            name=pipeline.name,
            description=(
                f"Trained Concrete band selector + AdaClip model from {pipeline.name} trainrun. "
                f"Selector reduces {input_channels} channels to 3 for AdaClip compatibility. "
                f"Selected bands: {bands_final.tolist()}"
            ),
            tags=[
                "gradient",
                "concrete",
                "gumbel",
                "adaclip",
                "band_selector",
                "anomaly_detection",
            ],
            author="cuvis.ai",
        ),
    )
    logger.info(f"  Created: {pipeline_output_path}")
    logger.info(f"  Weights: {pipeline_output_path.with_suffix('.pt')}")

    pipeline_config = pipeline.serialize()
    data_dict = OmegaConf.to_container(cfg.data, resolve=True)  # type: ignore[arg-type]

    trainrun_config = TrainRunConfig(
        name=cfg.name,
        pipeline=pipeline_config,
        data=data_dict,
        training=training_cfg,
        output_dir=str(output_dir),
        loss_nodes=loss_node_names,
        metric_nodes=metric_node_names,
        freeze_nodes=[],
        unfreeze_nodes=unfreeze_node_names,
    )

    trainrun_output_path = results_dir / f"{cfg.name}_trainrun.yaml"
    logger.info(f"Saving trainrun config to: {trainrun_output_path}")
    trainrun_config.save_to_file(str(trainrun_output_path))

    logger.info("=== Training Complete ===")
    logger.info(f"Trained pipeline saved: {pipeline_output_path}")
    logger.info(f"TrainRun config saved: {trainrun_output_path}")
    logger.info(f"TensorBoard logs: {tensorboard_node.output_dir}")
    logger.info("To restore this trainrun:")
    logger.info(
        "  uv run python examples/serialization/restore_trainrun.py "
        f"--trainrun-path {trainrun_output_path}"
    )
    logger.info(f"View TensorBoard: uv run tensorboard --logdir={output_dir / 'tensorboard'}")
    logger.info("TensorBoard will show:")
    logger.info("  - HSI input images (false-color RGB)")
    logger.info("  - Concrete selector output (what AdaClip sees as input)")
    logger.info("  - Ground truth masks")
    logger.info("  - AdaClip anomaly score heatmaps")
    logger.info(f"Final selected bands: {bands_final.tolist()}")


if __name__ == "__main__":
    main()
