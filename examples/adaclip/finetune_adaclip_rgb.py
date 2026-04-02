"""Fine-tune AdaCLIP adapter layers with standard RGB band selection.

Matches the AdaCLIP paper (Cao et al., ECCV 2024):
  - Only adapter layers (~10 M params) are updated; CLIP backbone is frozen.
  - Focal + Dice loss (pixel-level) + Focal loss (image-level).
  - lr=0.01, AdamW with betas=(0.5, 0.999), 5 epochs, batch_size=1.

Pipeline:
  HSI → LentilsAnomalyDataNode → MinMaxNormalizer → FixedWavelengthSelector(RGB)
      → AdaCLIPDetector (adapter layers unfrozen) → AdaCLIPFocalDiceLoss

Usage:
    cd /home/dev/anish/cuvis-ai
    uv run python examples/adaclip/finetune_adaclip_rgb.py
    uv run python examples/adaclip/finetune_adaclip_rgb.py \\
        data.splits_csv=lentils_splits.csv
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import GradientTrainer, StatisticalTrainer
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.pipeline import PipelineMetadata
from cuvis_ai_schemas.training import (
    CallbacksConfig,
    ModelCheckpointConfig,
    TrainingConfig,
    create_callbacks_from_config,
)
from pytorch_lightning.callbacks import Callback
from loguru import logger
from omegaconf import OmegaConf

from cuvis_ai.data import MultiFileCu3sDataModule
from cuvis_ai.deciders.binary_decider import QuantileBinaryDecider
from cuvis_ai.node.channel_selector import FixedWavelengthSelector
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.losses import AdaCLIPFocalDiceLoss
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.normalization import MinMaxNormalizer


class CudaEmptyCacheCallback(Callback):
    """Call ``torch.cuda.empty_cache()`` between phases to ease fragmentation OOMs."""

    def on_train_epoch_end(self, trainer: object, pl_module: object) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self, trainer: object, pl_module: object) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@hydra.main(
    config_path="../../configs/",
    config_name="trainrun/finetune_adaclip",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    logger.info("=== Fine-tune AdaCLIP adapter layers (RGB) ===")

    plugins_manifest = Path(__file__).resolve().parents[2] / "configs" / "plugins" / "adaclip.yaml"
    registry = NodeRegistry()
    registry.load_plugins(plugins_manifest)
    AdaCLIPDetector = NodeRegistry.get("cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector")
    logger.info("AdaCLIP plugin loaded")

    output_dir = Path(cfg.output_dir).resolve()
    output_dir = output_dir.parent / f"{output_dir.name}_rgb"

    # ── Data ──────────────────────────────────────────────────────────
    datamodule = MultiFileCu3sDataModule(
        splits_csv=cfg.data.splits_csv,
        batch_size=cfg.data.batch_size,
        processing_mode=cfg.data.processing_mode,
        num_workers=cfg.data.get("num_workers", 0),
    )
    datamodule.setup(stage="fit")

    train_loader = datamodule.train_dataloader()
    first_batch = next(iter(train_loader))
    input_channels = first_batch["cube"].shape[-1]
    logger.info(f"Input channels: {input_channels}, batch_size: {cfg.data.batch_size}")

    # ── Pipeline ──────────────────────────────────────────────────────
    pipeline = CuvisPipeline("finetune_adaclip_rgb")

    data_node = LentilsAnomalyDataNode(normal_class_ids=[0])

    normalizer = MinMaxNormalizer(
        eps=1e-6,
        use_running_stats=True,
        max_initialization_frames=cfg.get("minmax_init_frames", None),
    )

    selector = FixedWavelengthSelector(
        target_wavelengths=(650.0, 550.0, 450.0),
        norm_mode="running",
        running_warmup_frames=0,
        freeze_running_bounds_after_frames=20,
        name="rgb_selector",
    )
    # Avoid slow full-dataset statistical initialization for selector bounds.
    # Running mode updates bounds online during forward passes.
    selector._requires_initial_fit_override = False

    adaclip = AdaCLIPDetector(
        weight_name="pretrained_all",
        backbone="ViT-L-14-336",
        prompt_text="lentil",
        image_size=518,
        prompting_depth=4,
        prompting_length=5,
        gaussian_sigma=4.0,
        use_half_precision=False,
        enable_warmup=False,
        enable_gradients=True,
        training_aggregation=False,
        name="adaclip",
    )

    focal_dice = AdaCLIPFocalDiceLoss(
        weight=1.0,
        focal_gamma=2.0,
        image_loss_weight=1.0,
        name="focal_dice_loss",
    )

    decider = QuantileBinaryDecider(quantile=0.995, name="decider")
    metrics_node = AnomalyDetectionMetrics(name="metrics_anomaly")
    tb = TensorBoardMonitorNode(
        output_dir=str(output_dir / "tensorboard"),
        run_name=f"finetune_adaclip_rgb",
    )

    pipeline.connect(
        (data_node.outputs.cube, normalizer.data),
        (normalizer.normalized, selector.cube),
        (data_node.outputs.wavelengths, selector.wavelengths),
        (selector.rgb_image, adaclip.rgb_image),
        # Loss — per-layer maps + 2-channel image score (paper-faithful)
        (adaclip.scores, focal_dice.predictions),
        (data_node.outputs.mask, focal_dice.targets),
        (adaclip.per_layer_scores, focal_dice.per_layer_scores),
        (adaclip.image_score_2ch, focal_dice.image_score_2ch),
        # Metrics / decider (use aggregated scores)
        (adaclip.scores, decider.logits),
        (adaclip.scores, metrics_node.logits),
        (decider.decisions, metrics_node.decisions),
        (data_node.outputs.mask, metrics_node.targets),
        # Monitor
        (metrics_node.metrics, tb.metrics),
    )

    pipeline.visualize(
        format="render_graphviz",
        output_path=str(output_dir / "pipeline" / f"{pipeline.name}.png"),
        show_execution_stage=True,
    )

    # Move pipeline to GPU before statistical init so model loads on the
    # correct device. This also syncs the upstream AdaCLIP's self.device
    # attribute so text embeddings are generated on the same device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Moving pipeline to device: {device}")
    pipeline.to(device)

    # ── Training ──────────────────────────────────────────────────────
    training_cfg = TrainingConfig.from_dict(
        OmegaConf.to_container(cfg.training, resolve=True)
    )

    if training_cfg.trainer.callbacks is None:
        training_cfg.trainer.callbacks = CallbacksConfig()
    training_cfg.trainer.callbacks.checkpoint = ModelCheckpointConfig(
        dirpath=str(output_dir / "checkpoints"),
        monitor="metrics_anomaly/iou",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="{epoch:02d}",
        verbose=True,
    )
    lightning_callbacks = list(
        create_callbacks_from_config(training_cfg.trainer.callbacks)
    )
    lightning_callbacks.append(CudaEmptyCacheCallback())

    # Phase 1: Statistical init (normalizer running stats)
    if normalizer.requires_initial_fit:
        logger.info("Phase 1: Statistical initialization (normalizer)...")
        StatisticalTrainer(pipeline=pipeline, datamodule=datamodule).fit()

    # Phase 2: Unfreeze AdaCLIP adapter layers
    unfreeze_names = list(cfg.unfreeze_nodes) if "unfreeze_nodes" in cfg else ["adaclip"]
    pipeline.unfreeze_nodes_by_name(unfreeze_names)
    logger.info(f"Unfrozen: {unfreeze_names}")

    # Re-apply device placement after unfreeze so any newly-enabled params
    # (e.g. prompt tensors) are on the correct device before gradient training.
    pipeline.to(device)
    logger.info(f"Pipeline re-moved to {device} after unfreeze")

    # Log trainable param count
    n_train = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in pipeline.parameters())
    logger.info(f"Trainable: {n_train:,} / {n_total:,} params ({100*n_train/n_total:.2f}%)")

    # Phase 3: Gradient training
    logger.info("Phase 2: Gradient training (AdaCLIP adapter layers)...")
    grad_trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=datamodule,
        loss_nodes=[focal_dice],
        metric_nodes=[metrics_node],
        trainer_config=training_cfg.trainer,
        optimizer_config=training_cfg.optimizer,
        monitors=[tb],
        callbacks=lightning_callbacks,
    )
    grad_trainer.fit()

    # Evaluate
    logger.info("Validation with best checkpoint...")
    grad_trainer.validate(ckpt_path="best")
    logger.info("Test with best checkpoint...")
    grad_trainer.test(ckpt_path="best")

    # Save
    results_dir = output_dir / "trained_models"
    results_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = results_dir / f"{pipeline.name}.yaml"
    pipeline.save_to_file(
        str(pipeline_path),
        metadata=PipelineMetadata(
            name=pipeline.name,
            description="Fine-tuned AdaCLIP adapter layers with RGB band selection on lentils.",
            tags=["finetune", "adaclip", "rgb", "anomaly_detection"],
            author="cuvis.ai",
        ),
    )
    logger.info(f"Pipeline saved: {pipeline_path}")
    logger.info(f"TensorBoard: uv run tensorboard --logdir={output_dir / 'tensorboard'}")


if __name__ == "__main__":
    main()
