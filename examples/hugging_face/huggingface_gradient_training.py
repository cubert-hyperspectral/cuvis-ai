"""Demo: HuggingFace Gradient Training for Channel Selection (Phase 3)

This example demonstrates using a frozen HuggingFace model (AdaCLIP) as a
differentiable loss function to train a SoftChannelSelector. The goal is to
discover optimal hyperspectral bands for anomaly detection.

Architecture:
    HSI Cube -> ChannelSelector -> HyperspectralToRGB -> AdaCLIP(frozen) -> Loss

Training Flow:
    1. Channel selector reduces N channels to M selected channels
    2. RGB converter creates RGB from selected channels
    3. Frozen AdaCLIP evaluates RGB for anomalies
    4. Loss computed between AdaCLIP output and ground truth
    5. Gradients flow back ONLY to channel selector (AdaCLIP frozen)
    6. Channel weights optimized to maximize anomaly detection

Run:
    python examples/hugging_face/huggingface_gradient_training.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from cuvis_ai.data.lentils_anomaly import SingleCu3sDataModule
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.huggingface import AdaCLIPLocalNode
from cuvis_ai.node.metrics import SelectorDiversityMetric, SelectorEntropyMetric
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.node import Node
from cuvis_ai.node.selector import SoftChannelSelector
from cuvis_ai.node.visualizations import (
    AnomalyMask,
    CubeRGBVisualizer,
    ScoreHeatmapVisualizer,
)
from cuvis_ai.pipeline.pipeline import CuvisCanvas
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.training import GradientTrainer
from cuvis_ai.training.config import (
    CallbacksConfig,
    EarlyStoppingConfig,
    LearningRateMonitorConfig,
    ModelCheckpointConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
    TrainingConfig,
)

# ============================================================================
# Loss Function Implementation (Inline for Phase 3)
# ============================================================================


class BinaryMaskLoss(Node):
    """Loss for binary mask comparison.

    Computes loss between predicted and target binary masks using various metrics.
    Implemented inline for Phase 3 demo - may be moved to losses.py later.

    Parameters
    ----------
    loss_type : str, optional
        Type of loss (default: "dice")
        Options:
        - "bce": Binary Cross-Entropy
        - "dice": Dice loss (1 - Dice coefficient)
        - "iou": IoU-based loss (1 - IoU)
    reduction : str, optional
        Loss reduction method: "mean", "sum", "none" (default: "mean")
    smooth : float, optional
        Smoothing factor for Dice/IoU to avoid division by zero (default: 1e-6)
    """

    INPUT_SPECS = {
        "pred_scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Predicted anomaly scores [B, H, W, 1] in range [0, 1]",
        ),
        "target_mask": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Target binary mask [B, H, W, 1]",
        ),
    }

    OUTPUT_SPECS = {
        "loss": PortSpec(
            dtype=torch.float32,
            shape=(),
            description="Scalar loss value",
        ),
    }

    def __init__(
        self,
        loss_type: str = "dice",
        reduction: str = "mean",
        smooth: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        self.loss_type = loss_type
        self.reduction = reduction
        self.smooth = smooth

        super().__init__(
            loss_type=loss_type,
            reduction=reduction,
            smooth=smooth,
            **kwargs,
        )

    def forward(
        self,
        pred_scores: Tensor,
        target_mask: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Compute loss between predicted scores and target masks.

        Parameters
        ----------
        pred_scores : Tensor
            Predicted anomaly scores [B, H, W, 1] in range [0, 1]
        target_mask : Tensor
            Target binary mask [B, H, W, 1]

        Returns
        -------
        dict[str, Tensor]
            Dictionary with "loss" key
        """
        # pred_scores is already float, just need to convert target to float
        pred = pred_scores
        target = target_mask.float()

        # Compute loss based on type
        if self.loss_type == "bce":
            loss = nn.functional.binary_cross_entropy(pred, target, reduction=self.reduction)
        elif self.loss_type == "dice":
            # Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            loss = 1.0 - dice  # Dice loss
        elif self.loss_type == "iou":
            # IoU: |A ∩ B| / |A ∪ B|
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum() - intersection
            iou = (intersection + self.smooth) / (union + self.smooth)
            loss = 1.0 - iou  # IoU loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return {"loss": loss}


# ============================================================================
# RGB Converter (Reuse from Phase 2)
# ============================================================================


class HyperspectralToRGBNode(Node):
    """Convert hyperspectral cubes to RGB by selecting specific channel indices.

    Simplified version that only performs channel selection without wavelength logic.
    """

    INPUT_SPECS = {
        "hsi_cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C]",
        ),
    }

    OUTPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB image [B, H, W, 3]",
        ),
    }

    def __init__(
        self,
        band_indices: list[int] | None = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        self.band_indices = band_indices
        self.normalize = normalize

        super().__init__(
            band_indices=band_indices,
            normalize=normalize,
            **kwargs,
        )

    def forward(self, hsi_cube: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        """Convert hyperspectral cube to RGB by selecting 3 channels."""
        n_channels = hsi_cube.shape[-1]

        # Use provided indices or default to equally spaced
        if self.band_indices is not None:
            indices = self.band_indices
        else:
            # Default: equally spaced bands (R, G, B positions)
            indices = [n_channels // 4, n_channels // 2, 3 * n_channels // 4]

        # Select bands
        rgb = hsi_cube[..., indices]

        # Normalize to [0, 1] if requested
        if self.normalize:
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

        return {"rgb_image": rgb}


# ============================================================================
# Training Utilities
# ============================================================================


def verify_gradient_setup(
    pipeline: CuvisCanvas,
    channel_selector: SoftChannelSelector,
    adaclip_node: AdaCLIPLocalNode,
) -> bool:
    """Verify that only channel selector has gradients.

    Parameters
    ----------
    pipeline : CuvisCanvas
        The training pipeline
    channel_selector : SoftChannelSelector
        Channel selector node
    adaclip_node : AdaCLIPLocalNode
        Frozen AdaCLIP node

    Returns
    -------
    bool
        True if gradient setup is correct
    """
    logger.info("Verifying gradient configuration...")

    # Check channel selector
    selector_trainable = any(p.requires_grad for p in channel_selector.parameters())
    logger.info(f"  Channel selector trainable: {selector_trainable}")

    # Check AdaCLIP
    adaclip_trainable = any(p.requires_grad for p in adaclip_node.model.parameters())
    logger.info(f"  AdaCLIP trainable: {adaclip_trainable}")

    if selector_trainable and not adaclip_trainable:
        logger.success("✓ Gradient setup correct!")
        logger.success("  - Channel selector will be trained")
        logger.success("  - AdaCLIP will remain frozen")
        return True
    else:
        logger.error("✗ Gradient setup incorrect!")
        if not selector_trainable:
            logger.error("  - Channel selector is frozen (should be trainable)")
        if adaclip_trainable:
            logger.error("  - AdaCLIP is trainable (should be frozen)")
        return False


def analyze_channel_weights(
    channel_selector: SoftChannelSelector,
    wavelengths: Tensor | None,
    epoch: int | str,
    writer: SummaryWriter | None = None,
) -> None:
    """Analyze and visualize learned channel weights.

    Parameters
    ----------
    channel_selector : SoftChannelSelector
        Trained channel selector
    wavelengths : Tensor or None
        Wavelength values for each channel
    epoch : int or str
        Current training epoch
    writer : SummaryWriter, optional
        TensorBoard writer for logging
    """
    # Get channel weights (after softmax)
    with torch.no_grad():
        logits = channel_selector.channel_logits
        weights = torch.nn.functional.softmax(logits, dim=0)

    # Find top N channels
    n_select = channel_selector.n_select
    top_weights, top_indices = torch.topk(weights, k=n_select)

    logger.info(f"\nTop {n_select} selected channels (epoch {epoch}):")
    for i, (idx, weight) in enumerate(zip(top_indices, top_weights, strict=True)):
        if wavelengths is not None:
            wl = wavelengths[idx].item()
            logger.info(f"  {i + 1}. Channel {idx.item():3d} (λ={wl:.1f}nm): {weight.item():.4f}")
        else:
            logger.info(f"  {i + 1}. Channel {idx.item():3d}: {weight.item():.4f}")

    # Log to TensorBoard
    if writer is not None and isinstance(epoch, int):
        # Log weight distribution
        writer.add_histogram("channel_weights", weights, epoch)

        # Log top channel wavelengths
        if wavelengths is not None:
            top_wavelengths = wavelengths[top_indices].cpu().numpy()
            for i, wl in enumerate(top_wavelengths):
                writer.add_scalar(f"top_wavelengths/rank_{i + 1}", wl, epoch)


def save_best_channels(
    channel_selector: SoftChannelSelector,
    wavelengths: np.ndarray,
    output_dir: Path,
) -> None:
    """Save the best selected channels to a JSON file.

    Parameters
    ----------
    channel_selector : SoftChannelSelector
        Trained channel selector
    wavelengths : np.ndarray
        Wavelength values for each channel
    output_dir : Path
        Directory to save the JSON file
    """
    with torch.no_grad():
        logits = channel_selector.channel_logits
        weights = torch.nn.functional.softmax(logits, dim=0)
        n_select = channel_selector.n_select
        top_weights, top_indices = torch.topk(weights, k=n_select)

    top_indices_list = top_indices.cpu().numpy().tolist()
    top_weights_list = top_weights.cpu().numpy().tolist()
    top_wavelengths_list = wavelengths[top_indices_list].tolist()

    result = {
        "n_select": int(n_select),
        "selected_channels": [
            {
                "index": int(idx),
                "wavelength_nm": float(wl),
                "weight": float(wt),
            }
            for idx, wl, wt in zip(
                top_indices_list, top_wavelengths_list, top_weights_list, strict=True
            )
        ],
        "all_weights": weights.cpu().numpy().tolist(),
        "wavelengths": wavelengths.tolist(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "best_selected_channels.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.success(f"Saved best selected channels to {output_path}")


# ============================================================================
# Main Training Script
# ============================================================================


def main() -> None:
    """Run gradient training demo using GradientTrainer framework."""
    logger.info("=== HuggingFace Gradient Training Demo (Phase 3) ===")

    # Step 1: Setup data
    logger.info("Loading hyperspectral data...")
    # Resolve data directory relative to script location (works in debugger too)
    script_dir = Path(__file__).parent  # examples/hugging_face/
    project_root = script_dir.parent.parent  # gitlab_cuvis_ai_3/cuvis.ai/
    data_dir = project_root / "data" / "Lentils"

    datamodule = SingleCu3sDataModule(
        data_dir=str(data_dir),  # Convert to string for compatibility
        dataset_name="Lentils",
        batch_size=4,
        train_ids=[0],
        val_ids=[1, 2, 5],
        test_ids=[6, 7],
    )
    datamodule.setup(stage="fit")

    wavelengths = datamodule.train_ds.wavelengths
    n_channels = len(wavelengths)
    logger.info(
        f"Data loaded: {n_channels} channels, "
        f"λ ∈ [{wavelengths.min():.1f}, {wavelengths.max():.1f}] nm"
    )

    # Step 2: Build pipeline
    logger.info("Building training pipeline...")
    pipeline = CuvisCanvas("HF_Gradient_Training")

    # Data node
    data_node = LentilsAnomalyDataNode(
        normal_class_ids=[0, 1],
        name="data",
    )

    # Channel selector (trainable)
    channel_selector = SoftChannelSelector(
        n_select=3,
        input_channels=n_channels,
        init_method="variance",
        name="channel_selector",
    )
    channel_selector.unfreeze()  # Enable training

    # RGB converter
    rgb_converter = HyperspectralToRGBNode(
        band_indices=None,  # Will use equally spaced defaults
        normalize=True,
        name="rgb_converter",
    )

    # Frozen AdaCLIP
    adaclip_node = AdaCLIPLocalNode(
        model_name="openai/clip-vit-base-patch32",
        freeze=True,  # FROZEN - gradients pass through but params don't update
        device="auto",
        text_prompt="A photo of a normal lentil.",
        name="adaclip",
    )

    # Loss node (for GradientTrainer)
    loss_node = BinaryMaskLoss(
        loss_type="dice",
        reduction="mean",
        name="mask_loss",
    )

    # Metric nodes (optional)
    selector_entropy_metric = SelectorEntropyMetric(name="selector_entropy")
    selector_diversity_metric = SelectorDiversityMetric(name="selector_diversity")

    # Visualization nodes for TensorBoard
    score_viz = ScoreHeatmapVisualizer(normalize_scores=True, up_to=5, name="score_viz")
    viz_mask = AnomalyMask(channel=3, up_to=5, name="viz_mask")  # channel=3 for stone class
    rgb_viz = CubeRGBVisualizer(up_to=5, name="rgb_viz")
    tensorboard_node = TensorBoardMonitorNode(
        run_name="hf_gradient_training",
        output_dir="./outputs/",
        name="tensorboard_monitor",
    )

    # Step 3: Connect graph
    pipeline.connect(
        # Data flow
        (data_node.outputs.cube, channel_selector.data),
        (channel_selector.selected, rgb_converter.hsi_cube),
        (rgb_converter.rgb_image, adaclip_node.image),
        # Loss computation - use anomaly_scores (float, differentiable) not anomaly_mask (bool)
        (adaclip_node.anomaly_scores, loss_node.pred_scores),
        (data_node.outputs.mask, loss_node.target_mask),
        # Metrics (val/test only)
        (channel_selector.weights, selector_entropy_metric.weights),
        (channel_selector.weights, selector_diversity_metric.weights),
        # Visualizations for TensorBoard
        (adaclip_node.anomaly_scores, score_viz.scores),
        (score_viz.artifacts, tensorboard_node.artifacts),
        (adaclip_node.anomaly_mask, viz_mask.decisions),
        (adaclip_node.anomaly_scores, viz_mask.scores),
        (data_node.outputs.mask, viz_mask.mask),
        (data_node.outputs.cube, viz_mask.cube),
        (viz_mask.artifacts, tensorboard_node.artifacts),
        # RGB visualization with channel weights
        (data_node.outputs.cube, rgb_viz.cube),
        (channel_selector.weights, rgb_viz.weights),
        (data_node.outputs.wavelengths, rgb_viz.wavelengths),
        (rgb_viz.artifacts, tensorboard_node.artifacts),
    )

    logger.success("Pipeline built successfully")

    # Step 4: Verify gradient setup
    if not verify_gradient_setup(pipeline, channel_selector, adaclip_node):
        raise RuntimeError("Gradient setup verification failed!")

    # Step 5: Setup training configuration
    logger.info("Setting up training configuration...")

    training_cfg = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(
            max_epochs=20,
            accelerator="auto",
            log_every_n_steps=10,
            callbacks=CallbacksConfig(
                early_stopping=[
                    EarlyStoppingConfig(
                        monitor="val_loss",
                        mode="min",
                        patience=5,
                        min_delta=0.001,
                    ),
                ],
                model_checkpoint=ModelCheckpointConfig(
                    dirpath="./outputs/hf_gradient_training_checkpoints",
                    filename="best-{epoch:02d}-{val_loss:.4f}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                    verbose=True,
                ),
                learning_rate_monitor=LearningRateMonitorConfig(
                    logging_interval="epoch",
                ),
            ),
        ),
        optimizer=OptimizerConfig(
            name="adam",
            lr=0.01,
            scheduler=SchedulerConfig(
                name="reduce_on_plateau",
                monitor="val_loss",
                mode="min",
                factor=0.5,
                patience=3,
                threshold=0.001,
            ),
        ),
    )

    logger.info(f"Training configuration: max_epochs={training_cfg.trainer.max_epochs}")
    logger.info(f"Optimizer: {training_cfg.optimizer.name} (lr={training_cfg.optimizer.lr})")

    # Step 6: Create trainer and run training
    logger.info("Starting gradient training with GradientTrainer...")

    grad_trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=datamodule,
        loss_nodes=[loss_node],
        metric_nodes=[selector_entropy_metric, selector_diversity_metric],
        trainer_config=training_cfg.trainer,
        optimizer_config=training_cfg.optimizer,
    )

    # Train
    grad_trainer.fit()

    # Validate
    logger.info("Validating best checkpoint...")
    grad_trainer.validate(ckpt_path="best")

    # Test
    logger.info("Testing best checkpoint...")
    test_results = grad_trainer.test(ckpt_path="best")

    # Step 7: Final analysis
    logger.info("\n" + "=" * 60)
    logger.success("Training completed!")
    logger.info("=" * 60)

    if test_results:
        logger.info(f"Test results: {test_results[0]}")

    # Analyze final learned weights
    logger.info("\nFinal learned channel weights:")
    analyze_channel_weights(
        channel_selector,
        torch.tensor(wavelengths),
        epoch="final",
    )

    # Save best selected channels to JSON
    logger.info("\nSaving best selected channels...")
    save_best_channels(
        channel_selector,
        wavelengths,
        Path("./outputs/hf_gradient_training_checkpoints"),
    )

    logger.info("\nPhase 3 Complete! ✨")
    logger.info("Next steps:")
    logger.info("  1. View training curves and images in TensorBoard")
    logger.info("     tensorboard --logdir ./outputs/hf_gradient_training_runs")
    logger.info("  2. Analyze learned wavelengths")
    logger.info("  3. Compare with manually selected bands")
    logger.info("  4. Check checkpoints in ./outputs/hf_gradient_training_checkpoints")
    logger.info(
        "  5. Check best channels JSON in ./outputs/hf_gradient_training_checkpoints/best_selected_channels.json"
    )


if __name__ == "__main__":
    main()
