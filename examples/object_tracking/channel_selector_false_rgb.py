"""ALL-5288 Experiment A: LearnableChannelMixer for Object Tracking False RGB.

Two modes (via Hydra override ``mode=inspect`` or ``mode=train``):

**Inspect mode** — Discover annotated frames, export false RGB video with mask
overlays for visual inspection.  User then decides which frame IDs to use for
train/val/test splits.

**Train mode** — Build pipeline with LearnableChannelMixer optimized by
ForegroundContrastLoss + DistinctnessLoss, train via StatisticalTrainer →
GradientTrainer, save results.

Usage::

    uv run python examples/object_tracking/channel_selector_false_rgb.py mode=inspect
    uv run python examples/object_tracking/channel_selector_false_rgb.py mode=train
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from cuvis_ai_core.data.datasets import SingleCu3sDataModule, SingleCu3sDataset
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import GradientTrainer, StatisticalTrainer
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PipelineMetadata
from cuvis_ai_schemas.training import TrainingConfig, TrainRunConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from cuvis_ai.node.band_selection import RangeAverageFalseRGBSelector
from cuvis_ai.node.channel_mixer import LearnableChannelMixer
from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.losses import DistinctnessLoss, ForegroundContrastLoss
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.node.video import ToVideoNode
from cuvis_ai.node.visualizations import ChannelSelectorFalseRGBViz

# ---------------------------------------------------------------------------
# Inspect mode
# ---------------------------------------------------------------------------


def inspect_dataset(cfg: DictConfig) -> None:
    """Load CU3S data, discover annotated frames, export false RGB video with mask overlays."""

    output_dir = Path(cfg.output_dir) / "inspect"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)

    # Extract annotated frame indices directly from the COCO JSON
    annotation_json_path = data_cfg.get("annotation_json_path")
    assert annotation_json_path is not None, "annotation_json_path is required for inspect mode"

    with open(annotation_json_path) as f:
        coco = json.load(f)

    # image_id in COCO corresponds to the measurement/frame index
    annotated_image_ids = {ann["image_id"] for ann in coco["annotations"]}
    annotated_frame_ids = sorted(annotated_image_ids)

    logger.info(f"Frames with annotations (from COCO JSON): {len(annotated_frame_ids)}")
    logger.info(f"Annotated frame IDs: {annotated_frame_ids}")

    # Load only annotated frames via measurement_indices
    dataset = SingleCu3sDataset(
        cu3s_file_path=data_cfg["cu3s_file_path"],
        processing_mode=data_cfg.get("processing_mode", "Raw"),
        annotation_json_path=annotation_json_path,
        measurement_indices=annotated_frame_ids,
    )

    # Confirm annotations by checking masks on loaded frames
    confirmed_frame_ids = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        mask = sample.get("mask")
        if mask is not None and mask.any():
            confirmed_frame_ids.append(sample["mesu_index"])

    logger.info(
        f"Confirmed frames with non-empty masks: {len(confirmed_frame_ids)}/{len(annotated_frame_ids)}"
    )
    logger.info(f"Confirmed frame IDs: {confirmed_frame_ids}")

    # Export false RGB video + TensorBoard artifacts via pipeline (annotated frames only)
    batch_size = data_cfg.get("batch_size", 1)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0)

    pipeline = CuvisPipeline("FalseRGB_Inspect")

    cu3s_data = CU3SDataNode(name="cu3s_data")
    false_rgb = RangeAverageFalseRGBSelector(
        red_range=(580.0, 650.0),
        green_range=(500.0, 580.0),
        blue_range=(420.0, 500.0),
        name="range_average_false_rgb",
    )
    to_video = ToVideoNode(
        output__video_path=str(output_dir / "false_rgb.mp4"),
        frame_rate=10.0,
        name="to_video",
    )
    viz = ChannelSelectorFalseRGBViz(
        name="false_rgb_viz",
        max_samples=batch_size,
        log_every_n_batches=1,
        execution_stages={ExecutionStage.INFERENCE},
    )
    tensorboard_node = TensorBoardMonitorNode(
        output_dir=str(output_dir / "tensorboard"),
        run_name="inspect",
    )

    pipeline.connect(
        (cu3s_data.outputs.cube, false_rgb.cube),
        (cu3s_data.outputs.wavelengths, false_rgb.wavelengths),
        (false_rgb.rgb_image, to_video.rgb_image),
        (false_rgb.rgb_image, viz.rgb_output),
        (cu3s_data.outputs.mask, viz.mask),
        (viz.artifacts, tensorboard_node.artifacts),
    )

    total_frames = len(dataset)
    logger.info(
        f"Exporting false RGB video with mask overlays ({total_frames} annotated frames)..."
    )
    processed = 0
    try:
        for batch in dataloader:
            if batch["cube"].dtype != torch.uint16:
                batch["cube"] = batch["cube"].to(torch.uint16)
            pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)
            processed += batch["cube"].shape[0]
            if processed % 50 == 0 or processed == total_frames:
                logger.info(f"Processed {processed}/{total_frames} frames")
    finally:
        to_video.close()

    logger.success(f"Inspect video saved: {output_dir / 'false_rgb.mp4'}")
    logger.success(f"TensorBoard logs: {output_dir / 'tensorboard'}")

    # Print summary for user to copy into data config
    logger.info("=" * 60)
    logger.info("ANNOTATED FRAME IDS (copy into configs/data/tracking_cap_and_car.yaml):")
    logger.info(f"  All annotated: {annotated_frame_ids}")
    logger.info(f"  Confirmed (non-empty mask): {confirmed_frame_ids}")
    logger.info(f"  Total annotated: {len(annotated_frame_ids)}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Train mode
# ---------------------------------------------------------------------------


def run_experiment_mixer(cfg: DictConfig) -> None:
    """Build pipeline, train mixer with fg/bg contrast loss, save results."""

    output_dir = Path(cfg.output_dir)

    # Setup datamodule
    datamodule = SingleCu3sDataModule(**cfg.data)
    datamodule.setup(stage="fit")

    # Build pipeline
    pipeline = CuvisPipeline("Channel_Selector_FalseRGB")

    mixer_cfg = cfg.get("mixer", {})
    loss_cfg = cfg.get("losses", {})

    data_node = CU3SDataNode(name="cu3s_data")
    normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
    mixer = LearnableChannelMixer(
        input_channels=mixer_cfg.get("input_channels", 61),
        output_channels=mixer_cfg.get("output_channels", 3),
        init_method=mixer_cfg.get("init_method", "pca"),
        normalize_output=mixer_cfg.get("normalize_output", True),
    )

    fg_contrast_cfg = loss_cfg.get("foreground_contrast", {})
    contrast_loss = ForegroundContrastLoss(
        name="foreground_contrast",
        weight=fg_contrast_cfg.get("weight", 1.0),
        compactness_weight=fg_contrast_cfg.get("compactness_weight", 0.1),
    )

    dist_cfg = loss_cfg.get("distinctness", {})
    distinctness_loss = DistinctnessLoss(
        name="distinctness",
        weight=dist_cfg.get("weight", 0.1),
    )

    viz = ChannelSelectorFalseRGBViz(
        name="false_rgb_viz",
        max_samples=4,
        log_every_n_batches=1,
    )
    tensorboard_node = TensorBoardMonitorNode(
        output_dir=str(output_dir / "tensorboard"),
        run_name=pipeline.name,
    )

    # Connect pipeline
    pipeline.connect(
        # Processing flow
        (data_node.outputs.cube, normalizer.data),
        (normalizer.normalized, mixer.data),
        # Loss flow
        (mixer.rgb, contrast_loss.rgb),
        (data_node.outputs.mask, contrast_loss.mask),
        (mixer.weights, distinctness_loss.selection_weights),
        # Visualization flow
        (mixer.rgb, viz.rgb_output),
        (data_node.outputs.mask, viz.mask),
        (viz.artifacts, tensorboard_node.artifacts),
    )

    # Visualize pipeline graph
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

    # Training config
    training_cfg = TrainingConfig.from_dict(OmegaConf.to_container(cfg.training, resolve=True))

    # Phase 1: Statistical initialization (PCA init for mixer, running stats for normalizer)
    logger.info("Phase 1: Statistical initialization...")
    stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    stat_trainer.fit()

    # Phase 2: Unfreeze mixer for gradient training
    logger.info("Phase 2: Unfreezing mixer for gradient training...")
    unfreeze_node_names = [mixer.name]
    pipeline.unfreeze_nodes_by_name(unfreeze_node_names)
    logger.info(f"Unfrozen nodes: {unfreeze_node_names}")

    # Phase 3: Gradient training
    logger.info("Phase 3: Gradient-based channel mixing optimization...")
    grad_trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=datamodule,
        loss_nodes=[contrast_loss, distinctness_loss],
        metric_nodes=[],
        trainer_config=training_cfg.trainer,
        optimizer_config=training_cfg.optimizer,
        monitors=[tensorboard_node],
    )
    grad_trainer.fit()

    # Evaluation
    logger.info("Running validation evaluation...")
    val_results = grad_trainer.validate()
    logger.info(f"Validation results: {val_results}")

    logger.info("Running test evaluation...")
    test_results = grad_trainer.test()
    logger.info(f"Test results: {test_results}")

    # Save trained pipeline and trainrun config
    results_dir = output_dir / "trained_models"
    loss_node_names = [contrast_loss.name, distinctness_loss.name]

    pipeline_output_path = results_dir / f"{pipeline.name}.yaml"
    logger.info(f"Saving trained pipeline to: {pipeline_output_path}")

    pipeline.save_to_file(
        str(pipeline_output_path),
        metadata=PipelineMetadata(
            name=pipeline.name,
            description="LearnableChannelMixer trained with ForegroundContrastLoss for object tracking false RGB",
            tags=["channel_mixer", "false_rgb", "object_tracking", "foreground_contrast"],
            author="cuvis.ai",
        ),
    )

    pipeline_config = pipeline.serialize()
    trainrun_config = TrainRunConfig(
        name=cfg.name,
        pipeline=pipeline_config,
        data=cfg.data,
        training=training_cfg,
        output_dir=str(output_dir),
        loss_nodes=loss_node_names,
        metric_nodes=[],
        freeze_nodes=[],
        unfreeze_nodes=unfreeze_node_names,
    )

    trainrun_output_path = results_dir / f"{cfg.name}_trainrun.yaml"
    logger.info(f"Saving trainrun config to: {trainrun_output_path}")
    trainrun_config.save_to_file(str(trainrun_output_path))

    # Report
    logger.info("=== Training Complete ===")
    logger.info(f"Trained pipeline saved: {pipeline_output_path}")
    logger.info(f"TrainRun config saved: {trainrun_output_path}")
    logger.info(f"TensorBoard logs: {tensorboard_node.output_dir}")
    logger.info(f"View logs: uv run tensorboard --logdir={output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(
    config_path="../../configs/",
    config_name="trainrun/channel_selector_false_rgb",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Channel Selector False RGB for Object Tracking — inspect or train mode."""
    mode = cfg.get("mode", "train")
    logger.info(f"=== Channel Selector False RGB — mode={mode} ===")

    if mode == "inspect":
        inspect_dataset(cfg)
    elif mode == "train":
        run_experiment_mixer(cfg)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use mode=inspect or mode=train.")


if __name__ == "__main__":
    main()
