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
import sys
from pathlib import Path

import hydra
from cuvis_ai_core.data.datasets import SingleCu3sDataModule, SingleCu3sDataset
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import GradientTrainer, StatisticalTrainer
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PipelineMetadata
from cuvis_ai_schemas.training import TrainingConfig, TrainRunConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from cuvis_ai.node.anomaly_visualization import (
    ChannelSelectorFalseRGBViz,
    ChannelWeightsViz,
    MaskOverlayNode,
)
from cuvis_ai.node.channel_mixer import LearnableChannelMixer
from cuvis_ai.node.channel_selector import CIETristimulusFalseRGBSelector, NormMode
from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.losses import DistinctnessLoss, ForegroundContrastLoss
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.node.preprocessors import SpatialRotateNode
from cuvis_ai.node.video import ToVideoNode
from cuvis_ai.utils.false_rgb_sampling import initialize_false_rgb_sampled_fixed

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def resolve_spatial_rotation(cfg: DictConfig) -> int:
    """Resolve spatial rotation from config; defaults to 90 for backward compatibility."""
    rotation = int(cfg.get("spatial_rotation", 90))
    if rotation not in {0, 90, 180, 270}:
        raise ValueError(f"Unsupported spatial_rotation={rotation}. Use one of: 0, 90, 180, 270.")
    return rotation


# ---------------------------------------------------------------------------
# Inspect mode
# ---------------------------------------------------------------------------


def inspect_dataset(cfg: DictConfig) -> None:
    """Load CU3S data, discover annotated frames, export false RGB video with mask overlays."""

    output_dir = Path(cfg.output_dir) / "inspect"
    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_rotation = resolve_spatial_rotation(cfg)
    logger.info(f"Using spatial_rotation={spatial_rotation} (inspect mode)")

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
    processing_mode = data_cfg.get("processing_mode", "SpectralRadiance")
    logger.info(f"Processing mode: {processing_mode}")

    # Load only annotated frames via measurement_indices
    dataset = SingleCu3sDataset(
        cu3s_file_path=data_cfg["cu3s_file_path"],
        processing_mode=processing_mode,
        annotation_json_path=annotation_json_path,
        measurement_indices=annotated_frame_ids,
    )

    wl = dataset.wavelengths
    logger.info(f"Available wavelengths ({len(wl)}): {wl.tolist()}")

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
    rotate = SpatialRotateNode(rotation=spatial_rotation, name="spatial_rotate")
    false_rgb = CIETristimulusFalseRGBSelector(
        norm_mode=NormMode.STATISTICAL,
        name="cie_false_rgb",
    )
    sample_positions = initialize_false_rgb_sampled_fixed(
        false_rgb,
        dataset,
        sample_fraction=0.05,
    )
    logger.info(
        "False-RGB sampled-fixed calibration: sample_fraction=0.05, sample_count={}",
        len(sample_positions),
    )
    mask_overlay = MaskOverlayNode(name="mask_overlay")
    to_video = ToVideoNode(
        output_video_path=str(output_dir / "false_rgb.mp4"),
        frame_rate=1,
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
        # Rotate spatial data immediately after loading
        (cu3s_data.outputs.cube, rotate.inputs.cube),
        (cu3s_data.outputs.mask, rotate.inputs.mask),
        # False RGB from rotated cube + wavelengths (bypass rotate)
        (rotate.outputs.cube, false_rgb.cube),
        (cu3s_data.outputs.wavelengths, false_rgb.wavelengths),
        # Mask overlay on RGB before video export
        (false_rgb.rgb_image, mask_overlay.rgb_image),
        (rotate.outputs.mask, mask_overlay.mask),
        (mask_overlay.rgb_with_overlay, to_video.rgb_image),
        # Viz/TensorBoard gets rotated RGB + mask + frame IDs
        (false_rgb.rgb_image, viz.rgb_output),
        (rotate.outputs.mask, viz.mask),
        (cu3s_data.outputs.mesu_index, viz.mesu_index),
        (viz.artifacts, tensorboard_node.artifacts),
    )

    total_frames = len(dataset)
    logger.info(
        f"Exporting false RGB video with mask overlays ({total_frames} annotated frames)..."
    )
    processed = 0
    try:
        for batch in dataloader:
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
    spatial_rotation = resolve_spatial_rotation(cfg)
    mixer_inference_normalization = str(
        cfg.get("mixer_inference_normalization", "batchnorm_sigmoid")
    )
    mixer_use_activation = bool(cfg.get("mixer_use_activation", True))
    mixer_normalize_output = bool(cfg.get("mixer_normalize_output", True))
    post_mixer_normalizer = str(cfg.get("post_mixer_normalizer", "none"))
    if post_mixer_normalizer not in {"none", "minmax_per_frame"}:
        raise ValueError(
            f"Unsupported post_mixer_normalizer={post_mixer_normalizer!r}. "
            "Use one of: none, minmax_per_frame."
        )
    logger.info(f"Using spatial_rotation={spatial_rotation} (train mode)")
    logger.info(
        f"Using mixer_inference_normalization={mixer_inference_normalization} "
        "(applies during inference when normalize_output=True)"
    )
    logger.info(
        f"Using mixer_use_activation={mixer_use_activation}, "
        f"mixer_normalize_output={mixer_normalize_output}, "
        f"post_mixer_normalizer={post_mixer_normalizer}"
    )

    # Setup datamodule
    datamodule = SingleCu3sDataModule(**cfg.data)
    datamodule.setup(stage="fit")

    # Infer spectral channel count from training data to avoid hardcoded mismatches.
    first_batch = next(iter(datamodule.train_dataloader()), None)
    if first_batch is None:
        raise RuntimeError("Training dataloader is empty; cannot infer input channel count.")
    wavelengths = first_batch.get("wavelengths")
    cube = first_batch.get("cube")
    if wavelengths is not None:
        input_channels = int(wavelengths.shape[-1])
    elif cube is not None and cube.ndim >= 2:
        input_channels = int(cube.shape[1] if cube.ndim >= 4 else cube.shape[-1])
    else:
        raise RuntimeError(
            "Could not infer input channel count from training batch (missing wavelengths/cube)."
        )
    logger.info(f"Detected spectral channels for mixer: {input_channels}")

    # Build pipeline
    pipeline = CuvisPipeline("Channel_Selector_FalseRGB")

    data_node = CU3SDataNode(name="cu3s_data")
    rotate = SpatialRotateNode(rotation=spatial_rotation, name="spatial_rotate")
    normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
    mixer = LearnableChannelMixer(
        input_channels=input_channels,
        output_channels=3,
        init_method="pca",
        use_activation=mixer_use_activation,
        normalize_output=mixer_normalize_output,
        inference_normalization=mixer_inference_normalization,
    )
    post_mixer_norm = None
    if post_mixer_normalizer == "minmax_per_frame":
        post_mixer_norm = MinMaxNormalizer(
            eps=1.0e-6,
            use_running_stats=False,
            name="post_mixer_norm",
        )

    contrast_loss = ForegroundContrastLoss(
        name="foreground_contrast",
        weight=1.0,
        compactness_weight=0.1,
        color_space="oklab",
        assume_srgb=False,  # mixer output is linear RGB, no sRGB gamma
    )

    distinctness_loss = DistinctnessLoss(
        name="distinctness",
        weight=0.1,
    )

    viz = ChannelSelectorFalseRGBViz(
        name="false_rgb_viz",
        max_samples=4,
        log_every_n_batches=1,
    )
    weight_viz = ChannelWeightsViz(name="mixer_weights_viz")
    tensorboard_node = TensorBoardMonitorNode(
        output_dir=str(output_dir / "tensorboard"),
        run_name=pipeline.name,
    )

    # Inference-only nodes: skipped during training, active for restore-pipeline
    mask_overlay = MaskOverlayNode(
        name="mask_overlay",
        execution_stages={ExecutionStage.INFERENCE},
    )
    to_video = ToVideoNode(
        output_video_path=str(output_dir / "trained_false_rgb.mp4"),
        frame_rate=10,
        name="to_video",
        execution_stages={ExecutionStage.INFERENCE},
    )

    # Connect pipeline
    connections = [
        # Rotate data immediately after loading
        (data_node.outputs.cube, rotate.inputs.cube),
        (data_node.outputs.mask, rotate.inputs.mask),
        # Processing flow
        (rotate.outputs.cube, normalizer.data),
        (normalizer.normalized, mixer.data),
        # Distinctness loss and weight visualization always use raw mixer weights.
        (mixer.weights, distinctness_loss.selection_weights),
        (mixer.weights, weight_viz.weights),
        (data_node.outputs.wavelengths, weight_viz.wavelengths),
        (weight_viz.artifacts, tensorboard_node.artifacts),
    ]
    rgb_for_loss_and_video = mixer.rgb
    if post_mixer_norm is not None:
        connections.append((mixer.rgb, post_mixer_norm.data))
        rgb_for_loss_and_video = post_mixer_norm.normalized

    connections.extend(
        [
            # Loss flow
            (rgb_for_loss_and_video, contrast_loss.rgb),
            (rotate.outputs.mask, contrast_loss.mask),
            # Visualization flow (mesu_index for frame-identified TensorBoard tags)
            (rgb_for_loss_and_video, viz.rgb_output),
            (rotate.outputs.mask, viz.mask),
            (data_node.outputs.mesu_index, viz.mesu_index),
            (viz.artifacts, tensorboard_node.artifacts),
            # Video export (INFERENCE only — skipped during training)
            (rgb_for_loss_and_video, mask_overlay.rgb_image),
            (rotate.outputs.mask, mask_overlay.mask),
            (mask_overlay.rgb_with_overlay, to_video.rgb_image),
        ]
    )
    pipeline.connect(*connections)

    # Visualize pipeline graph
    pipeline.visualize(
        format="render_graphviz",
        output_path=str(output_dir / "pipeline" / f"{pipeline.name}.png"),
        show_execution_stage=True,
    )
    pipeline.visualize(
        format="render_mermaid",
        output_path=str(output_dir / "pipeline" / f"{pipeline.name}.md"),
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
        scheduler_config=training_cfg.scheduler,
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
        pipeline=pipeline_config.model_dump(),
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
    logger.info(
        "Generate video from trained pipeline:\n"
        f"  uv run restore-pipeline "
        f"--pipeline-path {pipeline_output_path} "
        f"--cu3s-file-path {cfg.data.cu3s_file_path} "
        f"--processing-mode {cfg.data.get('processing_mode', 'SpectralRadiance')} "
        f"--device cuda"
    )


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
    # Suppress DEBUG logs (e.g. TensorBoard monitor) to keep progress bar clean
    logger.remove()
    logger.add(sys.stderr, level="INFO")

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
