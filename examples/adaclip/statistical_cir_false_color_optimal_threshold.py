"""Statistical AdaCLIP CIR false-color example with optimal threshold.

Run this script to generate reusable pipeline and trainrun artifacts under:
`outputs/adaclip_cir_false_color_optimal_threshold/trained_models/`

Shipping to `configs/` is an explicit follow-up copy step.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import StatisticalTrainer
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.pipeline import PipelineMetadata
from cuvis_ai_schemas.training import (
    TrainingConfig,
    TrainRunConfig,
)
from loguru import logger

from cuvis_ai.deciders.two_stage_decider import TwoStageBinaryDecider
from cuvis_ai.node.anomaly_visualization import RGBAnomalyMask, ScoreHeatmapVisualizer
from cuvis_ai.node.channel_selector import CIRSelector
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.monitor import TensorBoardMonitorNode


def _parse_ids(csv: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(chunk.strip()) for chunk in csv.split(",") if chunk.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run AdaCLIP CIR false-color (optimal threshold) and produce reusable artifacts."
    )

    parser.add_argument(
        "--plugins-manifest",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "configs" / "plugins" / "adaclip.yaml"),
        help="Path to plugin manifest YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/adaclip_cir_false_color_optimal_threshold",
        help="Output directory for run artifacts.",
    )

    parser.add_argument(
        "--cu3s-file-path",
        type=str,
        default="data/Lentils/Lentils_000.cu3s",
        help="Path to CU3S file.",
    )
    parser.add_argument(
        "--annotation-json-path",
        type=str,
        default="data/Lentils/Lentils_000.json",
        help="Path to annotation JSON.",
    )
    parser.add_argument("--train-ids", type=str, default="0,2,3", help="Comma-separated train IDs.")
    parser.add_argument(
        "--val-ids", type=str, default="1,5", help="Comma-separated validation IDs."
    )
    parser.add_argument("--test-ids", type=str, default="1,5", help="Comma-separated test IDs.")
    parser.add_argument("--batch-size", type=int, default=1, help="DataLoader batch size.")
    parser.add_argument(
        "--processing-mode",
        type=str,
        default="Reflectance",
        choices=["Raw", "Reflectance"],
        help="CUVIS processing mode.",
    )

    parser.add_argument(
        "--weight-name",
        type=str,
        default="pretrained_all",
        help="AdaCLIP weight name.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="ViT-L-14-336",
        help="AdaCLIP backbone name.",
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="",
        help="Prompt text used by AdaCLIP.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=4.0,
        help="Gaussian sigma for AdaCLIP score smoothing.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=518,
        help="AdaCLIP image size.",
    )
    parser.add_argument(
        "--use-half-precision",
        action="store_true",
        help="Enable FP16 inference for AdaCLIP.",
    )
    parser.add_argument(
        "--enable-warmup",
        action="store_true",
        help="Enable one-time warmup inference for AdaCLIP.",
    )
    parser.add_argument(
        "--use-torch-preprocess",
        dest="use_torch_preprocess",
        action="store_true",
        help="Use tensor preprocessing (default).",
    )
    parser.add_argument(
        "--no-use-torch-preprocess",
        dest="use_torch_preprocess",
        action="store_false",
        help="Use PIL preprocessing instead of tensor preprocessing.",
    )
    parser.set_defaults(use_torch_preprocess=True)

    parser.add_argument(
        "--image-threshold",
        type=float,
        default=0.0847439244389534,
        help="Image-level top-k gate threshold (optimal F1 default).",
    )
    parser.add_argument(
        "--top-k-fraction",
        type=float,
        default=0.001,
        help="Fraction of top pixels for image-level gate.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.995,
        help="Pixel-level quantile threshold for two-stage decider.",
    )

    parser.add_argument("--nir-nm", type=float, default=860.0, help="NIR wavelength in nm.")
    parser.add_argument("--red-nm", type=float, default=670.0, help="Red wavelength in nm.")
    parser.add_argument("--green-nm", type=float, default=560.0, help="Green wavelength in nm.")
    parser.add_argument(
        "--visualize-upto",
        type=int,
        default=3,
        help="Maximum number of visualized samples per batch.",
    )

    return parser


def main() -> None:
    """Entry point."""
    args = _build_arg_parser().parse_args()
    run_start = time.perf_counter()

    logger.info("=== AdaCLIP CIR False Color (Optimal Threshold) ===")

    plugins_manifest = Path(args.plugins_manifest)
    logger.info(f"Loading AdaCLIP plugin from manifest: {plugins_manifest}")

    registry = NodeRegistry()
    registry.load_plugins(plugins_manifest)
    if "adaclip" not in registry.plugin_configs:
        raise KeyError("Plugin 'adaclip' not found in configs/plugins/adaclip.yaml")
    AdaCLIPDetector = NodeRegistry.get("cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector")
    logger.info("✓ AdaCLIP plugin loaded successfully")

    output_dir = Path(args.output_dir)
    data_config = {
        "cu3s_file_path": args.cu3s_file_path,
        "annotation_json_path": args.annotation_json_path,
        "train_ids": _parse_ids(args.train_ids),
        "val_ids": _parse_ids(args.val_ids),
        "test_ids": _parse_ids(args.test_ids),
        "batch_size": args.batch_size,
        "processing_mode": args.processing_mode,
    }

    datamodule = SingleCu3sDataModule(**data_config)
    datamodule.setup(stage=None)

    wavelengths = getattr(datamodule.train_ds, "wavelengths_nm", None)
    if wavelengths is not None:
        logger.info(
            f"Wavelength range: {float(wavelengths.min()):.1f}-{float(wavelengths.max()):.1f} nm"
        )

    logger.info(
        "Splits: train={}, val={}, test={}",
        data_config["train_ids"],
        data_config["val_ids"],
        data_config["test_ids"],
    )
    logger.info("Model: {} | Weights: {}", args.backbone, args.weight_name)
    logger.info("Prompt: {}", args.prompt_text)
    logger.info(
        "CIR wavelengths: NIR={:.1f} nm, Red={:.1f} nm, Green={:.1f} nm",
        args.nir_nm,
        args.red_nm,
        args.green_nm,
    )
    logger.info("Image-level threshold: {:.6f}", args.image_threshold)
    logger.info("Top-k fraction: {}", args.top_k_fraction)
    logger.info("Quantile: {}", args.quantile)

    pipeline = CuvisPipeline("AdaCLIP_CIR_FalseColor_OptimalThreshold")

    data_node = LentilsAnomalyDataNode(normal_class_ids=[0, 1])
    band_selector = CIRSelector(
        nir_nm=args.nir_nm,
        red_nm=args.red_nm,
        green_nm=args.green_nm,
    )
    adaclip = AdaCLIPDetector(
        weight_name=args.weight_name,
        backbone=args.backbone,
        prompt_text=args.prompt_text,
        image_size=args.image_size,
        gaussian_sigma=args.gaussian_sigma,
        use_half_precision=args.use_half_precision,
        enable_warmup=args.enable_warmup,
        use_torch_preprocess=args.use_torch_preprocess,
    )
    decider = TwoStageBinaryDecider(
        image_threshold=args.image_threshold,
        top_k_fraction=args.top_k_fraction,
        quantile=args.quantile,
    )
    metrics_node = AnomalyDetectionMetrics(name="detection_metrics")
    score_viz = ScoreHeatmapVisualizer(normalize_scores=True, up_to=args.visualize_upto)
    mask_viz = RGBAnomalyMask(up_to=args.visualize_upto)
    monitor = TensorBoardMonitorNode(
        output_dir=str(output_dir / "tensorboard"),
        run_name=pipeline.name,
    )

    pipeline.connect(
        (data_node.outputs.cube, band_selector.inputs.cube),
        (data_node.outputs.wavelengths, band_selector.inputs.wavelengths),
        (band_selector.outputs.rgb_image, adaclip.inputs.rgb_image),
        (adaclip.outputs.scores, decider.inputs.logits),
        (adaclip.outputs.scores, score_viz.inputs.scores),
        (adaclip.outputs.scores, mask_viz.inputs.scores),
        (decider.outputs.decisions, metrics_node.inputs.decisions),
        (data_node.outputs.mask, metrics_node.inputs.targets),
        (decider.outputs.decisions, mask_viz.inputs.decisions),
        (data_node.outputs.mask, mask_viz.inputs.mask),
        (band_selector.outputs.rgb_image, mask_viz.inputs.rgb_image),
        (metrics_node.outputs.metrics, monitor.inputs.metrics),
        (score_viz.outputs.artifacts, monitor.inputs.artifacts),
        (mask_viz.outputs.artifacts, monitor.inputs.artifacts),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Moving pipeline to device: {device}")
    pipeline.to(device)

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

    trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)

    if data_config["val_ids"]:
        logger.info("Running validation...")
        trainer.validate()
    else:
        logger.info("Skipping validation (no val_ids provided)")

    if data_config["test_ids"]:
        logger.info("Running test...")
        trainer.test()
    else:
        logger.info("Skipping test (no test_ids provided)")

    results_dir = output_dir / "trained_models"
    results_dir.mkdir(parents=True, exist_ok=True)

    pipeline_output_path = results_dir / f"{pipeline.name}.yaml"
    trainrun_output_path = results_dir / "adaclip_cir_false_color_optimal_threshold_trainrun.yaml"

    metadata = PipelineMetadata(
        name=pipeline.name,
        description=(
            "Statistical AdaCLIP CIR false-color pipeline with optimal threshold "
            "(LentilsAnomalyDataNode -> CIRSelector -> AdaCLIPDetector -> TwoStageBinaryDecider)"
        ),
        tags=["statistical", "adaclip", "cir_false_color", "optimal_threshold", "two_stage"],
        author="cuvis.ai",
    )
    pipeline.save_to_file(str(pipeline_output_path), metadata=metadata)
    logger.info(f"Saved pipeline: {pipeline_output_path}")
    logger.info(f"Saved weights: {pipeline_output_path.with_suffix('.pt')}")

    trainrun_config = TrainRunConfig(
        name="adaclip_cir_false_color_optimal_threshold",
        pipeline=pipeline.serialize(),
        data=data_config,
        training=TrainingConfig(seed=42),
        output_dir=str(output_dir),
        loss_nodes=[],
        metric_nodes=["detection_metrics"],
        freeze_nodes=[],
        unfreeze_nodes=[],
    )
    trainrun_config.save_to_file(str(trainrun_output_path))
    logger.info(f"Saved trainrun: {trainrun_output_path}")

    target_pipeline = Path(
        "configs/pipeline/anomaly/adaclip/adaclip_cir_false_color_optimal_threshold.yaml"
    )
    target_trainrun = Path("configs/trainrun/adaclip_cir_false_color_optimal_threshold.yaml")
    logger.info("Shipping step (explicit copy):")
    logger.info(f'  copy /Y "{pipeline_output_path}" "{target_pipeline}"')
    logger.info(f'  copy /Y "{trainrun_output_path}" "{target_trainrun}"')

    total_duration = time.perf_counter() - run_start
    logger.info(f"Complete in {total_duration:.2f}s")


if __name__ == "__main__":
    main()
