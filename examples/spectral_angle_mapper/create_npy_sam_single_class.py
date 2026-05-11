"""Create 3 single-class NpyReader+SAM pipelines from existing StatefulSAM weights.

Reads the learned signature from each of the 3 StatefulSpectralAngleMapper .pt
files and bakes it into a generic NpyReader (buffer-mode) + SpectralAngleMapper
pipeline.  Also creates thresholded variants with ScoreToLogit + BinaryDecider.

Output directory: configs/pipeline/sam/medical_npy_sam_single_class/

Base pipelines (3):
    npy_sam_class_{1,2,3}.yaml + .pt

Thresholded pipelines (3 classes × 5 thresholds = 15):
    npy_sam_class_{1,2,3}_thr_0p{10,15,20,25,30}.yaml + .pt

Usage::

    uv run python examples/spectral_angle_mapper/create_npy_sam_single_class.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.pipeline import PipelineMetadata
from loguru import logger

from cuvis_ai.deciders.binary_decider import BinaryDecider
from cuvis_ai.node.conversion import ScoreToLogit
from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.numpy_reader import NpyReader
from cuvis_ai.node.spectral_angle_mapper import SpectralAngleMapper

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STATEFUL_DIR = Path("configs/pipeline/sam/medical_stateful_reflectance_updated")
OUTPUT_DIR = Path("configs/pipeline/sam/medical_npy_sam_single_class")

NUM_CHANNELS = 39
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_signature(class_idx: int) -> torch.Tensor:
    """Load learned_signature [1,1,1,39] from the StatefulSAM .pt file."""
    pt_path = STATEFUL_DIR / f"stateful_sam_class_{class_idx}_{class_idx}.pt"
    state = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    sig = state["state_dict"]["stateful_sam"]["learned_signature"]  # [1,1,1,39]
    logger.info(
        "  Loaded signature class {}: shape={}, range=[{:.4f}, {:.4f}]",
        class_idx,
        list(sig.shape),
        float(sig.min()),
        float(sig.max()),
    )
    return sig


def _build_base_pipeline(class_idx: int, sig: torch.Tensor) -> tuple[Path, Path]:
    """Create base NpyReader+SAM pipeline (no threshold) and save."""
    name = f"npy_sam_class_{class_idx}"
    pipeline = CuvisPipeline(name=name)

    cu3s_node = CU3SDataNode(name="cu3s_data")
    ref_node = NpyReader(file_path=None, name="reference_signature")
    sam_node = SpectralAngleMapper(num_channels=NUM_CHANNELS, name="sam")

    # Populate buffer — sig is already [1,1,1,39], NpyReader keeps it as-is
    ref_node.load_from_array(sig)

    pipeline.connect((cu3s_node.outputs.cube, sam_node.cube))
    pipeline.connect((ref_node.outputs.data, sam_node.spectral_signature))

    yaml_path = OUTPUT_DIR / f"{name}.yaml"
    pipeline.save_to_file(
        str(yaml_path),
        metadata=PipelineMetadata(
            name=name,
            description=(
                f"Single-class SAM pipeline for class {class_idx}. "
                "Reference signature stored in NpyReader buffer (file_path=null). "
                "Lower best_scores value = better match."
            ),
            tags=["sam", "statistical", "material_detection", "npy_reader", "single_class"],
            author="cuvis.ai",
        ),
    )
    pt_path = yaml_path.with_suffix(".pt")
    logger.success("  Saved base pipeline: {}", yaml_path.name)
    return yaml_path, pt_path


def _build_thresholded_pipeline(
    class_idx: int, sig: torch.Tensor, threshold: float
) -> tuple[Path, Path]:
    """Create NpyReader+SAM+ScoreToLogit+BinaryDecider pipeline and save."""
    thr_str = f"0p{int(threshold * 100):02d}"
    name = f"npy_sam_class_{class_idx}_thr_{thr_str}"
    pipeline = CuvisPipeline(name=name)

    cu3s_node = CU3SDataNode(name="cu3s_data")
    ref_node = NpyReader(file_path=None, name="reference_signature")
    sam_node = SpectralAngleMapper(num_channels=NUM_CHANNELS, name="sam")
    s2l_node = ScoreToLogit(init_scale=-1.0, init_bias=threshold, name="score_to_logit")
    dec_node = BinaryDecider(threshold=0.5, name="decider")

    ref_node.load_from_array(sig)

    pipeline.connect((cu3s_node.outputs.cube, sam_node.cube))
    pipeline.connect((ref_node.outputs.data, sam_node.spectral_signature))
    pipeline.connect((sam_node.outputs.best_scores, s2l_node.scores))
    pipeline.connect((s2l_node.outputs.logits, dec_node.logits))

    yaml_path = OUTPUT_DIR / f"{name}.yaml"
    pipeline.save_to_file(
        str(yaml_path),
        metadata=PipelineMetadata(
            name=name,
            description=(
                f"Single-class SAM pipeline for class {class_idx} "
                f"with absolute angle threshold {threshold:.2f} rad "
                f"({threshold * 57.296:.1f} deg). "
                "Pixels with angle < threshold → True (match)."
            ),
            tags=[
                "sam",
                "statistical",
                "material_detection",
                "npy_reader",
                "single_class",
                "thresholded",
            ],
            author="cuvis.ai",
        ),
    )
    pt_path = yaml_path.with_suffix(".pt")
    logger.success("  Saved threshold={:.2f} pipeline: {}", threshold, yaml_path.name)
    return yaml_path, pt_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: {}", OUTPUT_DIR.resolve())

    for class_idx in [1, 2, 3]:
        logger.info("=== Class {} ===", class_idx)

        sig = _load_signature(class_idx)

        # Base pipeline
        _build_base_pipeline(class_idx, sig)

        # Thresholded variants
        for thr in THRESHOLDS:
            _build_thresholded_pipeline(class_idx, sig, thr)

    logger.success(
        "Done. Created {} files in {}",
        len(list(OUTPUT_DIR.glob("*.yaml"))),
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
