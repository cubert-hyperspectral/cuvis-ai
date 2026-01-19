"""Demo: HuggingFace Local Model Integration (Phase 2)

This example demonstrates running HuggingFace models locally for faster,
gradient-capable inference. It builds a CuvisCanvas graph, converts
hyperspectral data to RGB, runs AdaCLIP locally, and logs outputs.

Run:
    python examples/huggingface_local_demo.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import Tensor

from cuvis_ai.data.datasets import SingleCu3sDataModule


from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.adaclip import AdaCLIPLocalNode
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai_core.node import Node
from cuvis_ai.node.visualizations import AnomalyMask
from cuvis_ai_core.pipeline.pipeline import CuvisCanvas
from cuvis_ai_core.pipeline.ports import PortSpec
from cuvis_ai_core.utils.types import Context, ExecutionStage


class HyperspectralToRGBNode(Node):
    """Convert hyperspectral cubes to RGB using simple strategies."""

    INPUT_SPECS = {
        "hsi_cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C]",
        ),
        "wavelengths": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Wavelength array [C]",
            optional=True,
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
        method: str = "wavelength_based",
        wavelengths: Tensor | None = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            method=method,
            wavelengths=wavelengths,  # OK to keep as hparam
            normalize=normalize,
            **kwargs,
        )

        self.method = method
        self.normalize = normalize

        if wavelengths is not None:
            self.wavelengths = (
                wavelengths
                if isinstance(wavelengths, torch.Tensor)
                else torch.as_tensor(wavelengths, dtype=torch.float32)
            )
        else:
            self.wavelengths = None

    def _select_channels_equal_spacing(self, n_channels: int) -> list[int]:
        return [n_channels // 4, n_channels // 2, 3 * n_channels // 4]

    def _select_channels_wavelength(self, wavelengths: Tensor) -> list[int]:
        target_wavelengths = torch.tensor(
            [640.0, 540.0, 450.0], device=wavelengths.device
        )  # R, G, B in nm
        indices: list[int] = []
        for target in target_wavelengths:
            distances = torch.abs(wavelengths - target)
            indices.append(int(torch.argmin(distances).item()))
        logger.info(
            f"Selected wavelengths: "
            f"R={wavelengths[indices[0]]:.1f}nm, "
            f"G={wavelengths[indices[1]]:.1f}nm, "
            f"B={wavelengths[indices[2]]:.1f}nm"
        )
        return indices

    def _select_channels_variance(self, hsi_cube: Tensor) -> list[int]:
        variance = hsi_cube.var(dim=[0, 1, 2])  # [C]
        top_indices = torch.topk(variance, k=3).indices.tolist()
        logger.info(f"Selected high-variance channels: {top_indices}")
        return sorted(top_indices)

    def forward(
        self,
        hsi_cube: Tensor,
        wavelengths: Tensor | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        n_channels = hsi_cube.shape[-1]
        wl = wavelengths if wavelengths is not None else self.wavelengths
        if wl is not None and not isinstance(wl, torch.Tensor):
            wl = torch.as_tensor(wl, dtype=torch.float32, device=hsi_cube.device)
        elif wl is not None:
            wl = wl.to(hsi_cube.device)

        if self.method == "equal_spacing":
            indices = self._select_channels_equal_spacing(n_channels)
        elif self.method == "wavelength_based":
            if wl is None:
                raise ValueError("wavelength_based method requires wavelengths")
            indices = self._select_channels_wavelength(wl)
        elif self.method == "variance_based":
            indices = self._select_channels_variance(hsi_cube)
        elif self.method == "first_three":
            indices = [0, 1, 2]
        else:
            raise ValueError(f"Unknown method: {self.method}")

        rgb = hsi_cube[..., indices]
        if self.normalize:
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

        return {"rgb_image": rgb}

    def serialize(self, serial_dir: str) -> dict:
        return {**self.hparams}

    def load(self, params: dict, serial_dir: str) -> None:
        self.method = params.get("method", self.method)
        self.normalize = params.get("normalize", self.normalize)


def verify_gradient_passthrough(model: AdaCLIPLocalNode, image: Tensor) -> bool:
    """Check that gradients reach the input while the HF model stays frozen."""
    image = image.detach().requires_grad_(True)
    outputs = model.forward(image=image)
    scores = outputs["anomaly_scores"]

    loss = scores.mean()
    loss.backward()

    input_has_grad = image.grad is not None
    model_has_grad = any(p.grad is not None for p in model.model.parameters() if p.requires_grad)

    logger.info(f"Input gradient present: {input_has_grad}")
    logger.info(f"Model gradient present: {model_has_grad}")

    if input_has_grad and not model_has_grad:
        logger.success("Gradient passthrough verified (inputs only).")
        return True

    logger.warning("Unexpected gradient configuration detected.")
    return False


def _move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def main() -> None:
    logger.info("=== HuggingFace Local Model Demo (Phase 2) ===")

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
    wavelengths = datamodule.val_ds.wavelengths
    logger.info(f"Wavelengths loaded: {len(wavelengths)} channels.")

    # Step 2: Pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    pipeline = CuvisCanvas("AdaCLIP_Local_Demo")

    data_node = LentilsAnomalyDataNode(
        normal_class_ids=[0, 1],
        method="wavelength_based",
    )
    rgb_converter = HyperspectralToRGBNode(
        method="wavelength_based",
        wavelengths=torch.tensor(wavelengths, dtype=torch.float32),
        normalize=True,
        name="rgb_converter",
    )
    adaclip_local = AdaCLIPLocalNode(
        model_name="openai/clip-vit-base-patch32",  # Fallback if AdaCLIP is unavailable
        cache_dir=None,
        text_prompt="normal: lentils, anomaly: stones",
        name="adaclip_local",
    )
    viz_mask = AnomalyMask(channel=30, up_to=5, name="viz_mask")
    tensorboard_node = TensorBoardMonitorNode(
        run_name="adaclip_local_demo",
        output_dir="./outputs/",
        name="tensorboard_monitor",
    )

    pipeline.connect(
        (data_node.outputs.cube, rgb_converter.hsi_cube),
        (rgb_converter.rgb_image, adaclip_local.image),
        (adaclip_local.anomaly_mask, viz_mask.decisions),
        (adaclip_local.anomaly_scores, viz_mask.scores),
        (data_node.outputs.mask, viz_mask.mask),
        (data_node.outputs.cube, viz_mask.cube),
        (viz_mask.artifacts, tensorboard_node.artifacts),
    )

    pipeline.to(device)
    pipeline.visualize(
        format="render_graphviz",
        output_path="outputs/canvases/adaclip_local_demo.png",
        show_execution_stage=True,
    )
    pipeline.visualize(
        format="render_mermaid",
        output_path="outputs/canvases/adaclip_local_demo.md",
        direction="LR",
        include_node_class=True,
        wrap_markdown=True,
        show_execution_stage=True,
    )
    logger.success("Pipeline ready.")

    # Step 3: Run inference
    val_loader = datamodule.val_dataloader()
    context = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)
    total_time = 0.0
    total_batches = 0
    total_anomalous = 0
    total_pixels = 0

    for batch_idx, batch in enumerate(val_loader):
        batch = _move_batch_to_device(batch, device)
        context.batch_idx = batch_idx
        start = time.time()
        outputs = pipeline.forward(batch, context=context)
        batch_time = time.time() - start

        anomaly_mask = outputs[(adaclip_local.name, "anomaly_mask")]
        total_time += batch_time
        total_batches += 1
        total_anomalous += anomaly_mask.sum().item()
        total_pixels += anomaly_mask.numel()

        logger.info(
            f"Batch {batch_idx + 1}: {batch_time:.3f}s, "
            f"Anomalies: {anomaly_mask.sum().item()}/{anomaly_mask.numel()}"
        )

    avg_time = total_time / max(total_batches, 1)
    pct_anomalous = 100 * total_anomalous / max(total_pixels, 1)

    logger.success("=== Local Inference Complete ===")
    logger.info(f"Total batches: {total_batches}")
    logger.info(f"Average time per batch: {avg_time:.3f}s")
    logger.info(f"Anomaly percentage: {pct_anomalous:.2f}%")

    # Step 4: Gradient passthrough check
    sample_batch = _move_batch_to_device(next(iter(val_loader)), device)
    sample_outputs = pipeline.forward(sample_batch, context=context)
    sample_rgb = sample_outputs[(rgb_converter.name, "rgb_image")]
    if verify_gradient_passthrough(adaclip_local, sample_rgb):
        logger.success("Ready for gradient-based Phase 3.")

    logger.info("Outputs:")
    logger.info("  - Pipeline visualization: outputs/canvases/adaclip_local_demo.png")
    logger.info("  - TensorBoard logs: outputs/adaclip_local_demo_runs/")
    logger.info("View logs with: uv run tensorboard --logdir=./outputs/adaclip_local_demo_runs")


if __name__ == "__main__":
    main()
