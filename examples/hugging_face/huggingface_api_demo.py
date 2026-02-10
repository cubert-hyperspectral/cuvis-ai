"""Demo: HuggingFace API Integration for Anomaly Detection

This example demonstrates Phase 1 of HuggingFace integration using API calls
to the AdaCLIP Space for anomaly detection on hyperspectral data.

Features:
- Load hyperspectral data
- Convert to RGB using simple band selection (SimpleRGBConverter node)
- Call AdaCLIP via HuggingFace Spaces API (AdaCLIPAPINode)
- Visualize anomaly detection results

This demonstrates proper pipeline node connections and execution flow.

Note: This uses API backend (non-differentiable). For gradient training,
see examples/huggingface_gradient_training.py (Phase 3).

Run:
    python examples/huggingface_api_demo.py

Graph Visualization:
```mermaid
graph LR
    Data[Data Node<br/>Lentils Cubes] --> RGB[RGB<br/>Converter]
    RGB --> API[AdaCLIP<br/>API Node]
    API --> Viz[Anomaly<br/>Mask Viz]
    Data --> |mask| Viz
    Data --> |cube| Viz
    Viz --> TB[TensorBoard<br/>Monitor]

    style Data fill:#e1f5ff
    style RGB fill:#fff4e1
    style API fill:#ffe1f5
    style Viz fill:#ffe1e1
    style TB fill:#e1e1ff
```
"""

from typing import Any

import torch
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisCanvas
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger
from torch import Tensor

from cuvis_ai.node.adaclip import AdaCLIPAPINode
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.visualizations import AnomalyMask


class SimpleRGBConverter(Node):
    """Simple RGB conversion node for hyperspectral data.

    This is a temporary solution for Phase 1 demo. In Phase 3, this will be
    replaced by the full HyperspectralToRGB node with multiple conversion methods.

    Converts hyperspectral cube to RGB by selecting 3 equally-spaced channels.

    Parameters
    ----------
    method : str, optional
        Conversion method (default: "equal_spacing")
        - "equal_spacing": Select 3 equally-spaced channels as R, G, B
        - "first_three": Use first 3 channels
    **kwargs
        Additional arguments passed to Node base class
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

    def __init__(self, method: str = "equal_spacing", **kwargs) -> None:
        self.method = method
        super().__init__(method=method, **kwargs)

    def forward(self, hsi_cube: Tensor, context: Context) -> dict[str, Any]:
        """Convert hyperspectral cube to RGB.

        Parameters
        ----------
        hsi_cube : Tensor
            Hyperspectral cube [B, H, W, C]
        context : Context
            Execution context

        Returns
        -------
        dict[str, Any]
            Dictionary with "rgb_image" key containing RGB tensor [B, H, W, 3]
        """
        n_channels = hsi_cube.shape[-1]

        if self.method == "equal_spacing":
            # Select 3 equally spaced channels as R, G, B
            indices = [
                n_channels // 4,  # Red-ish
                n_channels // 2,  # Green-ish
                3 * n_channels // 4,  # Blue-ish
            ]
            rgb = hsi_cube[..., indices]
        elif self.method == "first_three":
            # Simply use first 3 channels
            rgb = hsi_cube[..., :3]
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Normalize to [0, 1] if needed
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

        return {"rgb_image": rgb}


def main() -> None:
    """Run AdaCLIP API demo with proper pipeline connections."""
    logger.info("=== HuggingFace API Integration Demo (Phase 1) ===")

    # Step 1: Setup data
    logger.info("Loading hyperspectral data...")
    datamodule = SingleCu3sDataModule(
        data_dir="../data/Lentils",
        dataset_name="Lentils",
        batch_size=2,  # Small batch for demo
        train_ids=[0, 1],  # Minimal training data (not used in demo)
        val_ids=[3, 4],  # Validation data for demo
    )
    datamodule.setup(stage="fit")

    # Get wavelengths
    wavelengths = datamodule.val_ds.wavelengths
    logger.info(f"Wavelengths: {len(wavelengths)} channels")
    logger.info(f"Range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")

    # Step 2: Build pipeline with proper node connections
    logger.info("Building pipeline with node connections...")
    pipeline = CuvisCanvas("AdaCLIP_API_Demo")

    # Create nodes
    data_node = LentilsAnomalyDataNode(
        normal_class_ids=[0, 1],
        name="data_node",
    )

    rgb_converter = SimpleRGBConverter(
        method="equal_spacing",
        name="rgb_converter",
    )

    adaclip_api = AdaCLIPAPINode(
        space_url="Caoyunkang/AdaCLIP",
        use_hf_token=True,
        default_text_prompt="defect",
        name="adaclip_api",
    )

    viz_mask = AnomalyMask(
        channel=30,
        up_to=5,
        name="viz_mask",
    )

    tensorboard_node = TensorBoardMonitorNode(
        run_name="adaclip_api_demo",
        output_dir="./outputs/",
        name="tensorboard_monitor",
    )

    # Connect nodes
    logger.info("Connecting nodes in pipeline...")
    pipeline.connect(
        # Processing flow: Data -> RGB Converter -> AdaCLIP API
        (data_node.outputs.cube, rgb_converter.hsi_cube),
        (rgb_converter.rgb_image, adaclip_api.image),
        # Visualization flow
        (adaclip_api.anomaly_mask, viz_mask.decisions),
        (data_node.outputs.mask, viz_mask.mask),
        (data_node.outputs.cube, viz_mask.cube),
        (viz_mask.artifacts, tensorboard_node.artifacts),
    )

    # Visualize pipeline structure
    logger.info("Generating pipeline visualizations...")
    pipeline.visualize(
        format="render_graphviz",
        output_path="outputs/canvases/adaclip_api_demo.png",
        show_execution_stage=True,
    )
    pipeline.visualize(
        format="render_mermaid",
        output_path="outputs/canvases/adaclip_api_demo.md",
        direction="LR",
        include_node_class=True,
        wrap_markdown=True,
        show_execution_stage=True,
    )
    logger.success("Pipeline built and visualized")

    # Step 3: Run inference using pipeline
    logger.info("Running inference through pipeline...")

    # Get one batch
    val_loader = datamodule.val_dataloader()
    batch = next(iter(val_loader))

    logger.info(f"Input shape: {batch['cube'].shape}")

    # Execute pipeline with proper context
    logger.info("Calling AdaCLIP API via pipeline (this may take a few seconds)...")
    from cuvis_ai_schemas.enums import ExecutionStage

    context = Context(
        stage=ExecutionStage.VAL,
        epoch=0,
        batch_idx=0,
    )

    outputs = pipeline.forward(batch, context=context)

    # Extract results
    anomaly_mask = outputs[(adaclip_api.name, "anomaly_mask")]
    logger.success(f"API call successful! Mask shape: {anomaly_mask.shape}")

    # Basic statistics
    n_anomalous = anomaly_mask.sum().item()
    n_total = anomaly_mask.numel()
    pct_anomalous = 100 * n_anomalous / n_total

    logger.info("Anomaly detection results:")
    logger.info(f"  - Total pixels: {n_total}")
    logger.info(f"  - Anomalous pixels: {n_anomalous}")
    logger.info(f"  - Anomaly percentage: {pct_anomalous:.2f}%")

    # Compare with ground truth if available
    gt_mask = batch.get("mask")
    if gt_mask is not None:
        gt_anomalous = gt_mask.sum().item()
        logger.info(f"Ground truth: {gt_anomalous} anomalous pixels")

    logger.success("Demo completed successfully!")
    logger.info("\nGenerated outputs:")
    logger.info("  - Pipeline visualization: outputs/canvases/adaclip_api_demo.png")
    logger.info("  - TensorBoard logs: outputs/adaclip_api_demo_runs/")
    logger.info("\nView logs: uv run tensorboard --logdir=./outputs/adaclip_api_demo_runs")


if __name__ == "__main__":
    main()
