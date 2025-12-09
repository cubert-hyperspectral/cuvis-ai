"""Restore trained pipeline for inference without requiring full experiment config."""

from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from cuvis_ai.data.datasets import SingleCu3sDataset
from cuvis_ai.pipeline.pipeline import CuvisPipeline
from cuvis_ai.utils.types import Context, ExecutionStage


def restore_pipeline(
    pipeline_path: str | Path,
    weights_path: str | Path | None = None,
    device: str = "auto",
    cu3s_file_path: str | Path | None = None,
    processing_mode: str = "Reflectance",
    config_overrides: list[str] | None = None,
) -> CuvisPipeline:
    """Restore pipeline from configuration and weights for inference.

    Parameters
    ----------
    pipeline_path : str | Path
        Path to pipeline YAML configuration file
    weights_path : str | Path | None
        Optional path to weights file (.pt). If None, defaults to pipeline_path with .pt extension
    device : str
        Device to load weights to ('cpu', 'cuda', 'auto')
    cu3s_file_path : str | Path | None
        Optional path to .cu3s file for inference
    processing_mode : str
        Cuvis processing mode string ("Raw", "Reflectance")
    config_overrides : list[str] | None
        Optional list of config overrides in dot notation (e.g., ["nodes.10.params.output_dir=outputs/my_tb"])

    Returns
    -------
    CuvisPipeline
        Loaded pipeline ready for inference
    """
    pipeline_path = Path(pipeline_path)
    if weights_path is None:
        weights_path = pipeline_path.with_suffix(".pt")
    else:
        weights_path = Path(weights_path)

    logger.info(f"Loading pipeline from {pipeline_path}")

    load_device = device if device != "auto" else None
    pipeline = CuvisPipeline.load_from_file(
        str(pipeline_path),
        weights_path=str(weights_path) if weights_path.exists() else None,
        device=load_device,
        config_overrides=config_overrides,
    )

    # If cu3s_file_path provided, setup data and run inference
    if cu3s_file_path:
        data = SingleCu3sDataset(
            cu3s_file_path=str(cu3s_file_path),
            processing_mode=processing_mode,
        )
        dataloader = DataLoader(data, shuffle=False, batch_size=1)

        for module in pipeline.torch_layers:
            module.eval()

        # Process all batches
        results = []
        global_step = 0  # Track step across batches
        with torch.no_grad():
            for batch in dataloader:
                # Create context with incrementing step
                context = Context(
                    stage=ExecutionStage.INFERENCE, batch_idx=global_step, global_step=global_step
                )
                outputs = pipeline.forward(batch=batch, context=context)
                results.append(outputs)
                global_step += 1  # Increment for next batch

        logger.info(f"Processed {len(results)} measurements")

    else:
        # Just display input/output specs
        input_specs = pipeline.get_input_specs()
        output_specs = pipeline.get_output_specs()

        print("\nInput Specs:")
        for name, spec in input_specs.items():
            print(f"  {name}: {spec}")

        print("\nOutput Specs:")
        for name, spec in output_specs.items():
            print(f"  {name}: {spec}")

    logger.info("Pipeline ready for inference")
    return pipeline


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Restore trained pipeline for inference")
    parser.add_argument("--pipeline-path", type=str, required=True, help="Path to pipeline YAML")
    parser.add_argument("--weights-path", type=str, default=None, help="Path to weights (.pt)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cu3s-file-path", type=str, default=None, help="Path to .cu3s file")
    parser.add_argument(
        "--processing-mode", type=str, default="Reflectance", help="Processing mode"
    )
    parser.add_argument(
        "--override",
        action="append",
        help="Override config values in dot notation (e.g., nodes.10.params.output_dir=outputs/my_tb). Can be specified multiple times.",
    )

    args = parser.parse_args()

    restore_pipeline(
        pipeline_path=args.pipeline_path,
        weights_path=args.weights_path,
        device=args.device,
        cu3s_file_path=args.cu3s_file_path,
        processing_mode=args.processing_mode,
        config_overrides=args.override,
    )


if __name__ == "__main__":
    main()
