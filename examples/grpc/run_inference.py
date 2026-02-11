"""Restore trained pipeline for inference using gRPC.

gRPC equivalent of the serialization restore_pipeline.py script.
This script demonstrates how to restore a trained pipeline from YAML configuration
and weights files using the gRPC API, and optionally run inference on CU3S data.
"""

from __future__ import annotations

from pathlib import Path

from cuvis_ai.data.datasets import SingleCu3sDataset
from cuvis_ai_core.grpc import helpers
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
from loguru import logger
from torch.utils.data import DataLoader
from workflow_utils import (
    build_stub,
    config_search_paths,
    create_session_with_search_paths,
)


def run_inference(
    pipeline_path: str | Path,
    weights_path: str | Path,
    cu3s_file_path: str | Path,
    server_address: str = "localhost:50051",
    device: str = "auto",
    config_overrides: list[str] | None = None,
    processing_mode: str = "Reflectance",
) -> None:
    """Restore pipeline from configuration and weights for inference using gRPC.

    Parameters
    ----------
    pipeline_path : str | Path
        Path to pipeline YAML configuration file
    weights_path : str | Path | None
        Optional path to weights file (.pt). If None, defaults to pipeline_path with .pt extension
    server_address : str
        gRPC server address (default: localhost:50051)
    device : str
        Device to load weights to ('cpu', 'cuda', 'auto')
    config_overrides : list[str] | None
        Optional list of config overrides in dot notation (e.g., ["nodes.10.params.output_dir=outputs/my_tb"])
    cu3s_file_path : str | Path | None
        Optional path to .cu3s file for inference
    processing_mode : str
        Cuvis processing mode string ("Raw", "Reflectance")
    """
    pipeline_path = Path(pipeline_path)

    logger.info(f"Connecting to gRPC server at {server_address}")
    # Use larger message size to handle big hyperspectral data (300MB should be sufficient)
    stub = build_stub(server_address, max_msg_size=600 * 1024 * 1024)

    logger.info("Creating session and setting search paths")
    session_id = create_session_with_search_paths(stub, config_search_paths())

    logger.info(f"Loading pipeline from {pipeline_path}")

    # Resolve pipeline config via ConfigService
    pipeline_config = stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="pipeline",
            path=str(pipeline_path),
            overrides=config_overrides or [],
        )
    )

    # Build the pipeline structure
    stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=pipeline_config.config_bytes),
        )
    )

    # Load weights if available
    logger.info(f"Loading weights from {weights_path}")
    stub.LoadPipelineWeights(
        cuvis_ai_pb2.LoadPipelineWeightsRequest(
            session_id=session_id,
            weights_path=str(weights_path),
            strict=True,
        )
    )

    # Get pipeline input/output specs
    inputs_response = stub.GetPipelineInputs(
        cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
    )
    outputs_response = stub.GetPipelineOutputs(
        cuvis_ai_pb2.GetPipelineOutputsRequest(session_id=session_id)
    )

    print("\nInput Specs:")
    for name, spec in inputs_response.input_specs.items():
        shape_str = "x".join(str(dim) for dim in spec.shape)
        print(
            f"  {name}: shape=[{shape_str}], dtype={cuvis_ai_pb2.DType.Name(spec.dtype)}, required={spec.required}"
        )

    print("\nOutput Specs:")
    for name, spec in outputs_response.output_specs.items():
        shape_str = "x".join(str(dim) for dim in spec.shape)
        print(
            f"  {name}: shape=[{shape_str}], dtype={cuvis_ai_pb2.DType.Name(spec.dtype)}, required={spec.required}"
        )

    logger.info("Pipeline ready for inference via gRPC")

    # If cu3s_file_path provided, setup data and run inference
    logger.info(f"Loading CU3S data from {cu3s_file_path}")
    data = SingleCu3sDataset(
        cu3s_file_path=str(cu3s_file_path),
        processing_mode=processing_mode,
    )
    dataloader = DataLoader(data, shuffle=False, batch_size=1)

    logger.info("Running inference on CU3S data")
    results = []

    # Process data in smaller batches to avoid gRPC message size limits
    for batch_idx, batch in enumerate(dataloader):
        logger.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
        # Prepare input data for gRPC using pipeline input specs

        # Execute inference via gRPC
        inference_response = stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.tensor_to_proto(batch["cube"]),
                    wavelengths=helpers.tensor_to_proto(batch["wavelengths"]),
                ),
            )
        )

        # Convert outputs back to numpy
        output_dict = {}
        for name, tensor_proto in inference_response.outputs.items():
            output_dict[name] = helpers.proto_to_numpy(tensor_proto)

        results.append(output_dict)

    logger.info(f"Processed {len(results)} measurements")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
    logger.info("Session closed.")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Restore trained pipeline for inference using gRPC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Display pipeline info
  python run_inference.py --pipeline-path configs/pipeline/channel_selector.yaml

  # Restore pipeline with custom weights
  python run_inference.py --pipeline-path configs/pipeline/channel_selector.yaml --weights-path outputs/my_weights.pt

  # Override config values
  python run_inference.py --pipeline-path configs/pipeline/channel_selector.yaml --override nodes.10.params.output_dir=outputs/my_tb

  # Use custom gRPC server
  python run_inference.py --pipeline-path configs/pipeline/channel_selector.yaml --server localhost:50052
        """,
    )

    parser.add_argument(
        "--pipeline-path",
        type=str,
        required=True,
        help="Path to pipeline YAML file",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        required=True,
        help="Path to weights (.pt) file",
    )
    parser.add_argument(
        "--cu3s-file-path",
        type=str,
        required=True,
        help="Path to .cu3s file for inference",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:50051",
        help="gRPC server address (default: localhost:50051)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on (default: auto)",
    )
    parser.add_argument(
        "--override",
        action="append",
        help="Override config values in dot notation (e.g., nodes.10.params.output_dir=outputs/my_tb). Can be specified multiple times.",
    )
    parser.add_argument(
        "--processing-mode",
        type=str,
        default="Reflectance",
        choices=["Raw", "Reflectance"],
        help="Cuvis processing mode (default: Reflectance)",
    )

    args = parser.parse_args()

    run_inference(
        pipeline_path=args.pipeline_path,
        weights_path=args.weights_path,
        server_address=args.server,
        device=args.device,
        config_overrides=args.override,
        cu3s_file_path=args.cu3s_file_path,
        processing_mode=args.processing_mode,
    )


if __name__ == "__main__":
    main()
