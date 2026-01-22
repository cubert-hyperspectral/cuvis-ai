"""AdaCLIP baseline inference on Lentils dataset using gRPC.

This client demonstrates running AdaCLIP anomaly detection on lentils dataset via gRPC:
  - Loads the adaclip_baseline pipeline configuration
  - Loads lentils dataset samples
  - Runs inference with AdaCLIP detector
  - Outputs anomaly scores and visualizations
"""

from __future__ import annotations

import numpy as np
from cuvis_ai.data.datasets import SingleCu3sDataset
from cuvis_ai_core.grpc import cuvis_ai_pb2, helpers
from torch.utils.data import DataLoader
from workflow_utils import (
    build_stub,
    config_search_paths,
    create_session_with_search_paths,
)


def main(
    lentils_path: str = "path/to/lentils/dataset",
    server_address: str = "localhost:50051",
    max_samples: int | None = None,
) -> None:
    """Run AdaCLIP baseline inference on lentils dataset.

    Parameters
    ----------
    lentils_path : str
        Path to lentils dataset directory containing .cu3s files
    server_address : str
        gRPC server address (default: localhost:50051)
    max_samples : int | None
        Maximum number of samples to process (None = all)
    """
    stub = build_stub(server_address, max_msg_size=300 * 1024 * 1024)
    session_id = create_session_with_search_paths(stub, config_search_paths())
    print(f"Session created: {session_id}")

    # Load AdaCLIP baseline pipeline
    pipeline_config = stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="pipeline",
            path="configs/pipeline/adaclip_baseline.yaml",
            overrides=[],
        )
    )

    stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=pipeline_config.config_bytes),
        )
    )
    print("Loaded AdaCLIP baseline pipeline (using pretrained weights)")

    # Get pipeline specs
    inputs_resp = stub.GetPipelineInputs(
        cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
    )
    outputs_resp = stub.GetPipelineOutputs(
        cuvis_ai_pb2.GetPipelineOutputsRequest(session_id=session_id)
    )

    print("\nInput specs:", list(inputs_resp.input_specs.keys()))
    print("Output specs:", list(outputs_resp.output_specs.keys()))

    # Load lentils dataset
    print(f"\nLoading lentils dataset from {lentils_path}...")
    dataset = SingleCu3sDataset(
        cu3s_file_path=lentils_path,
        processing_mode="Reflectance",
    )
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

    print(f"Running inference on {len(dataloader)} samples...")
    results = []

    for idx, batch in enumerate(dataloader):
        if max_samples and idx >= max_samples:
            break

        # Run inference
        inference = stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.tensor_to_proto(batch["cube"]),
                    wavelengths=helpers.tensor_to_proto(batch["wavelengths"]),
                ),
            )
        )

        # Extract key outputs
        output_dict = {}
        for name, tensor_proto in inference.outputs.items():
            output_dict[name] = helpers.proto_to_numpy(tensor_proto)

        results.append(output_dict)

        # Print progress
        if "adaclip_detector.scores" in output_dict:
            scores = output_dict["adaclip_detector.scores"]
            decisions = output_dict.get("quantile_decider.decisions")
            anomaly_ratio = np.mean(decisions > 0.5) if decisions is not None else 0
            print(
                f"  Sample {idx + 1}: scores=[{scores.min():.3f}, {scores.max():.3f}], "
                f"anomaly_ratio={anomaly_ratio:.1%}"
            )

    print(f"\nProcessed {len(results)} samples successfully")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
    print("Session closed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run AdaCLIP baseline inference on lentils dataset via gRPC"
    )
    parser.add_argument(
        "--lentils-path",
        type=str,
        default="path/to/lentils/dataset",
        help="Path to lentils dataset directory",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:50051",
        help="gRPC server address (default: localhost:50051)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )

    args = parser.parse_args()
    main(
        lentils_path=args.lentils_path,
        server_address=args.server,
        max_samples=args.max_samples,
    )
