"""Discover available pipelines and load one using the Phase 5 workflow."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from workflow_utils import build_stub, config_search_paths, create_session_with_search_paths

from cuvis_ai.grpc import cuvis_ai_pb2


def main() -> None:
    stub = build_stub()

    response = stub.ListAvailablePipelinees(cuvis_ai_pb2.ListAvailablePipelineesRequest())

    print(f"Found {len(response.pipelinees)} pipeline(es):")
    for pipeline in response.pipelinees:
        print(f"  {pipeline.name}")
        print(f"    Description: {pipeline.metadata.description}")
        print(f"    Tags: {', '.join(pipeline.tags)}")
        print(f"    Has weights: {pipeline.has_weights}")

    if not response.pipelinees:
        print("No pipelinees found")
        return

    anomaly_response = stub.ListAvailablePipelinees(
        cuvis_ai_pb2.ListAvailablePipelineesRequest(filter_tag="anomaly")
    )
    print(f"\nPipelinees with 'anomaly' tag: {len(anomaly_response.pipelinees)}")
    for pipeline in anomaly_response.pipelinees:
        print(f"  - {pipeline.name}")

    selected_pipeline_name = response.pipelinees[0].name
    info_response = stub.GetPipelineInfo(
        cuvis_ai_pb2.GetPipelineInfoRequest(pipeline_name=selected_pipeline_name)
    )

    pipeline_info = info_response.pipeline_info
    print(f"\nPipeline details: {pipeline_info.name}")
    print(f"  Path: {pipeline_info.path}")
    print(f"  Description: {pipeline_info.metadata.description}")
    print(f"  Tags: {', '.join(pipeline_info.tags)}")
    print(f"  Has weights: {pipeline_info.has_weights}")

    # Load the selected pipeline + weights into a fresh session
    session_id = create_session_with_search_paths(stub, config_search_paths())
    pipeline_bytes = json.dumps(yaml.safe_load(Path(pipeline_info.path).read_text())).encode(
        "utf-8"
    )
    stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=pipeline_bytes),
        )
    )
    if pipeline_info.has_weights:
        stub.LoadPipelineWeights(
            cuvis_ai_pb2.LoadPipelineWeightsRequest(
                session_id=session_id,
                weights_path=pipeline_info.weights_path,
                strict=True,
            )
        )
    print(f"\nSession created: {session_id}")
    if pipeline_info.has_weights:
        print(f"  (Weights loaded from {pipeline_info.weights_path})")

    inputs_response = stub.GetPipelineInputs(
        cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
    )
    print(f"Pipeline inputs: {', '.join(inputs_response.input_names)}")

    outputs_response = stub.GetPipelineOutputs(
        cuvis_ai_pb2.GetPipelineOutputsRequest(session_id=session_id)
    )
    print(f"Pipeline outputs: {', '.join(outputs_response.output_names)}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))


if __name__ == "__main__":
    main()
