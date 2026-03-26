"""Discover available pipelines and load one using the Phase 5 workflow."""

from __future__ import annotations

import json
from pathlib import Path

import click
import yaml
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
from workflow_utils import build_stub, config_search_paths, create_session_with_search_paths


def main() -> None:
    stub = build_stub()

    response = stub.ListAvailablePipelines(cuvis_ai_pb2.ListAvailablePipelinesRequest())

    print(f"Found {len(response.pipelines)} pipeline(s):")
    for pipeline in response.pipelines:
        print(f"  {pipeline.pipeline_path}")
        print(f"    Description: {pipeline.metadata.description}")
        print(f"    Tags: {', '.join(pipeline.metadata.tags)}")
        print(f"    Has weights: {bool(pipeline.weights_path)}")

    if not response.pipelines:
        print("No pipelines found")
        return

    anomaly_response = stub.ListAvailablePipelines(
        cuvis_ai_pb2.ListAvailablePipelinesRequest(filter_tag="anomaly")
    )
    print(f"\nPipelines with 'anomaly' tag: {len(anomaly_response.pipelines)}")
    for pipeline in anomaly_response.pipelines:
        print(f"  - {pipeline.pipeline_path}")

    selected_pipeline_path = response.pipelines[0].pipeline_path
    info_response = stub.GetPipelineInfo(
        cuvis_ai_pb2.GetPipelineInfoRequest(pipeline_path=selected_pipeline_path)
    )

    pipeline_info = info_response.pipeline_info
    print(f"\nPipeline details: {pipeline_info.pipeline_path}")
    print(f"  Path: {pipeline_info.path}")
    print(f"  Description: {pipeline_info.metadata.description}")
    print(f"  Tags: {', '.join(pipeline_info.metadata.tags)}")
    print(f"  Has weights: {bool(pipeline_info.weights_path)}")

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
    if pipeline_info.weights_path:
        stub.LoadPipelineWeights(
            cuvis_ai_pb2.LoadPipelineWeightsRequest(
                session_id=session_id,
                weights_path=pipeline_info.weights_path,
                strict=True,
            )
        )
    print(f"\nSession created: {session_id}")
    if pipeline_info.weights_path:
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


@click.command()
def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
