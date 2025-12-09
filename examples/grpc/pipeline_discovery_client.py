"""Pipeline Discovery Client Example - Workflow 4: Discovering and inspecting available pipelinees."""

import grpc

from cuvis_ai.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc


def main() -> None:
    channel = grpc.insecure_channel("localhost:50051")
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

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

    # Server automatically loads weights if they exist alongside the YAML config
    session_response = stub.CreateSession(
        cuvis_ai_pb2.CreateSessionRequest(
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=pipeline_info.path.encode("utf-8"))
        )
    )
    print(f"\nSession created: {session_response.session_id}")
    if pipeline_info.has_weights:
        print(f"  (Weights automatically loaded from {pipeline_info.weights_path})")

    try:
        inputs_response = stub.GetPipelineInputs(
            cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_response.session_id)
        )
        print(f"Pipeline inputs: {', '.join(inputs_response.input_names)}")

        outputs_response = stub.GetPipelineOutputs(
            cuvis_ai_pb2.GetPipelineOutputsRequest(session_id=session_response.session_id)
        )
        print(f"Pipeline outputs: {', '.join(outputs_response.output_names)}")
    except grpc.RpcError as e:
        print(f"Could not verify session: {e.details()}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_response.session_id))


if __name__ == "__main__":
    main()
