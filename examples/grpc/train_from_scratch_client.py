"""Train from Scratch Client Example - Workflow 1: Training a pipeline from scratch."""

import grpc

from cuvis_ai.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc


def main() -> None:
    channel = grpc.insecure_channel("localhost:50051")
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    # Use RestoreExperiment to load full config (pipeline + data + training + loss/metric nodes)
    response = stub.RestoreExperiment(
        cuvis_ai_pb2.RestoreExperimentRequest(
            experiment_path="configs/experiment/channel_selector.yaml"
        )
    )
    session_id = response.session_id
    print(f"Session: {session_id}")

    try:
        # Train with gradient trainer (config already loaded from experiment)
        for progress in stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
            )
        ):
            epoch = progress.context.epoch
            losses = ", ".join([f"{k}={v:.4f}" for k, v in progress.losses.items()])
            metrics = ", ".join([f"{k}={v:.4f}" for k, v in progress.metrics.items()])
            print(f"Epoch {epoch}: {losses} {metrics}")

        print("Training complete")
    except grpc.RpcError as e:
        print(f"Training failed: {e.details()}")
        return

    save_response = stub.SavePipeline(
        cuvis_ai_pb2.SavePipelineRequest(
            session_id=session_id,
            pipeline_path="models/trained_selector.yaml",
            metadata=cuvis_ai_pb2.PipelineMetadata(
                name="Trained Channel Selector",
                description="Channel selector trained on lentils dataset",
                created="2024-11-27",
                cuvis_ai_version="0.1.5",
            ),
        )
    )
    print(f"Pipeline saved: {save_response.pipeline_path}")
    print(f"Weights saved: {save_response.weights_path}")

    exp_response = stub.SaveExperiment(
        cuvis_ai_pb2.SaveExperimentRequest(
            session_id=session_id,
            experiment_path="experiments/run_001.yaml",
        )
    )
    print(f"Experiment saved: {exp_response.experiment_path}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))


if __name__ == "__main__":
    main()
