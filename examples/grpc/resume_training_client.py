"""Resume Training Client Example - Workflow 3: Resuming training from a saved experiment."""

import grpc

from cuvis_ai.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc


def main() -> None:
    channel = grpc.insecure_channel("localhost:50051")
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    # Restore experiment with all configs (pipeline + data + training + loss/metric nodes)
    restore_response = stub.RestoreExperiment(
        cuvis_ai_pb2.RestoreExperimentRequest(
            experiment_path="configs/experiment/deep_svdd.yaml",
        )
    )

    session_id = restore_response.session_id
    experiment = restore_response.experiment

    print(f"Experiment restored: {experiment.name}")
    print(f"Session: {session_id}")
    print("Pipeline config loaded from experiment")

    print(f"Data: {experiment.data.cu3s_file_path}")
    print(f"Batch size: {experiment.data.batch_size}")

    try:
        epoch_count = 0
        # Resume training with configs already loaded from experiment
        # Note: If you need to modify training params (lr, epochs, etc.),
        # you should create a new experiment YAML file with the desired changes
        for progress in stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
            )
        ):
            epoch = progress.context.epoch
            losses = ", ".join([f"{k}={v:.4f}" for k, v in progress.losses.items()])

            if epoch != epoch_count:
                print(f"Epoch {epoch}: {losses}")
                epoch_count = epoch

        print("Training complete")
    except grpc.RpcError as e:
        print(f"Training failed: {e.details()}")
        return

    save_pipeline_response = stub.SavePipeline(
        cuvis_ai_pb2.SavePipelineRequest(
            session_id=session_id,
            pipeline_path="configs/pipeline/resumed_model.yaml",
            metadata=cuvis_ai_pb2.PipelineMetadata(
                name="Resumed Training Model",
                description="Continued training from deep_svdd experiment",
                created="2024-11-27",
            ),
        )
    )
    print(f"Pipeline saved: {save_pipeline_response.pipeline_path}")
    print(f"Weights saved: {save_pipeline_response.weights_path}")

    save_exp_response = stub.SaveExperiment(
        cuvis_ai_pb2.SaveExperimentRequest(
            session_id=session_id,
            experiment_path="experiments/resumed_training.yaml",
        )
    )
    print(f"Experiment saved: {save_exp_response.experiment_path}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))


if __name__ == "__main__":
    main()
