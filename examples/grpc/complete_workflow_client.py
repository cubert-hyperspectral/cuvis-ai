"""End-to-end gRPC workflow: session setup, training, inference, checkpointing."""

import argparse
from pathlib import Path

import grpc
import numpy as np

from cuvis_ai.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc, helpers


def parse_args() -> argparse.Namespace:
    default_experiment = (
        Path(__file__).resolve().parents[2] / "configs" / "experiment" / "deep_svdd.yaml"
    )
    parser = argparse.ArgumentParser(description="cuvis.ai gRPC end-to-end client")
    parser.add_argument("--target", default="localhost:50051", help="gRPC target host:port")
    parser.add_argument(
        "--experiment", default=str(default_experiment), help="Path to experiment YAML"
    )
    parser.add_argument("--checkpoint", default="model.ckpt", help="Checkpoint output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    channel = grpc.insecure_channel(args.target)
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    # Restore experiment from config file (contains pipeline, data, training configs)
    session_id = stub.RestoreExperiment(
        cuvis_ai_pb2.RestoreExperimentRequest(experiment_path=str(args.experiment))
    ).session_id
    print(f"Session created from experiment: {session_id}")

    try:
        inputs = stub.GetPipelineInputs(
            cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
        )
        outputs = stub.GetPipelineOutputs(
            cuvis_ai_pb2.GetPipelineOutputsRequest(session_id=session_id)
        )
        print(f"Inputs: {inputs.input_names}")
        print(f"Outputs: {outputs.output_names}")

        # Statistical training
        for update in stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            )
        ):
            print(f"[statistical] status={update.status}")

        # Gradient training (config already loaded from experiment)
        for progress in stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
            )
        ):
            print(
                f"[gradient] epoch={progress.context.epoch} "
                f"status={progress.status} losses={dict(progress.losses)} "
                f"metrics={dict(progress.metrics)}"
            )

        pipeline_path = str(Path(args.checkpoint).with_suffix(".yaml"))
        save_resp = stub.SavePipeline(
            cuvis_ai_pb2.SavePipelineRequest(
                session_id=session_id,
                pipeline_path=pipeline_path,
            )
        )
        print(f"Pipeline saved: {save_resp.pipeline_path}")
        print(f"Weights saved: {save_resp.weights_path}")

        cube = np.random.rand(1, 32, 32, 61).astype(np.float32)
        inference = stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(cube=helpers.numpy_to_proto(cube)),
            )
        )
        print(f"Outputs: {list(inference.outputs.keys())}")
        print(f"Metrics: {dict(inference.metrics)}")
    finally:
        stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
        channel.close()


if __name__ == "__main__":
    main()
