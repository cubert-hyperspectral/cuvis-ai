"""Example client demonstrating statistical training workflow."""

import grpc
import numpy as np
from cuvis_ai.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc


def main():
    """Demonstrate statistical training."""
    channel = grpc.insecure_channel('localhost:50051')
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)
    
    # Create session
    print("Creating session...")
    create_req = cuvis_ai_pb2.CreateSessionRequest(
        pipeline_type="statistical",
        data_config=cuvis_ai_pb2.DataConfig(
            cu3s_file_path="data/Lentils/Lentils_000.cu3s",
            annotation_json_path="data/Lentils/Lentils_000.json",
            train_ids=[0, 1, 2, 3, 4],
            val_ids=[5, 6],
            test_ids=[7, 8],
            batch_size=4,
            processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE
        )
    )
    session_resp = stub.CreateSession(create_req)
    session_id = session_resp.session_id
    print(f"Session: {session_id}\n")
    
    # Statistical training
    print("Starting statistical training...")
    train_req = cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL
    )
    
    for progress in stub.Train(train_req):
        stage = cuvis_ai_pb2.ExecutionStage.Name(progress.context.stage)
        status = cuvis_ai_pb2.TrainStatus.Name(progress.status)
        print(f"[{stage}] Status: {status} - {progress.message}")
    
    print("\nTraining complete!")
    
    # Check status
    status_req = cuvis_ai_pb2.GetTrainStatusRequest(session_id=session_id)
    status_resp = stub.GetTrainStatus(status_req)
    status_name = cuvis_ai_pb2.TrainStatus.Name(status_resp.status)
    print(f"Final status: {status_name}")
    
    # Run inference to verify
    print("\nRunning inference to verify trained model...")
    cube = np.random.rand(1, 32, 32, 61).astype(np.float32)
    
    inference_req = cuvis_ai_pb2.InferenceRequest(
        session_id=session_id,
        inputs=cuvis_ai_pb2.InputBatch(
            cube=cuvis_ai_pb2.Tensor(
                shape=[1, 32, 32, 61],
                dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
                raw_data=cube.tobytes()
            )
        ),
        output_specs=["mask", "decisions"]
    )
    
    inference_resp = stub.Inference(inference_req)
    print(f"Inference outputs: {list(inference_resp.outputs.keys())}")
    
    # Cleanup
    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
    print("\nSession closed.")


if __name__ == '__main__':
    main()
