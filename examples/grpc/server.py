"""Simple gRPC server for testing and demonstration purposes."""

from concurrent import futures
import grpc
from cuvis_ai.grpc import CuvisAIService, cuvis_ai_pb2_grpc


def serve(port: int = 50051) -> None:
    """Start the gRPC server.
    
    Args:
        port: Port number to listen on (default: 50051)
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service = CuvisAIService()
    cuvis_ai_pb2_grpc.add_CuvisAIServiceServicer_to_server(service, server)
    
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    
    print(f"gRPC server started on port {port}")
    print("Press Ctrl+C to stop")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop(0)


if __name__ == "__main__":
    serve()
