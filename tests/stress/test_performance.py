"""Performance benchmarks for gRPC service operations.

This module provides comprehensive performance testing for:
- Config resolution time (simple and complex)
- Schema generation time
- Training performance (statistical and gradient)
- Memory usage characteristics

All benchmarks should meet the Phase 5 requirements:
- Simple config resolution: <500ms
- Complex config resolution: <1000ms
- Schema generation: <500ms
"""

import json
import time

import pytest

from cuvis_ai_core.grpc.service import CuvisAIService
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2
from cuvis_ai_core.utils.config_helpers import resolve_config_with_hydra


class PerformanceBenchmark:
    """Performance benchmarking utilities."""

    def __init__(self):
        self.results = {}

    def benchmark(self, name: str, func, *args, **kwargs) -> float:
        """Run a benchmark and record the result."""
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        self.results[name] = elapsed
        return result

    def assert_performance(self, name: str, max_ms: float):
        """Assert that a benchmark meets performance requirements."""
        elapsed = self.results.get(name)
        assert elapsed is not None, f"Benchmark '{name}' not found"
        assert elapsed <= max_ms, (
            f"Performance requirement failed: {name} took {elapsed:.2f}ms (max {max_ms}ms)"
        )


@pytest.fixture
def benchmark():
    """Provide performance benchmark fixture."""
    return PerformanceBenchmark()


@pytest.fixture
def grpc_service():
    """Provide gRPC service for performance testing."""
    return CuvisAIService()


def test_config_resolution_simple_performance(benchmark, grpc_service):
    """Test simple config resolution performance (<500ms)."""

    # Benchmark simple config resolution
    def resolve_simple():
        return resolve_config_with_hydra(
            config_type="trainrun",
            config_path="trainrun/rx_statistical",
            search_paths=["configs"],
            overrides=[],
        )

    result = benchmark.benchmark("config_resolution_simple", resolve_simple)
    assert isinstance(result, dict)
    assert "name" in result

    # Assert performance requirement
    benchmark.assert_performance("config_resolution_simple", 500.0)


def test_config_resolution_complex_performance(benchmark, grpc_service):
    """Test complex config resolution performance with overrides (<1000ms)."""

    # Benchmark complex config resolution with multiple overrides
    def resolve_complex():
        return resolve_config_with_hydra(
            config_type="trainrun",
            config_path="trainrun/rx_statistical",
            search_paths=["configs"],
            overrides=[
                "training.optimizer.lr=0.01",
                "training.optimizer.weight_decay=0.05",
                "training.trainer.max_epochs=20",
                "data.batch_size=16",
                "data.processing_mode=Reflectance",
            ],
        )

    result = benchmark.benchmark("config_resolution_complex", resolve_complex)
    assert isinstance(result, dict)
    assert "name" in result

    # Assert performance requirement
    benchmark.assert_performance("config_resolution_complex", 1000.0)


def test_schema_generation_performance(benchmark, grpc_service):
    """Test schema generation performance (<500ms)."""

    # Benchmark schema generation
    def generate_schema():
        request = cuvis_ai_pb2.GetParameterSchemaRequest(config_type="trainrun")
        response = grpc_service.GetParameterSchema(request, None)
        return json.loads(response.json_schema)

    schema = benchmark.benchmark("schema_generation", generate_schema)
    assert isinstance(schema, dict)
    assert schema.get("title") == "TrainRunConfig"

    # Assert performance requirement
    benchmark.assert_performance("schema_generation", 500.0)


def test_session_creation_performance(benchmark, grpc_service):
    """Test session creation performance."""

    def create_session():
        request = cuvis_ai_pb2.CreateSessionRequest()
        response = grpc_service.CreateSession(request, None)
        return response.session_id

    session_id = benchmark.benchmark("session_creation", create_session)
    assert isinstance(session_id, str)
    assert len(session_id) > 0

    # No strict requirement, but should be fast
    assert benchmark.results["session_creation"] < 100.0


def test_batch_config_resolutions(benchmark):
    """Test performance of multiple config resolutions in sequence."""

    def resolve_multiple():
        results = []
        for i in range(10):
            result = resolve_config_with_hydra(
                config_type="trainrun",
                config_path="trainrun/rx_statistical",
                search_paths=["configs"],
                overrides=[f"training.seed={i}"],
            )
            results.append(result)
        return results

    results = benchmark.benchmark("batch_config_resolutions", resolve_multiple)
    assert len(results) == 10
    assert all(isinstance(r, dict) for r in results)

    # Average should still be reasonable
    avg_time = benchmark.results["batch_config_resolutions"] / 10
    assert avg_time < 300.0, f"Average resolution time too high: {avg_time:.2f}ms"


def test_memory_usage_characteristics():
    """Test memory usage characteristics (basic check)."""
    import os

    import psutil

    # Get current process
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Perform some operations
    _ = CuvisAIService()

    for i in range(5):
        resolve_config_with_hydra(
            config_type="trainrun",
            config_path="trainrun/rx_statistical",
            search_paths=["configs"],
            overrides=[f"training.seed={i}"],
        )

    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory

    # Memory increase should be reasonable (not leaking)
    assert memory_increase < 50.0, f"Memory increase too high: {memory_increase:.2f}MB"


def test_performance_summary():
    """Generate performance summary for documentation."""
    benchmark = PerformanceBenchmark()

    # Run all benchmarks
    test_config_resolution_simple_performance(benchmark, CuvisAIService())
    test_config_resolution_complex_performance(benchmark, CuvisAIService())
    test_schema_generation_performance(benchmark, CuvisAIService())
    test_session_creation_performance(benchmark, CuvisAIService())
    test_batch_config_resolutions(benchmark)

    # Generate summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": benchmark.results,
        "requirements": {
            "config_resolution_simple": {
                "max_ms": 500,
                "met": benchmark.results["config_resolution_simple"] <= 500,
            },
            "config_resolution_complex": {
                "max_ms": 1000,
                "met": benchmark.results["config_resolution_complex"] <= 1000,
            },
            "schema_generation": {
                "max_ms": 500,
                "met": benchmark.results["schema_generation"] <= 500,
            },
        },
        "overall_status": "PASS"
        if all(
            [
                benchmark.results["config_resolution_simple"] <= 500,
                benchmark.results["config_resolution_complex"] <= 1000,
                benchmark.results["schema_generation"] <= 500,
            ]
        )
        else "FAIL",
    }

    # Print summary
    print("\n=== PERFORMANCE SUMMARY ===")
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Overall Status: {summary['overall_status']}")
    print("\nBenchmark Results:")
    for name, time_ms in benchmark.results.items():
        print(f"  {name}: {time_ms:.2f}ms")

    print("\nRequirement Status:")
    for req_name, req_data in summary["requirements"].items():
        status = "✅ PASS" if req_data["met"] else "❌ FAIL"
        print(
            f"  {req_name}: {status} ({benchmark.results[req_name]:.2f}ms <= {req_data['max_ms']}ms)"
        )

    return summary
