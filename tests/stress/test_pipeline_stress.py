"""Stress tests for cuvis.ai training pipeline with synthetic data.

This module tests the training pipeline at various scales to identify
performance characteristics, memory requirements, and potential bottlenecks.
"""

import time

import psutil
import pytest
import torch
from torch.utils.data import DataLoader

from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.node.losses import OrthogonalityLoss
from cuvis_ai.node.metrics import ExplainedVarianceMetric
from cuvis_ai.node.node import Node
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.pipeline.pipeline import CuvisPipeline
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.training.datamodule import CuvisDataModule
from cuvis_ai.training.trainers import StatisticalTrainer

from .synthetic_data import (
    create_medium_scale_dataset,
    create_small_scale_dataset,
)


class SimpleDataNode(Node):
    """Simple data node that extracts cube from batch for stress tests."""

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C]",
        ),
    }

    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C]",
        ),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, cube, context):
        """Pass through the cube data."""
        return {"cube": cube}


class SyntheticDataModule(CuvisDataModule):
    """DataModule wrapper for synthetic dataset."""

    def __init__(self, dataset, batch_size=4, num_workers=0):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Split into train/val (80/20)
        n_train = int(0.8 * len(dataset))
        self.train_dataset = torch.utils.data.Subset(dataset, range(n_train))
        self.val_dataset = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


def get_memory_usage() -> dict[str, float]:
    """Get current memory usage statistics.

    Returns
    -------
    dict
        Memory usage in MB for CPU and GPU (if available)
    """
    process = psutil.Process()
    mem_info = {
        "cpu_mb": process.memory_info().rss / (1024 * 1024),
    }

    if torch.cuda.is_available():
        mem_info["gpu_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        mem_info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)

    return mem_info


def create_test_graph(n_channels: int = 10) -> tuple[CuvisPipeline, dict]:
    """Create a test graph with standard nodes for stress testing.

    Parameters
    ----------
    n_channels : int
        Number of input channels

    Returns
    -------
    tuple
        (graph, config_dict) where config_dict contains node references
    """
    pipeline = CuvisPipeline("stress_test")

    # Data node to extract cube from batch
    data_node = SimpleDataNode()

    # Normalizer
    normalizer = MinMaxNormalizer(use_running_stats=True)
    pipeline.connect(data_node.outputs.cube, normalizer.data)

    # PCA (trainable) - before RX to reduce dimensionality for downstream metrics
    n_components = min(3, n_channels)  # Don't exceed available channels
    pca = TrainablePCA(n_components=n_components, trainable=True)
    pipeline.connect(normalizer.normalized, pca.data)

    # RX detector operates on the normalized full channel space to keep statistics aligned
    rx = RXGlobal(num_channels=n_channels)
    pipeline.connect(normalizer.normalized, rx.data)

    # Loss leaf on PCA
    orth_loss = OrthogonalityLoss(weight=0.1)
    pipeline.connect(pca.outputs.components, orth_loss.components)

    # Metric leaf on PCA
    var_metric = ExplainedVarianceMetric()
    pipeline.connect(pca.outputs.explained_variance_ratio, var_metric.explained_variance_ratio)

    config = {
        "data_node": data_node,
        "normalizer": normalizer,
        "pca": pca,
        "rx": rx,
        "orth_loss": orth_loss,
        "var_metric": var_metric,
    }

    return pipeline, config


@pytest.mark.stress
def test_small_scale():
    """Test with small dataset: 10 samples × 64×64 × 10 channels (~0.3 MB).

    This test verifies correctness on a small dataset and establishes
    baseline memory and throughput metrics.
    """
    print("\n" + "=" * 80)
    print("STRESS TEST: Small Scale (10 samples × 64×64 × 10 channels)")
    print("=" * 80)

    # Create dataset
    dataset = create_small_scale_dataset(seed=42)
    stats = dataset.get_statistics()
    print("\nDataset Statistics:")
    print(f"  Total samples: {stats['n_samples']}")
    print(f"  Cube shape: {stats['cube_shape']}")
    print(f"  Total pixels: {stats['total_pixels']:,}")
    print(f"  Anomaly pixels: {stats['anomaly_pixels']:,}")
    print(f"  Memory estimate: {stats['memory_mb']:.2f} MB")

    # Create datamodule
    datamodule = SyntheticDataModule(dataset, batch_size=2)

    # Create graph
    graph, nodes = create_test_graph(n_channels=10)

    # Track memory before training
    mem_before = get_memory_usage()
    print("\nMemory Before Training:")
    print(f"  CPU: {mem_before['cpu_mb']:.2f} MB")
    if "gpu_mb" in mem_before:
        gpu_reserved = mem_before["gpu_reserved_mb"]
        print(f"  GPU: {mem_before['gpu_mb']:.2f} MB (reserved: {gpu_reserved:.2f} MB)")

    # Train using StatisticalTrainer
    # (since max_epochs was 2, we'll do statistical init + a bit of gradient)
    # But based on the test context, this is primarily testing statistical initialization
    start_time = time.time()
    trainer = StatisticalTrainer(pipeline=graph, datamodule=datamodule)
    trainer.fit()
    train_time = time.time() - start_time

    # Track memory after training
    mem_after = get_memory_usage()
    print("\nMemory After Training:")
    cpu_delta = mem_after["cpu_mb"] - mem_before["cpu_mb"]
    print(f"  CPU: {mem_after['cpu_mb']:.2f} MB (delta: {cpu_delta:.2f} MB)")
    if "gpu_mb" in mem_after:
        gpu_delta = mem_after["gpu_mb"] - mem_before.get("gpu_mb", 0)
        print(f"  GPU: {mem_after['gpu_mb']:.2f} MB (delta: {gpu_delta:.2f} MB)")

    # Performance metrics
    throughput = stats["n_samples"] / train_time
    print("\nPerformance:")
    print(f"  Training time: {train_time:.2f} seconds")
    print(f"  Throughput: {throughput:.2f} samples/second")

    # Verify correctness
    assert nodes["rx"].mu is not None, "RX detector not initialized"
    assert nodes["pca"]._components is not None, "PCA not initialized"

    # Test forward pass
    # Add batch dimension to all items
    test_batch = {k: v.unsqueeze(0) for k, v in dataset[0].items()}
    output = graph.forward(test_batch)
    # Output is a dict of node outputs, just verify we got results
    assert len(output) > 0, "Forward pass should return outputs"

    print("\n✓ Small scale test passed")


@pytest.mark.slow
@pytest.mark.stress
def test_medium_scale():
    """Test with medium dataset: 100 samples × 128×128 × 50 channels (~80 MB).

    This test verifies performance with a realistic dataset size.
    """
    print("\n" + "=" * 80)
    print("STRESS TEST: Medium Scale (100 samples × 128×128 × 50 channels)")
    print("=" * 80)

    # Create dataset
    dataset = create_medium_scale_dataset(seed=42)
    stats = dataset.get_statistics()
    print("\nDataset Statistics:")
    print(f"  Total samples: {stats['n_samples']}")
    print(f"  Cube shape: {stats['cube_shape']}")
    print(f"  Total pixels: {stats['total_pixels']:,}")
    print(f"  Memory estimate: {stats['memory_mb']:.2f} MB")

    # Create datamodule
    datamodule = SyntheticDataModule(dataset, batch_size=4)

    # Create graph
    graph, nodes = create_test_graph(n_channels=50)

    # Track memory
    mem_before = get_memory_usage()
    print("\nMemory Before Training:")
    print(f"  CPU: {mem_before['cpu_mb']:.2f} MB")
    if "gpu_mb" in mem_before:
        print(f"  GPU: {mem_before['gpu_mb']:.2f} MB")

    # Train using StatisticalTrainer
    start_time = time.time()
    trainer = StatisticalTrainer(pipeline=graph, datamodule=datamodule)
    trainer.fit()
    train_time = time.time() - start_time

    # Track memory
    mem_after = get_memory_usage()
    mem_delta = mem_after["cpu_mb"] - mem_before["cpu_mb"]
    print("\nMemory After Training:")
    print(f"  CPU: {mem_after['cpu_mb']:.2f} MB (delta: {mem_delta:.2f} MB)")
    if "gpu_mb" in mem_after:
        gpu_delta = mem_after["gpu_mb"] - mem_before.get("gpu_mb", 0)
        print(f"  GPU: {mem_after['gpu_mb']:.2f} MB (delta: {gpu_delta:.2f} MB)")

    # Performance metrics
    throughput = stats["n_samples"] / train_time
    print("\nPerformance:")
    print(f"  Training time: {train_time:.2f} seconds")
    print(f"  Throughput: {throughput:.2f} samples/second")
    print(f"  Time per sample: {train_time / stats['n_samples']:.4f} seconds")

    # Verify gradient computation
    components = nodes["pca"]._components
    assert isinstance(components, torch.Tensor), "PCA components should be a tensor after init"
    assert components.grad is None, "Statistical init should not track gradients yet"

    print("\n✓ Medium scale test passed")


@pytest.mark.stress
def test_varying_channels():
    """Test with varying number of spectral channels: 10, 50, 100, 200.

    This test verifies that channel dimension scaling works correctly.
    """
    print("\n" + "=" * 80)
    print("STRESS TEST: Varying Channel Dimensions")
    print("=" * 80)

    channel_configs = [10, 50, 100, 200]
    results = []

    for n_channels in channel_configs:
        print(f"\nTesting with {n_channels} channels...")

        # Create dataset
        dataset = create_small_scale_dataset(
            n_channels=n_channels,
            height=32,  # Smaller spatial for speed
            width=32,
            n_samples=20,
        )

        # Create datamodule
        datamodule = SyntheticDataModule(dataset, batch_size=4)

        # Create graph
        graph, nodes = create_test_graph(n_channels=n_channels)

        # Train and measure using StatisticalTrainer
        start_time = time.time()
        trainer = StatisticalTrainer(pipeline=graph, datamodule=datamodule)
        trainer.fit()
        train_time = time.time() - start_time

        # Verify RX detector dimension
        assert nodes["rx"].mu.shape[0] == n_channels, (
            f"RX dimension mismatch for {n_channels} channels"
        )

        results.append(
            {
                "n_channels": n_channels,
                "time": train_time,
                "throughput": 20 / train_time,
            }
        )

        print(f"  Time: {train_time:.2f}s, Throughput: {20 / train_time:.2f} samples/s")

    # Print summary
    print("\nChannel Scaling Summary:")
    print(f"{'Channels':>10} | {'Time (s)':>10} | {'Throughput':>12} | {'Scaling':>10}")
    print("-" * 50)
    base_time = results[0]["time"]
    for r in results:
        scaling = r["time"] / base_time
        print(
            f"{r['n_channels']:>10} | {r['time']:>10.2f} | "
            f"{r['throughput']:>12.2f} | {scaling:>10.2f}x"
        )

    print("\nChannel scaling test passed")


@pytest.mark.stress
def test_varying_spatial():
    """Test with varying spatial dimensions: 64×64, 128×128, 256×256.

    This test verifies that spatial dimension scaling works correctly.
    """
    print("\n" + "=" * 80)
    print("STRESS TEST: Varying Spatial Dimensions")
    print("=" * 80)

    spatial_configs = [64, 128, 256]
    results = []

    for size in spatial_configs:
        print(f"\nTesting with {size}×{size} spatial dimensions...")

        # Create dataset
        dataset = create_small_scale_dataset(
            height=size,
            width=size,
            n_channels=10,
            n_samples=10,
        )

        stats = dataset.get_statistics()
        print(f"  Memory estimate: {stats['memory_mb']:.2f} MB")

        # Create datamodule
        datamodule = SyntheticDataModule(dataset, batch_size=2)

        # Create graph
        graph, nodes = create_test_graph(n_channels=10)

        # Train and measure using StatisticalTrainer
        mem_before = get_memory_usage()
        start_time = time.time()
        trainer = StatisticalTrainer(pipeline=graph, datamodule=datamodule)
        trainer.fit()
        train_time = time.time() - start_time
        mem_after = get_memory_usage()

        mem_delta = mem_after["cpu_mb"] - mem_before["cpu_mb"]

        results.append(
            {
                "size": size,
                "pixels": size * size,
                "time": train_time,
                "memory_mb": mem_delta,
                "throughput": 10 / train_time,
            }
        )

        print(f"  Time: {train_time:.2f}s, Memory delta: {mem_delta:.2f} MB")

    # Print summary
    print("\nSpatial Scaling Summary:")
    header = (
        f"{'Size':>10} | {'Pixels':>10} | {'Time (s)':>10} | "
        f"{'Memory (MB)':>12} | {'Throughput':>12}"
    )
    print(header)
    print("-" * 70)
    for r in results:
        print(
            f"{r['size']:>10} | {r['pixels']:>10,} | {r['time']:>10.2f} | "
            f"{r['memory_mb']:>12.2f} | {r['throughput']:>12.2f}"
        )

    print("\n✓ Spatial scaling test passed")


@pytest.mark.stress
def test_forward_pass_latency():
    """Test forward pass latency at different scales.

    This test measures inference time for single samples and batches.
    """
    print("\n" + "=" * 80)
    print("STRESS TEST: Forward Pass Latency")
    print("=" * 80)

    # Create and train a small model
    dataset = create_small_scale_dataset(n_samples=20)
    datamodule = SyntheticDataModule(dataset, batch_size=4)

    graph, nodes = create_test_graph(n_channels=10)

    # Use StatisticalTrainer for initialization
    trainer = StatisticalTrainer(pipeline=graph, datamodule=datamodule)
    trainer.fit()

    # Test single sample latency
    print("\nSingle Sample Latency:")
    test_batch = {k: v.unsqueeze(0) for k, v in dataset[0].items()}

    # Warmup
    for _ in range(5):
        _ = graph.forward(test_batch)

    # Measure
    n_runs = 50
    start_time = time.time()
    for _ in range(n_runs):
        _ = graph.forward(test_batch)
    avg_latency_ms = (time.time() - start_time) / n_runs * 1000

    print(f"  Average latency: {avg_latency_ms:.2f} ms per sample")
    print(f"  Throughput: {1000 / avg_latency_ms:.2f} samples/second")

    # Test batch latency
    print("\nBatch Latency:")
    for batch_size in [1, 4, 8, 16]:
        # Create batch dict with all keys from dataset
        batch = {
            k: torch.stack([dataset[i][k] for i in range(min(batch_size, len(dataset)))])
            for k in dataset[0].keys()
        }

        # Warmup
        for _ in range(5):
            _ = graph.forward(batch)

        # Measure
        start_time = time.time()
        for _ in range(10):
            _ = graph.forward(batch)
        avg_time = (time.time() - start_time) / 10

        ms_per_sample = avg_time / batch_size * 1000
        print(
            f"  Batch size {batch_size:2d}: {avg_time * 1000:.2f} ms total, "
            f"{ms_per_sample:.2f} ms per sample"
        )

    print("\n✓ Forward pass latency test passed")


if __name__ == "__main__":
    # Run tests when executed directly
    print("Running stress tests...")
    print("Note: Use pytest with markers to run specific tests:")
    print("  pytest tests/stress/test_pipeline_stress.py -v -m stress")
    print("  pytest tests/stress/test_pipeline_stress.py -v -m 'stress and not slow'")
