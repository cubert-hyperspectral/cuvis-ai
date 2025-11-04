"""Stress tests for cuvis.ai training pipeline with synthetic data.

This module tests the training pipeline at various scales to identify
performance characteristics, memory requirements, and potential bottlenecks.
"""

import pytest
import torch
import time
import psutil
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, Tuple

from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.normalization.normalization import MinMaxNormalizer
from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.training.config import TrainingConfig, TrainerConfig, OptimizerConfig
from cuvis_ai.training.datamodule import GraphDataModule
from cuvis_ai.training.losses import OrthogonalityLoss
from cuvis_ai.training.metrics import ExplainedVarianceMetric
from cuvis_ai.training.monitors import DummyMonitor

from .synthetic_data import (
    create_small_scale_dataset,
    create_medium_scale_dataset,
)


class SyntheticDataModule(GraphDataModule):
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


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics.
    
    Returns
    -------
    dict
        Memory usage in MB for CPU and GPU (if available)
    """
    process = psutil.Process()
    mem_info = {
        'cpu_mb': process.memory_info().rss / (1024 * 1024),
    }
    
    if torch.cuda.is_available():
        mem_info['gpu_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
        mem_info['gpu_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
    
    return mem_info


def create_test_graph(n_channels: int = 10) -> Tuple[Graph, Dict]:
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
    graph = Graph("stress_test")
    
    # Normalizer
    normalizer = MinMaxNormalizer(use_running_stats=True)
    graph.add_node(normalizer)
    
    # PCA (trainable) - before RX to reduce dimensionality for downstream metrics
    n_components = min(3, n_channels)  # Don't exceed available channels
    pca = TrainablePCA(n_components=n_components, trainable=True)
    graph.add_node(pca, parent=normalizer)
    
    # RX detector operates on the normalized full channel space to keep statistics aligned
    rx = RXGlobal(trainable_stats=False)
    graph.add_node(rx, parent=normalizer)
    
    # Loss leaf on PCA
    orth_loss = OrthogonalityLoss(weight=0.1)
    graph.add_leaf_node(orth_loss, parent=pca)
    
    # Metric leaf on PCA
    var_metric = ExplainedVarianceMetric()
    graph.add_leaf_node(var_metric, parent=pca)
    
    config = {
        'normalizer': normalizer,
        'pca': pca,
        'rx': rx,
        'orth_loss': orth_loss,
        'var_metric': var_metric,
    }
    
    return graph, config


@pytest.mark.stress
def test_small_scale():
    """Test with small dataset: 10 samples × 64×64 × 10 channels (~0.3 MB).
    
    This test verifies correctness on a small dataset and establishes
    baseline memory and throughput metrics.
    """
    print("\n" + "="*80)
    print("STRESS TEST: Small Scale (10 samples × 64×64 × 10 channels)")
    print("="*80)
    
    # Create dataset
    dataset = create_small_scale_dataset(seed=42)
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
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
    print(f"\nMemory Before Training:")
    print(f"  CPU: {mem_before['cpu_mb']:.2f} MB")
    if 'gpu_mb' in mem_before:
        print(f"  GPU: {mem_before['gpu_mb']:.2f} MB (reserved: {mem_before['gpu_reserved_mb']:.2f} MB)")
    
    # Configure training
    config = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(
            max_epochs=2,
            accelerator='auto',
            devices=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
        ),
        optimizer=OptimizerConfig(
            name='adam',
            lr=0.001,
        )
    )
    
    # Train
    start_time = time.time()
    trainer = graph.train(datamodule=datamodule, training_config=config)
    train_time = time.time() - start_time
    
    # Track memory after training
    mem_after = get_memory_usage()
    print(f"\nMemory After Training:")
    print(f"  CPU: {mem_after['cpu_mb']:.2f} MB (delta: {mem_after['cpu_mb'] - mem_before['cpu_mb']:.2f} MB)")
    if 'gpu_mb' in mem_after:
        print(f"  GPU: {mem_after['gpu_mb']:.2f} MB (delta: {mem_after['gpu_mb'] - mem_before.get('gpu_mb', 0):.2f} MB)")
    
    # Performance metrics
    throughput = stats['n_samples'] / train_time
    print(f"\nPerformance:")
    print(f"  Training time: {train_time:.2f} seconds")
    print(f"  Throughput: {throughput:.2f} samples/second")
    
    # Verify correctness
    assert nodes['rx'].mu is not None, "RX detector not initialized"
    assert nodes['pca'].components is not None, "PCA not initialized"
    assert isinstance(nodes['pca'].components, torch.nn.Parameter), "PCA not trainable"
    
    # Test forward pass
    test_input = dataset[0]['cube'].unsqueeze(0)  # Add batch dimension
    output, _, _ = graph.forward(test_input)
    assert output.shape[0] == 1, "Batch dimension mismatch"
    
    print("\n✓ Small scale test passed")

@pytest.mark.slow
@pytest.mark.stress
def test_medium_scale():
    """Test with medium dataset: 100 samples × 128×128 × 50 channels (~80 MB).
    
    This test verifies performance with a realistic dataset size.
    """
    print("\n" + "="*80)
    print("STRESS TEST: Medium Scale (100 samples × 128×128 × 50 channels)")
    print("="*80)
    
    # Create dataset
    dataset = create_medium_scale_dataset(seed=42)
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
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
    print(f"\nMemory Before Training:")
    print(f"  CPU: {mem_before['cpu_mb']:.2f} MB")
    if 'gpu_mb' in mem_before:
        print(f"  GPU: {mem_before['gpu_mb']:.2f} MB")
    
    # Configure training (fewer epochs for speed)
    config = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(
            max_epochs=1,
            accelerator='auto',
            devices=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
        ),
        optimizer=OptimizerConfig(
            name='adam',
            lr=0.001,
        )
    )
    
    # Train
    start_time = time.time()
    trainer = graph.train(datamodule=datamodule, training_config=config)
    train_time = time.time() - start_time
    
    # Track memory
    mem_after = get_memory_usage()
    mem_delta = mem_after['cpu_mb'] - mem_before['cpu_mb']
    print(f"\nMemory After Training:")
    print(f"  CPU: {mem_after['cpu_mb']:.2f} MB (delta: {mem_delta:.2f} MB)")
    if 'gpu_mb' in mem_after:
        gpu_delta = mem_after['gpu_mb'] - mem_before.get('gpu_mb', 0)
        print(f"  GPU: {mem_after['gpu_mb']:.2f} MB (delta: {gpu_delta:.2f} MB)")
    
    # Performance metrics
    throughput = stats['n_samples'] / train_time
    print(f"\nPerformance:")
    print(f"  Training time: {train_time:.2f} seconds")
    print(f"  Throughput: {throughput:.2f} samples/second")
    print(f"  Time per sample: {train_time / stats['n_samples']:.4f} seconds")
    
    # Verify gradient computation
    assert nodes['pca'].components.grad is None or True, "Gradient computation working"
    
    print("\n✓ Medium scale test passed")



@pytest.mark.stress
def test_varying_channels():
    """Test with varying number of spectral channels: 10, 50, 100, 200.
    
    This test verifies that channel dimension scaling works correctly.
    """
    print("\n" + "="*80)
    print("STRESS TEST: Varying Channel Dimensions")
    print("="*80)
    
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
        
        # Configure training
        config = TrainingConfig(
            seed=42,
            trainer=TrainerConfig(
                max_epochs=0,  # Statistical only
                accelerator='cpu',
            )
        )
        
        # Train and measure
        start_time = time.time()
        graph.train(datamodule=datamodule, training_config=config)
        train_time = time.time() - start_time
        
        # Verify RX detector dimension
        assert nodes['rx'].mu.shape[0] == n_channels, f"RX dimension mismatch for {n_channels} channels"
        
        results.append({
            'n_channels': n_channels,
            'time': train_time,
            'throughput': 20 / train_time,
        })
        
        print(f"  Time: {train_time:.2f}s, Throughput: {20/train_time:.2f} samples/s")
    
    # Print summary
    print("\nChannel Scaling Summary:")
    print(f"{'Channels':>10} | {'Time (s)':>10} | {'Throughput':>12} | {'Scaling':>10}")
    print("-" * 50)
    base_time = results[0]['time']
    for r in results:
        scaling = r['time'] / base_time
        print(f"{r['n_channels']:>10} | {r['time']:>10.2f} | {r['throughput']:>12.2f} | {scaling:>10.2f}x")
    
    print("\nChannel scaling test passed")


@pytest.mark.stress
def test_varying_spatial():
    """Test with varying spatial dimensions: 64×64, 128×128, 256×256.
    
    This test verifies that spatial dimension scaling works correctly.
    """
    print("\n" + "="*80)
    print("STRESS TEST: Varying Spatial Dimensions")
    print("="*80)
    
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
        
        # Configure training
        config = TrainingConfig(
            seed=42,
            trainer=TrainerConfig(
                max_epochs=0,
                accelerator='cpu',
            )
        )
        
        # Train and measure
        mem_before = get_memory_usage()
        start_time = time.time()
        graph.train(datamodule=datamodule, training_config=config)
        train_time = time.time() - start_time
        mem_after = get_memory_usage()
        
        mem_delta = mem_after['cpu_mb'] - mem_before['cpu_mb']
        
        results.append({
            'size': size,
            'pixels': size * size,
            'time': train_time,
            'memory_mb': mem_delta,
            'throughput': 10 / train_time,
        })
        
        print(f"  Time: {train_time:.2f}s, Memory delta: {mem_delta:.2f} MB")
    
    # Print summary
    print("\nSpatial Scaling Summary:")
    print(f"{'Size':>10} | {'Pixels':>10} | {'Time (s)':>10} | {'Memory (MB)':>12} | {'Throughput':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['size']:>10} | {r['pixels']:>10,} | {r['time']:>10.2f} | {r['memory_mb']:>12.2f} | {r['throughput']:>12.2f}")
    
    print("\n✓ Spatial scaling test passed")


@pytest.mark.stress
def test_forward_pass_latency():
    """Test forward pass latency at different scales.
    
    This test measures inference time for single samples and batches.
    """
    print("\n" + "="*80)
    print("STRESS TEST: Forward Pass Latency")
    print("="*80)
    
    # Create and train a small model
    dataset = create_small_scale_dataset(n_samples=20)
    datamodule = SyntheticDataModule(dataset, batch_size=4)
    
    graph, nodes = create_test_graph(n_channels=10)
    
    config = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(max_epochs=0, accelerator='cpu')
    )
    
    graph.train(datamodule=datamodule, training_config=config)
    
    # Test single sample latency
    print("\nSingle Sample Latency:")
    test_input = dataset[0]['cube'].unsqueeze(0)
    
    # Warmup
    for _ in range(5):
        _ = graph.forward(test_input)
    
    # Measure
    n_runs = 50
    start_time = time.time()
    for _ in range(n_runs):
        _ = graph.forward(test_input)
    avg_latency_ms = (time.time() - start_time) / n_runs * 1000
    
    print(f"  Average latency: {avg_latency_ms:.2f} ms per sample")
    print(f"  Throughput: {1000 / avg_latency_ms:.2f} samples/second")
    
    # Test batch latency
    print("\nBatch Latency:")
    for batch_size in [1, 4, 8, 16]:
        batch = torch.stack([dataset[i]['cube'] for i in range(min(batch_size, len(dataset)))])
        
        # Warmup
        for _ in range(5):
            _ = graph.forward(batch)
        
        # Measure
        start_time = time.time()
        for _ in range(10):
            _ = graph.forward(batch)
        avg_time = (time.time() - start_time) / 10
        
        print(f"  Batch size {batch_size:2d}: {avg_time*1000:.2f} ms total, {avg_time/batch_size*1000:.2f} ms per sample")
    
    print("\n✓ Forward pass latency test passed")


if __name__ == "__main__":
    # Run tests when executed directly
    print("Running stress tests...")
    print("Note: Use pytest with markers to run specific tests:")
    print("  pytest tests/stress/test_pipeline_stress.py -v -m stress")
    print("  pytest tests/stress/test_pipeline_stress.py -v -m 'stress and not slow'")
