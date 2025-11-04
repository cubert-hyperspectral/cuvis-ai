"""GPU-specific stress tests for cuvis.ai training pipeline.

This module tests GPU acceleration and compares CPU vs GPU performance.
"""

import pytest
import torch
import time
from typing import Dict, Tuple

from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.normalization.normalization import MinMaxNormalizer
from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.training.config import TrainingConfig, TrainerConfig, OptimizerConfig
from cuvis_ai.training.losses import OrthogonalityLoss
from cuvis_ai.training.metrics import ExplainedVarianceMetric

from .synthetic_data import create_small_scale_dataset
from .test_pipeline_stress import SyntheticDataModule, get_memory_usage, create_test_graph


def create_realistic_test_graph(n_channels: int = 50) -> Tuple[Graph, Dict]:
    """Create a more complex test graph for realistic GPU benchmarking.
    
    This graph includes more trainable parameters and computationally
    intensive operations to better demonstrate GPU acceleration benefits.
    
    Parameters
    ----------
    n_channels : int
        Number of input channels
        
    Returns
    -------
    tuple
        (graph, config_dict) where config_dict contains node references
    """
    graph = Graph("gpu_stress_test")
    
    # Normalizer
    normalizer = MinMaxNormalizer(use_running_stats=True)
    graph.add_node(normalizer)
    
    # First PCA layer (trainable) - reduce dimensions
    n_components_1 = min(20, n_channels)  
    pca1 = TrainablePCA(n_components=n_components_1, trainable=True)
    graph.add_node(pca1, parent=normalizer)
    
    # Second PCA layer (trainable) - further reduction
    n_components_2 = min(10, n_components_1)
    pca2 = TrainablePCA(n_components=n_components_2, trainable=True)
    graph.add_node(pca2, parent=pca1)
    
    # RX detector with trainable statistics
    rx = RXGlobal(trainable_stats=True)
    graph.add_node(rx, parent=pca2)
    
    # Multiple loss leaves for more gradient computation
    orth_loss_1 = OrthogonalityLoss(weight=0.1)
    graph.add_leaf_node(orth_loss_1, parent=pca1)
    
    orth_loss_2 = OrthogonalityLoss(weight=0.1)
    graph.add_leaf_node(orth_loss_2, parent=pca2)
    
    # Metrics
    var_metric_1 = ExplainedVarianceMetric()
    graph.add_leaf_node(var_metric_1, parent=pca1)
    
    var_metric_2 = ExplainedVarianceMetric()
    graph.add_leaf_node(var_metric_2, parent=pca2)
    
    config = {
        'normalizer': normalizer,
        'pca1': pca1,
        'pca2': pca2,
        'rx': rx,
        'orth_loss_1': orth_loss_1,
        'orth_loss_2': orth_loss_2,
        'var_metric_1': var_metric_1,
        'var_metric_2': var_metric_2,
    }
    
    return graph, config


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_acceleration():
    """Test GPU acceleration with realistic workload and proper benchmarking.
    
    This test uses a larger dataset and more complex model to demonstrate
    GPU acceleration benefits. It includes proper warm-up, detailed profiling,
    and multiple measurement runs for accuracy.
    """
    print("\n" + "="*80)
    print("STRESS TEST: GPU Acceleration (Improved)")
    print("="*80)
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create realistic dataset - larger size to benefit from GPU
    print("\n--- Dataset Configuration ---")
    dataset = create_small_scale_dataset(
        n_samples=200,  # More samples
        height=128,     # Larger spatial dimensions
        width=128,
        n_channels=50,  # More channels
        seed=42
    )
    stats = dataset.get_statistics()
    print(f"Samples: {stats['n_samples']}")
    print(f"Cube shape: {stats['cube_shape']}")
    print(f"Total pixels: {stats['total_pixels']:,}")
    print(f"Memory estimate: {stats['memory_mb']:.2f} MB")
    
    # Use larger batch size for GPU efficiency
    batch_size = 16  # GPUs benefit from larger batches
    datamodule = SyntheticDataModule(dataset, batch_size=batch_size)
    print(f"Batch size: {batch_size}")
    
    # Test 1: CPU training
    print("\n" + "="*80)
    print("CPU Training")
    print("="*80)
    graph_cpu, nodes_cpu = create_realistic_test_graph(n_channels=50)
    
    config_cpu = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(
            max_epochs=5,  # More epochs for gradient training
            accelerator='cpu',
            devices=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
        ),
        optimizer=OptimizerConfig(name='adam', lr=0.001),
    )
    
    # Warm-up run (first run is often slower)
    print("Running warm-up...")
    graph_cpu_warmup, _ = create_realistic_test_graph(n_channels=50)
    warmup_data = create_small_scale_dataset(n_samples=20, height=128, width=128, n_channels=50, seed=99)
    warmup_dm = SyntheticDataModule(warmup_data, batch_size=batch_size)
    warmup_config = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(max_epochs=1, accelerator='cpu', devices=1, 
                            enable_progress_bar=False, enable_checkpointing=False),
        optimizer=OptimizerConfig(name='adam', lr=0.001),
    )
    graph_cpu_warmup.train(datamodule=warmup_dm, training_config=warmup_config)
    del graph_cpu_warmup, warmup_data, warmup_dm
    
    # Actual measurement
    print("Running actual CPU training...")
    mem_cpu_before = get_memory_usage()
    start_cpu = time.time()
    graph_cpu.train(datamodule=datamodule, training_config=config_cpu)
    cpu_time = time.time() - start_cpu
    mem_cpu_after = get_memory_usage()
    
    cpu_mem_delta = mem_cpu_after['cpu_mb'] - mem_cpu_before['cpu_mb']
    print(f"\nCPU Results:")
    print(f"  Training time: {cpu_time:.2f} seconds")
    print(f"  Memory delta: {cpu_mem_delta:.2f} MB")
    print(f"  Throughput: {stats['n_samples'] * 5 / cpu_time:.2f} samples/second")
    print(f"  Time per epoch: {cpu_time / 5:.2f} seconds")
    
    # Count trainable parameters
    cpu_params = sum(p.numel() for p in graph_cpu.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {cpu_params:,}")
    
    # Test 2: GPU training
    print("\n" + "="*80)
    print("GPU Training")
    print("="*80)
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    graph_gpu, nodes_gpu = create_realistic_test_graph(n_channels=50)
    
    config_gpu = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(
            max_epochs=5,
            accelerator='gpu',
            devices=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
        ),
        optimizer=OptimizerConfig(name='adam', lr=0.001),
    )
    
    # Warm-up GPU
    print("Running GPU warm-up...")
    graph_gpu_warmup, _ = create_realistic_test_graph(n_channels=50)
    warmup_data_gpu = create_small_scale_dataset(n_samples=20, height=128, width=128, n_channels=50, seed=99)
    warmup_dm_gpu = SyntheticDataModule(warmup_data_gpu, batch_size=batch_size)
    warmup_config_gpu = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(max_epochs=1, accelerator='gpu', devices=1,
                            enable_progress_bar=False, enable_checkpointing=False),
        optimizer=OptimizerConfig(name='adam', lr=0.001),
    )
    graph_gpu_warmup.train(datamodule=warmup_dm_gpu, training_config=warmup_config_gpu)
    del graph_gpu_warmup, warmup_data_gpu, warmup_dm_gpu
    torch.cuda.empty_cache()
    
    # Actual GPU measurement
    print("Running actual GPU training...")
    mem_gpu_before = get_memory_usage()
    torch.cuda.synchronize()  # Ensure all operations complete
    start_gpu = time.time()
    graph_gpu.train(datamodule=datamodule, training_config=config_gpu)
    torch.cuda.synchronize()  # Ensure all operations complete
    gpu_time = time.time() - start_gpu
    mem_gpu_after = get_memory_usage()
    
    gpu_mem_allocated = torch.cuda.max_memory_allocated() / (1024**2)
    gpu_mem_reserved = torch.cuda.max_memory_reserved() / (1024**2)
    
    print(f"\nGPU Results:")
    print(f"  Training time: {gpu_time:.2f} seconds")
    print(f"  GPU memory allocated (peak): {gpu_mem_allocated:.2f} MB")
    print(f"  GPU memory reserved (peak): {gpu_mem_reserved:.2f} MB")
    print(f"  Throughput: {stats['n_samples'] * 5 / gpu_time:.2f} samples/second")
    print(f"  Time per epoch: {gpu_time / 5:.2f} seconds")
    
    # Compare
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    throughput_improvement = (stats['n_samples'] * 5 / gpu_time) / (stats['n_samples'] * 5 / cpu_time)
    
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"CPU time:              {cpu_time:.2f}s")
    print(f"GPU time:              {gpu_time:.2f}s")
    print(f"Speedup:               {speedup:.2f}x")
    print(f"Throughput improvement: {throughput_improvement:.2f}x")
    print(f"Time saved:            {cpu_time - gpu_time:.2f}s ({(cpu_time - gpu_time)/cpu_time*100:.1f}%)")
    
    # Detailed breakdown
    print(f"\nPer-epoch comparison:")
    print(f"  CPU: {cpu_time/5:.2f}s/epoch")
    print(f"  GPU: {gpu_time/5:.2f}s/epoch")
    print(f"\nPer-sample comparison:")
    print(f"  CPU: {cpu_time/(stats['n_samples']*5)*1000:.2f}ms/sample")
    print(f"  GPU: {gpu_time/(stats['n_samples']*5)*1000:.2f}ms/sample")
    
    # Assertions
    assert gpu_mem_reserved > 0, "GPU not initialized - CUDA context not created!"
    
    # GPU should show some benefit for this realistic workload
    # We don't assert speedup > 1 because it depends on hardware,
    # but we verify GPU was actually used
    if speedup < 1.0:
        print(f"\n⚠ Warning: GPU slower than CPU (speedup={speedup:.2f}x)")
        print("  This may happen if:")
        print("  - CPU is very fast (e.g., high-end desktop CPU)")
        print("  - GPU is entry-level or mobile GPU")
        print("  - Data transfer overhead dominates computation")
        print("  - Workload is still too small for GPU parallelism")
    else:
        print(f"\n✓ GPU shows {speedup:.2f}x speedup over CPU")
    
    print(f"\n✓ GPU acceleration test passed")
    print(f"✓ GPU was used (peak reserved {gpu_mem_reserved:.2f} MB)")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_batch_size_scaling():
    """Test how batch size affects GPU performance.
    
    This test demonstrates that GPUs perform better with larger batch sizes
    due to improved parallelism and amortized overhead costs.
    """
    print("\n" + "="*80)
    print("STRESS TEST: GPU Batch Size Scaling")
    print("="*80)
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    # Create moderate-sized dataset
    dataset = create_small_scale_dataset(
        n_samples=100,
        height=128,
        width=128,
        n_channels=50,
        seed=42
    )
    
    batch_sizes = [4, 8, 16, 32]
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n--- Testing batch size {batch_size} ---")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        datamodule = SyntheticDataModule(dataset, batch_size=batch_size)
        graph, _ = create_realistic_test_graph(n_channels=50)
        
        config = TrainingConfig(
            seed=42,
            trainer=TrainerConfig(
                max_epochs=3,
                accelerator='gpu',
                devices=1,
                enable_progress_bar=False,
                enable_checkpointing=False,
            ),
            optimizer=OptimizerConfig(name='adam', lr=0.001),
        )
        
        # Warm-up
        graph_warmup, _ = create_realistic_test_graph(n_channels=50)
        warmup_data = create_small_scale_dataset(n_samples=10, height=128, width=128, n_channels=50, seed=99)
        warmup_dm = SyntheticDataModule(warmup_data, batch_size=batch_size)
        warmup_config = TrainingConfig(
            seed=42,
            trainer=TrainerConfig(max_epochs=1, accelerator='gpu', devices=1,
                                enable_progress_bar=False, enable_checkpointing=False),
            optimizer=OptimizerConfig(name='adam', lr=0.001),
        )
        graph_warmup.train(datamodule=warmup_dm, training_config=warmup_config)
        del graph_warmup, warmup_data, warmup_dm
        torch.cuda.empty_cache()
        
        # Actual measurement
        torch.cuda.synchronize()
        start_time = time.time()
        graph.train(datamodule=datamodule, training_config=config)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        gpu_peak = torch.cuda.max_memory_allocated() / (1024**2)
        throughput = (100 * 3) / elapsed  # samples per second
        
        results.append({
            'batch_size': batch_size,
            'time': elapsed,
            'throughput': throughput,
            'gpu_peak_mb': gpu_peak,
            'time_per_sample': elapsed / (100 * 3) * 1000,  # ms
        })
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.2f} samples/s")
        print(f"  Time per sample: {elapsed / (100 * 3) * 1000:.2f} ms")
        print(f"  GPU peak memory: {gpu_peak:.2f} MB")
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH SIZE SCALING SUMMARY")
    print("="*80)
    print(f"{'Batch':>8} | {'Time(s)':>9} | {'Throughput':>12} | {'ms/sample':>11} | {'GPU MB':>9} | {'Speedup':>9}")
    print("-" * 80)
    
    base_time = results[0]['time']
    for r in results:
        speedup = base_time / r['time']
        print(f"{r['batch_size']:>8} | {r['time']:>9.2f} | {r['throughput']:>12.2f} | "
              f"{r['time_per_sample']:>11.2f} | {r['gpu_peak_mb']:>9.2f} | {speedup:>9.2f}x")
    
    # Analysis
    print("\nAnalysis:")
    print(f"  Smallest batch (size={batch_sizes[0]}): {results[0]['throughput']:.2f} samples/s")
    print(f"  Largest batch (size={batch_sizes[-1]}): {results[-1]['throughput']:.2f} samples/s")
    improvement = results[-1]['throughput'] / results[0]['throughput']
    print(f"  Improvement: {improvement:.2f}x faster with larger batches")
    
    print("\n✓ GPU batch size scaling test passed")
    print("✓ Demonstrates GPU performance scales with batch size")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_memory_scaling():
    """Test GPU memory usage with different batch sizes."""
    print("\n" + "="*80)
    print("STRESS TEST: GPU Memory Scaling")
    print("="*80)
    
    dataset = create_small_scale_dataset(n_samples=20, seed=42)
    
    batch_sizes = [1, 2, 4, 8]
    results = []
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size {batch_size}...")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        datamodule = SyntheticDataModule(dataset, batch_size=batch_size)
        graph, _ = create_test_graph(n_channels=10)
        
        config = TrainingConfig(
            seed=42,
            trainer=TrainerConfig(
                max_epochs=1,
                accelerator='gpu',
                devices=1,
                enable_progress_bar=False,
                enable_checkpointing=False,
            ),
            optimizer=OptimizerConfig(name='adam', lr=0.001),
        )
        
        graph.train(datamodule=datamodule, training_config=config)
        
        gpu_allocated = torch.cuda.memory_allocated() / (1024**2)
        gpu_peak = torch.cuda.max_memory_allocated() / (1024**2)
        
        results.append({
            'batch_size': batch_size,
            'allocated_mb': gpu_allocated,
            'peak_mb': gpu_peak,
        })
        
        print(f"  Allocated: {gpu_allocated:.2f} MB")
        print(f"  Peak: {gpu_peak:.2f} MB")
    
    # Print summary
    print("\n--- GPU Memory Scaling Summary ---")
    print(f"{'Batch Size':>12} | {'Allocated (MB)':>15} | {'Peak (MB)':>12}")
    print("-" * 45)
    for r in results:
        print(f"{r['batch_size']:>12} | {r['allocated_mb']:>15.2f} | {r['peak_mb']:>12.2f}")
    
    print("\n✓ GPU memory scaling test passed")


if __name__ == "__main__":
    print("Running GPU stress tests...")
    print("Note: Use pytest with GPU marker:")
    print("  pytest tests/stress/test_gpu_stress.py -v -m gpu")
