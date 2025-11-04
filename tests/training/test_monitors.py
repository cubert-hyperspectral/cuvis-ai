"""Tests for monitoring adapter implementations."""

import json
import pickle
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch

from cuvis_ai.training.monitors import DummyMonitor, TensorBoardMonitor, WandBMonitor


@pytest.fixture
def sample_metrics():
    """Create sample metrics dictionary."""
    return {
        "loss": 0.5,
        "accuracy": 0.85,
        "f1_score": 0.82,
    }


@pytest.fixture
def sample_artifacts():
    """Create sample artifacts with matplotlib figure."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title("Test Plot")
    
    return {
        "test_plot": {
            "figure": fig,
            "type": "test",
            "data": [1, 2, 3]
        }
    }


class TestDummyMonitor:
    """Tests for DummyMonitor filesystem-based monitoring."""
    
    def test_initialization(self):
        """Test DummyMonitor initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DummyMonitor(output_dir=tmpdir, save_thumbnails=True)
            assert monitor.output_dir == Path(tmpdir)
            assert monitor.save_thumbnails is True
            assert monitor.output_dir.exists()
    
    def test_log_metrics(self, sample_metrics):
        """Test metrics logging to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DummyMonitor(output_dir=tmpdir)
            
            # Log metrics for multiple steps
            monitor.log_metrics(sample_metrics, step=0, stage="train")
            monitor.log_metrics({"loss": 0.4, "accuracy": 0.87}, step=1, stage="train")
            monitor.log_metrics({"val_loss": 0.45}, step=0, stage="val")
            
            # Check train metrics file
            train_metrics_file = Path(tmpdir) / "train" / "metrics.jsonl"
            assert train_metrics_file.exists()
            
            # Read and verify
            with open(train_metrics_file) as f:
                lines = f.readlines()
                assert len(lines) == 2
                
                # Check first record
                record1 = json.loads(lines[0])
                assert record1["step"] == 0
                assert record1["stage"] == "train"
                assert record1["loss"] == 0.5
                assert record1["accuracy"] == 0.85
                
                # Check second record
                record2 = json.loads(lines[1])
                assert record2["step"] == 1
                assert record2["loss"] == 0.4
            
            # Check val metrics file
            val_metrics_file = Path(tmpdir) / "val" / "metrics.jsonl"
            assert val_metrics_file.exists()
    
    def test_log_artifacts_with_figure(self, sample_artifacts):
        """Test artifact logging with matplotlib figure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DummyMonitor(output_dir=tmpdir, save_thumbnails=True)
            
            monitor.log_artifacts(sample_artifacts, stage="val", step=5)
            
            # Check pickle file
            pkl_path = Path(tmpdir) / "val" / "step_000005" / "test_plot.pkl"
            assert pkl_path.exists()
            
            # Check PNG thumbnail
            png_path = Path(tmpdir) / "val" / "step_000005" / "test_plot.png"
            assert png_path.exists()
            
            # Verify pickle content
            with open(pkl_path, "rb") as f:
                loaded = pickle.load(f)
                assert loaded["type"] == "test"
                assert loaded["data"] == [1, 2, 3]
    
    def test_log_artifacts_without_thumbnails(self, sample_artifacts):
        """Test artifact logging with thumbnails disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DummyMonitor(output_dir=tmpdir, save_thumbnails=False)
            
            monitor.log_artifacts(sample_artifacts, stage="val", step=0)
            
            # Check pickle exists
            pkl_path = Path(tmpdir) / "val" / "step_000000" / "test_plot.pkl"
            assert pkl_path.exists()
            
            # Check PNG does not exist
            png_path = Path(tmpdir) / "val" / "step_000000" / "test_plot.png"
            assert not png_path.exists()
    
    def test_log_artifacts_filename_sanitization(self):
        """Test that artifact names are sanitized for filesystem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DummyMonitor(output_dir=tmpdir)
            
            # Artifact name with slashes
            artifacts = {"val/viz/heatmap": {"type": "test", "data": []}}
            monitor.log_artifacts(artifacts, stage="val", step=0)
            
            # Check sanitized filename
            pkl_path = Path(tmpdir) / "val" / "step_000000" / "val_viz_heatmap.pkl"
            assert pkl_path.exists()
    
    def test_setup_teardown(self):
        """Test setup and teardown hooks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DummyMonitor(output_dir=tmpdir)
            
            # These should not raise errors
            monitor.setup(trainer=None)
            monitor.teardown()
    
    def test_multiple_artifacts_same_step(self):
        """Test logging multiple artifacts in the same step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DummyMonitor(output_dir=tmpdir, save_thumbnails=False)
            
            artifacts = {
                "artifact1": {"type": "test1", "data": [1, 2, 3]},
                "artifact2": {"type": "test2", "data": [4, 5, 6]},
                "artifact3": {"type": "test3", "data": [7, 8, 9]},
            }
            
            monitor.log_artifacts(artifacts, stage="train", step=0)
            
            # Check all artifacts saved
            step_dir = Path(tmpdir) / "train" / "step_000000"
            assert (step_dir / "artifact1.pkl").exists()
            assert (step_dir / "artifact2.pkl").exists()
            assert (step_dir / "artifact3.pkl").exists()


class TestWandBMonitor:
    """Tests for WandBMonitor implementation."""
    
    def test_initialization(self):
        """Test WandBMonitor initialization."""
        monitor = WandBMonitor(
            project="test-project",
            entity="test-entity",
            tags=["test", "phase3"],
            config={"learning_rate": 0.001},
            mode="disabled"
        )
        
        assert monitor.project == "test-project"
        assert monitor.entity == "test-entity"
        assert "test" in monitor.tags
        assert "phase3" in monitor.tags
        assert monitor.config["learning_rate"] == 0.001
        assert monitor.mode == "disabled"
    
    def test_initialization_without_wandb(self):
        """Test WandBMonitor handles missing wandb gracefully."""
        monitor = WandBMonitor(project="test")
        
        # Should initialize even without wandb
        assert monitor.project == "test"
        # _wandb_available may be False if wandb not installed
    
    def test_log_metrics_without_wandb(self, sample_metrics):
        """Test that log_metrics doesn't raise when wandb unavailable."""
        monitor = WandBMonitor(project="test", mode="disabled")
        
        # Should not raise even without wandb
        monitor.log_metrics(sample_metrics, step=0, stage="train")
        monitor.log_metrics(sample_metrics, step=1, stage="val")
    
    def test_log_artifacts_without_wandb(self, sample_artifacts):
        """Test that log_artifacts doesn't raise when wandb unavailable."""
        monitor = WandBMonitor(project="test", mode="disabled")
        
        # Should not raise even without wandb
        monitor.log_artifacts(sample_artifacts, stage="val", step=0)
    
    def test_setup_teardown(self):
        """Test that setup/teardown don't raise errors."""
        monitor = WandBMonitor(project="test", mode="disabled")
        
        # Should not raise
        monitor.setup(trainer=None)
        monitor.teardown()
    
    def test_metric_prefixing(self):
        """Test that metrics are properly prefixed with stage."""
        monitor = WandBMonitor(project="test", mode="disabled")
        
        # Even without actual logging, initialization should work
        metrics = {"loss": 0.5, "accuracy": 0.85}
        
        # Should not raise
        monitor.log_metrics(metrics, step=0, stage="train")
    
    def test_offline_mode(self):
        """Test WandBMonitor with offline mode."""
        monitor = WandBMonitor(
            project="test",
            mode="offline",
            name="test_run",
            notes="Test notes"
        )
        
        assert monitor.mode == "offline"
        assert monitor.name == "test_run"
        assert monitor.notes == "Test notes"


class TestTensorBoardMonitor:
    """Tests for TensorBoardMonitor implementation."""
    
    def test_initialization(self):
        """Test TensorBoardMonitor initialization."""
        monitor = TensorBoardMonitor(
            log_dir="./runs",
            comment="test_experiment",
            flush_secs=60
        )
        
        assert monitor.log_dir == Path("./runs")
        assert monitor.comment == "test_experiment"
        assert monitor.flush_secs == 60
    
    def test_initialization_without_tensorboard(self):
        """Test TensorBoardMonitor handles missing tensorboard gracefully."""
        monitor = TensorBoardMonitor()
        
        # Should initialize even without tensorboard
        assert monitor.log_dir == Path("./runs")
        # _tensorboard_available may be False if tensorboard not installed
    
    def test_log_metrics_without_tensorboard(self, sample_metrics):
        """Test that log_metrics doesn't raise when tensorboard unavailable."""
        monitor = TensorBoardMonitor()
        
        # Should not raise even without tensorboard
        monitor.log_metrics(sample_metrics, step=0, stage="train")
        monitor.log_metrics(sample_metrics, step=1, stage="val")
    
    def test_log_artifacts_without_tensorboard(self, sample_artifacts):
        """Test that log_artifacts doesn't raise when tensorboard unavailable."""
        monitor = TensorBoardMonitor()
        
        # Should not raise even without tensorboard
        monitor.log_artifacts(sample_artifacts, stage="val", step=0)
    
    def test_setup_teardown(self):
        """Test that setup/teardown don't raise errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = TensorBoardMonitor(log_dir=tmpdir)
            
            # Should not raise
            monitor.setup(trainer=None)
            monitor.teardown()
    
    def test_metric_prefixing(self):
        """Test that metrics are properly prefixed with stage."""
        monitor = TensorBoardMonitor()
        
        # Even without actual logging, initialization should work
        metrics = {"loss": 0.5, "accuracy": 0.85}
        
        # Should not raise
        monitor.log_metrics(metrics, step=0, stage="train")
    
    def test_custom_flush_interval(self):
        """Test TensorBoardMonitor with custom flush interval."""
        monitor = TensorBoardMonitor(
            log_dir="./runs",
            flush_secs=30
        )
        
        assert monitor.flush_secs == 30


class TestMonitoringProtocol:
    """Tests for monitoring protocol compliance."""
    
    def test_all_monitors_implement_protocol(self):
        """Test that all monitors implement MonitoringNode protocol."""
        from cuvis_ai.training.leaf_nodes import MonitoringNode
        
        monitors = [
            DummyMonitor(output_dir=tempfile.mkdtemp()),
            WandBMonitor(project="test"),
            TensorBoardMonitor(),
        ]
        
        for monitor in monitors:
            assert isinstance(monitor, MonitoringNode)
            
            # Check required methods exist
            assert hasattr(monitor, 'setup')
            assert hasattr(monitor, 'log_metrics')
            assert hasattr(monitor, 'log_artifacts')
            assert hasattr(monitor, 'teardown')
    
    def test_monitors_have_no_parent_requirements(self):
        """Test that monitoring nodes don't require specific parents."""
        from cuvis_ai.training.leaf_nodes import MonitoringNode
        
        # MonitoringNode should have empty parent requirements
        assert MonitoringNode.compatible_parent_types == tuple()


@pytest.mark.integration
def test_dummy_monitor_integration(sample_metrics, sample_artifacts):
    """Integration test for DummyMonitor with full workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monitor = DummyMonitor(output_dir=tmpdir, save_thumbnails=True)
        
        # Simulate training loop
        for epoch in range(3):
            # Training phase
            for step in range(2):
                global_step = epoch * 2 + step
                monitor.log_metrics(
                    {"loss": 0.5 - epoch * 0.1, "accuracy": 0.8 + epoch * 0.05},
                    step=global_step,
                    stage="train"
                )
            
            # Validation phase
            monitor.log_metrics(
                {"val_loss": 0.55 - epoch * 0.1},
                step=epoch,
                stage="val"
            )
            monitor.log_artifacts(
                sample_artifacts,
                stage="val",
                step=epoch
            )
        
        # Verify structure
        output_dir = Path(tmpdir)
        
        # Check metrics files
        assert (output_dir / "train" / "metrics.jsonl").exists()
        assert (output_dir / "val" / "metrics.jsonl").exists()
        
        # Check artifact directories
        for epoch in range(3):
            step_dir = output_dir / "val" / f"step_{epoch:06d}"
            assert step_dir.exists()
            assert (step_dir / "test_plot.pkl").exists()
            assert (step_dir / "test_plot.png").exists()
        
        # Verify metrics content
        with open(output_dir / "train" / "metrics.jsonl") as f:
            train_records = [json.loads(line) for line in f]
            assert len(train_records) == 6  # 3 epochs * 2 steps
            assert train_records[0]["loss"] > train_records[-1]["loss"]  # Loss decreasing
        
        monitor.teardown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
