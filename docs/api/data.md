# Data

Complete API reference for data loading, datasets, and preprocessing.

## Overview

The `cuvis_ai.data` module provides dataset implementations and utilities for loading hyperspectral data.

### Key Features

- **Dictionary-based Batches**: Consistent batch format across all datasets
- **PyTorch Integration**: Compatible with `torch.utils.data.DataLoader`
- **Lightning DataModule**: Drop-in compatibility with PyTorch Lightning
- **Built-in Datasets**: Public datasets for benchmarking and examples

### Batch Format

All datasets return dictionary batches with standardized keys:

```python
batch = {
    "cube": torch.Tensor,      # Shape: [B, H, W, C] or [B, C, H, W]
    "mask": torch.Tensor,      # Optional: [B, H, W] or [B, 1, H, W]
    "labels": torch.Tensor,    # Optional: [B] or [B, H, W]
    "metadata": Dict[str, Any] # Optional: Additional info
}
```

## Quick Example

```python
from cuvis_ai.data.lentils_anomaly import LentilsAnomaly
from torch.utils.data import DataLoader

# Create dataset
dataset = LentilsAnomaly(
    data_dir="./data/Lentils",
    batch_size=4,
    num_workers=2
)

# Setup and get loaders
dataset.setup()
train_loader = dataset.train_dataloader()
val_loader = dataset.val_dataloader()

# Iterate
for batch in train_loader:
    cube = batch["cube"]    # [B, H, W, C]
    mask = batch["mask"]    # [B, H, W]
    # Process batch...
```

## Base Dataset Classes

### BaseDataset

Abstract base class for all CUVIS.AI datasets.

**Key Methods:**
- `__getitem__()`: Return single sample as dictionary
- `__len__()`: Dataset size
- `get_metadata()`: Return dataset metadata

**Contract:**
- Must return dictionary with at least `"cube"` or `"x"` key
- Optional keys: `"mask"`, `"labels"`, `"metadata"`

::: cuvis_ai.data.datasets

## Built-in Datasets

### LentilsAnomaly

Lentils anomaly detection dataset.

**Description:**
Hyperspectral images of lentils with anomaly annotations for quality control tasks.

**Features:**
- Train/val split
- Anomaly masks
- Configurable batch size
- Multi-worker support

**Usage:**
```python
from cuvis_ai.data.lentils_anomaly import LentilsAnomaly

datamodule = LentilsAnomaly(
    data_dir="./data/Lentils",
    batch_size=4,
    num_workers=2,
    train_val_split=0.8
)

datamodule.setup()
train_loader = datamodule.train_dataloader()
```

**Data Format:**
- Cube shape: [H, W, C] where C is spectral channels
- Mask shape: [H, W] with binary anomaly labels (0=normal, 1=anomaly)

::: cuvis_ai.data.lentils_anomaly

### Public Datasets

Access to public hyperspectral datasets for benchmarking.

**Available Datasets:**
- Indian Pines
- Pavia University
- Salinas
- Custom dataset loaders

**Usage:**
```python
from cuvis_ai.data.public_datasets import load_indian_pines

dataset = load_indian_pines(data_dir="./data")
```

::: cuvis_ai.data.public_datasets

## Creating Custom Datasets

### Example: Custom Hyperspectral Dataset

```python
from torch.utils.data import Dataset
import numpy as np
import torch

class MyHyperspectralDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        # Load file list, metadata, etc.
        self.samples = self._load_samples()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load hyperspectral cube
        cube = self._load_cube(idx)  # Shape: [H, W, C]
        
        # Load mask if available
        mask = self._load_mask(idx)  # Shape: [H, W]
        
        # Convert to tensors
        cube = torch.from_numpy(cube).float()
        mask = torch.from_numpy(mask).long()
        
        # Return dictionary
        return {
            "cube": cube,
            "mask": mask,
            "metadata": {
                "sample_id": idx,
                "filename": self.samples[idx]
            }
        }
    
    def _load_cube(self, idx):
        # Your loading logic here
        pass
    
    def _load_mask(self, idx):
        # Your loading logic here
        pass
```

### Example: Custom Lightning DataModule

```python
from cuvis_ai.training.datamodule import GraphDataModule
from torch.utils.data import DataLoader

class MyDataModule(GraphDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 4,
        num_workers: int = 2
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        # Create train/val datasets
        if stage == "fit" or stage is None:
            self.train_dataset = MyHyperspectralDataset(
                data_dir=f"{self.data_dir}/train"
            )
            self.val_dataset = MyHyperspectralDataset(
                data_dir=f"{self.data_dir}/val"
            )
        
        # Create test dataset
        if stage == "test" or stage is None:
            self.test_dataset = MyHyperspectralDataset(
                data_dir=f"{self.data_dir}/test"
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
```

## Data Loading Best Practices

### Memory Management

For large hyperspectral datasets:

```python
# Use memory mapping for large files
import numpy as np
cube = np.memmap(
    filename,
    dtype='float32',
    mode='r',
    shape=(height, width, channels)
)
```

### Multi-Worker Loading

```python
# Enable multi-worker loading for faster training
datamodule = LentilsAnomaly(
    data_dir="./data",
    batch_size=4,
    num_workers=4  # Adjust based on CPU cores
)
```

### Persistent Workers

```python
# For Windows or when workers are slow to spawn
train_loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=2,
    persistent_workers=True  # Keep workers alive between epochs
)
```

## Data Augmentation

### Example Augmentation Pipeline

```python
import torch
import torchvision.transforms.functional as TF

class HyperspectralAugmentation:
    def __init__(self, flip_prob=0.5, rotate_prob=0.5):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
    
    def __call__(self, batch):
        cube = batch["cube"]  # [B, H, W, C]
        mask = batch["mask"]  # [B, H, W]
        
        # Random horizontal flip
        if torch.rand(1) < self.flip_prob:
            cube = torch.flip(cube, dims=[2])
            mask = torch.flip(mask, dims=[2])
        
        # Random rotation (90, 180, 270)
        if torch.rand(1) < self.rotate_prob:
            k = torch.randint(1, 4, (1,)).item()
            cube = torch.rot90(cube, k=k, dims=[1, 2])
            mask = torch.rot90(mask, k=k, dims=[1, 2])
        
        batch["cube"] = cube
        batch["mask"] = mask
        return batch
```

## Troubleshooting

### Windows Multi-Worker Issues

**Problem:** `num_workers > 0` causes errors on Windows.

**Solutions:**
```python
# Option 1: Disable multi-worker
datamodule = LentilsAnomaly(num_workers=0)

# Option 2: Use persistent workers
train_loader = DataLoader(
    dataset,
    num_workers=2,
    persistent_workers=True
)
```

### Memory Issues

**Problem:** Dataset doesn't fit in RAM.

**Solutions:**
```python
# Option 1: Use streaming dataset
from torch.utils.data import IterableDataset

class StreamingDataset(IterableDataset):
    def __iter__(self):
        # Yield samples one at a time from disk
        pass

# Option 2: Reduce batch size
datamodule = LentilsAnomaly(batch_size=1)

# Option 3: Use memory mapping
# See "Memory Management" section above
```

## See Also

- **[Training API](training.md)**: Use datasets with training pipeline
- **[Configuration Guide](../user-guide/configuration.md)**: Configure datasets with Hydra
- **[Quickstart Tutorial](../user-guide/quickstart.md)**: Basic data loading example
