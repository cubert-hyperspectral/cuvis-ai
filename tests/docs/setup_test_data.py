#!/usr/bin/env python3
"""Quick setup script for minimal test data infrastructure.

This script creates the basic directory structure and mock files needed
to validate documentation examples that reference data files.

Usage:
    uv run python tests/docs/setup_test_data.py
"""

import json
from pathlib import Path

# Base directories
REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def create_directory_structure():
    """Create required directory structure."""
    directories = [
        DATA_DIR / "lentils",
        DATA_DIR / "test_samples",
        OUTPUTS_DIR / "trained_models",
        OUTPUTS_DIR / "trainrun",
        OUTPUTS_DIR / "channel_selector" / "trained_models",
    ]

    print("Creating directory structure...")
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {dir_path.relative_to(REPO_ROOT)}")


def create_mock_cu3s_files():
    """Create placeholder CU3S files."""
    cu3s_files = [
        DATA_DIR / "lentils" / "Demo_000.cu3s",
        DATA_DIR / "lentils" / "Demo_001.cu3s",
        DATA_DIR / "test_samples" / "small_cube.cu3s",
        DATA_DIR / "test_samples" / "single_pixel.cu3s",
    ]

    print("\nCreating mock CU3S files...")
    for cu3s_file in cu3s_files:
        if not cu3s_file.exists():
            # Create a mock binary file with header
            with open(cu3s_file, "wb") as f:
                # Write simple header
                f.write(b"MOCK_CU3S_v1.0\n")
                f.write(b"WIDTH: 640\n")
                f.write(b"HEIGHT: 512\n")
                f.write(b"CHANNELS: 125\n")
                f.write(b"WAVELENGTH_MIN: 450\n")
                f.write(b"WAVELENGTH_MAX: 950\n")
                f.write(b"\x00" * 100)  # Some mock binary data
            print(f"  [OK] {cu3s_file.relative_to(REPO_ROOT)}")
        else:
            print(f"  - {cu3s_file.relative_to(REPO_ROOT)} (exists)")


def create_annotations():
    """Create COCO format annotations."""
    annotations_file = DATA_DIR / "lentils" / "annotations.json"

    print("\nCreating annotation files...")
    if not annotations_file.exists():
        annotations = {
            "info": {
                "description": "Mock lentils dataset for documentation testing",
                "version": "1.0",
                "year": 2026,
            },
            "images": [
                {
                    "id": 1,
                    "file_name": "Demo_000.cu3s",
                    "width": 640,
                    "height": 512,
                },
                {
                    "id": 2,
                    "file_name": "Demo_001.cu3s",
                    "width": 640,
                    "height": 512,
                },
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 100, 50, 50],
                    "area": 2500,
                    "segmentation": [],
                    "iscrowd": 0,
                }
            ],
            "categories": [
                {"id": 1, "name": "defect", "supercategory": "anomaly"},
                {"id": 2, "name": "normal", "supercategory": "background"},
            ],
        }

        with open(annotations_file, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2)
        print(f"  [OK] {annotations_file.relative_to(REPO_ROOT)}")
    else:
        print(f"  - {annotations_file.relative_to(REPO_ROOT)} (exists)")


def create_metadata():
    """Create dataset metadata YAML files."""
    metadata_file = DATA_DIR / "lentils" / "metadata.yaml"

    print("\nCreating metadata files...")
    if not metadata_file.exists():
        metadata_content = """dataset:
  name: lentils_demo
  version: "1.0"
  description: "Mock dataset for documentation testing"
  num_samples: 2

camera:
  model: "Cubert S185"
  fps: 30
  integration_time_ms: 10
  wavelength_range: [450, 950]
  spectral_channels: 125
  spatial_resolution: [640, 512]

preprocessing:
  reflectance_calibration: "white_reference_20231115.dat"
  dark_current_correction: true
  bad_pixel_correction: true

classes:
  - id: 1
    name: "defect"
  - id: 2
    name: "normal"
"""
        with open(metadata_file, "w", encoding="utf-8") as f:
            f.write(metadata_content)
        print(f"  [OK] {metadata_file.relative_to(REPO_ROOT)}")
    else:
        print(f"  - {metadata_file.relative_to(REPO_ROOT)} (exists)")


def create_pipeline_configs():
    """Create mock pipeline configuration files."""
    configs = {
        OUTPUTS_DIR / "trained_models" / "channel_selector.yaml": """pipeline:
  name: channel_selector
  version: "1.0"
  nodes:
    - id: 0
      type: ChannelSelector
      params:
        selected_channels: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    - id: 1
      type: Normalizer
      params:
        method: "minmax"
        feature_range: [0, 1]
    - id: 2
      type: Flatten
""",
        OUTPUTS_DIR / "trained_models" / "deep_svdd.yaml": """pipeline:
  name: deep_svdd
  version: "1.0"
  nodes:
    - id: 0
      type: Normalizer
      params:
        method: "standard"
    - id: 1
      type: DeepSVDD
      params:
        latent_dim: 32
        nu: 0.1
""",
        OUTPUTS_DIR / "trainrun" / "trainrun.yaml": """trainrun:
  pipeline_path: "outputs/trained_models/channel_selector.yaml"
  weights_path: "outputs/trained_models/channel_selector.pt"
  dataset: "lentils"
  mode: "train"
  training:
    epochs: 100
    batch_size: 32
    learning_rate: 0.001
  validation:
    split: 0.2
    metric: "accuracy"
""",
        OUTPUTS_DIR / "channel_selector" / "trained_models" / "trainrun.yaml": """trainrun:
  pipeline_path: "outputs/channel_selector/trained_models/pipeline.yaml"
  weights_path: "outputs/channel_selector/trained_models/weights.pt"
  dataset: "lentils"
  mode: "validate"
  created_at: "2026-02-05T12:00:00"
""",
    }

    print("\nCreating pipeline configuration files...")
    for config_file, content in configs.items():
        if not config_file.exists():
            with open(config_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  [OK] {config_file.relative_to(REPO_ROOT)}")
        else:
            print(f"  - {config_file.relative_to(REPO_ROOT)} (exists)")


def create_mock_weights():
    """Create mock PyTorch weights files."""
    try:
        import torch

        weights_files = [
            OUTPUTS_DIR / "trained_models" / "channel_selector.pt",
            OUTPUTS_DIR / "trained_models" / "deep_svdd.pt",
            OUTPUTS_DIR / "channel_selector" / "trained_models" / "weights.pt",
        ]

        print("\nCreating mock PyTorch weights...")
        for weights_file in weights_files:
            if not weights_file.exists():
                # Create a minimal state dict
                state_dict = {
                    "layer1.weight": torch.randn(64, 10),
                    "layer1.bias": torch.randn(64),
                    "layer2.weight": torch.randn(32, 64),
                    "layer2.bias": torch.randn(32),
                    "output.weight": torch.randn(10, 32),
                    "output.bias": torch.randn(10),
                }
                torch.save(state_dict, weights_file)
                print(f"  [OK] {weights_file.relative_to(REPO_ROOT)}")
            else:
                print(f"  - {weights_file.relative_to(REPO_ROOT)} (exists)")
    except ImportError:
        print("  [!] PyTorch not available - skipping weight files")
        print("    Install with: uv sync")


def create_readme():
    """Create README in data directory."""
    readme_file = DATA_DIR / "README.md"

    readme_content = """# Test Data Directory

This directory contains mock data files for validating documentation examples.

## Contents

- `lentils/` - Mock lentils dataset
  - `Demo_000.cu3s`, `Demo_001.cu3s` - Mock hyperspectral files
  - `annotations.json` - COCO format annotations
  - `metadata.yaml` - Dataset metadata

- `test_samples/` - Minimal test files
  - `small_cube.cu3s` - Small mock hyperspectral cube
  - `single_pixel.cu3s` - Single pixel for unit tests

## Note

These are **mock files** created by `tests/docs/setup_test_data.py`.
They contain minimal data sufficient for:
- Path validation
- File existence checks
- Basic syntax testing

For actual hyperspectral data analysis, you need real CU3S files from
hyperspectral cameras or sample datasets.

## Setup

To regenerate this structure:
```bash
uv run python tests/docs/setup_test_data.py
```

See `tests/docs/SETUP_TEST_DATA.md` for more options.
"""

    print("\nCreating README...")
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"  [OK] {readme_file.relative_to(REPO_ROOT)}")


def print_summary():
    """Print summary of created files."""
    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)

    # Count created files
    num_cu3s = len(list(DATA_DIR.rglob("*.cu3s")))
    num_yaml = len(list((OUTPUTS_DIR).rglob("*.yaml")))
    num_pt = len(list((OUTPUTS_DIR).rglob("*.pt")))

    print("\nCreated test data infrastructure:")
    print(f"  - {num_cu3s} CU3S files (mock)")
    print(f"  - {num_yaml} YAML configuration files")
    print(f"  - {num_pt} PyTorch weight files (mock)")
    print("  - 1 COCO annotations file")
    print("  - 1 metadata file")

    print("\nNext steps:")
    print("  1. Validate setup:")
    print(
        "     uv run python -c \"from pathlib import Path; print(Path('data/lentils/Demo_000.cu3s').exists())\""
    )
    print("\n  2. Run documentation tests:")
    print("     uv run pytest tests/docs/ -v")
    print("\n  3. For advanced testing, see:")
    print("     tests/docs/SETUP_TEST_DATA.md")


def main():
    """Main setup function."""
    print("=" * 70)
    print("CUVIS-AI TEST DATA SETUP")
    print("=" * 70)
    print("\nThis script creates minimal mock data for documentation testing.")
    print("It does NOT provide real hyperspectral data.\n")

    create_directory_structure()
    create_mock_cu3s_files()
    create_annotations()
    create_metadata()
    create_pipeline_configs()
    create_mock_weights()
    create_readme()
    print_summary()


if __name__ == "__main__":
    main()
