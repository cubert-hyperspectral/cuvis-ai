#!/usr/bin/env python3
"""Download hyperspectral datasets from Hugging Face.

This script downloads datasets for CUVIS.AI documentation and testing.

Available datasets:
    - lentils: LentilsAnomaly dataset (nghorbani/LentilsAnomaly)

Usage:
    # Download default dataset (lentils)
    uv run python scripts/download_data.py

    # Download specific dataset
    uv run python scripts/download_data.py --dataset lentils

    # Specify custom data directory
    uv run python scripts/download_data.py --data-dir /path/to/data

    # Force re-download
    uv run python scripts/download_data.py --force
"""

import argparse
import shutil
from pathlib import Path

# Dataset configurations
DATASETS = {
    "lentils": {
        "repo_id": "nghorbani/LentilsAnomaly",
        "repo_type": "dataset",
        "target_dir": "Lentils",
        "description": "Lentils anomaly detection dataset (real hyperspectral data)",
        "size": "~200MB",
    },
    # Future datasets can be added here
    # "mnist-hsi": {
    #     "repo_id": "username/MNIST-HSI",
    #     "repo_type": "dataset",
    #     "target_dir": "MNIST-HSI",
    #     "description": "MNIST hyperspectral adaptation",
    #     "size": "~500MB",
    # },
}


def check_existing_data(data_dir: Path, dataset_name: str) -> dict:
    """Check for existing dataset directory.

    Args:
        data_dir: Base data directory
        dataset_name: Name of dataset to check

    Returns:
        Dictionary with status of dataset directory
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    target_dir = DATASETS[dataset_name]["target_dir"]
    status = {
        target_dir: (data_dir / target_dir).exists(),
    }
    return status


def create_symlink_if_needed(data_dir: Path) -> bool:
    """Create symlink from lentils -> Lentils for case-insensitive access.

    Args:
        data_dir: Base data directory

    Returns:
        True if symlink exists or was created, False otherwise
    """
    lentils_upper = data_dir / "Lentils"
    lentils_lower = data_dir / "lentils"

    if lentils_upper.exists() and not lentils_lower.exists():
        try:
            # On Windows, might need admin for symlinks
            lentils_lower.symlink_to("Lentils", target_is_directory=True)
            print("  [OK] Created symlink: lentils -> Lentils")
            return True
        except OSError as e:
            print(f"  [!] Could not create symlink: {e}")
            print("      Run as administrator or use --copy flag")
            return False
    elif lentils_lower.exists():
        print("  - lentils directory already exists")
        return True
    return False


def copy_if_symlink_fails(data_dir: Path) -> None:
    """Copy Lentils to lentils if symlink failed.

    Args:
        data_dir: Base data directory
    """
    lentils_upper = data_dir / "Lentils"
    lentils_lower = data_dir / "lentils"

    if lentils_upper.exists() and not lentils_lower.exists():
        print("\n  Copying Lentils -> lentils (this may take a while)...")
        shutil.copytree(lentils_upper, lentils_lower)
        print(f"  [OK] Copied {lentils_upper} to {lentils_lower}")


def download_from_huggingface(data_dir: Path, dataset_name: str, force: bool = False) -> bool:
    """Download dataset from Hugging Face.

    Args:
        data_dir: Base data directory
        dataset_name: Name of dataset to download
        force: Force re-download even if data exists

    Returns:
        True if download succeeded or data already exists, False otherwise
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    dataset_config = DATASETS[dataset_name]
    target_dir = data_dir / dataset_config["target_dir"]

    if target_dir.exists() and not force:
        print(f"\n  [OK] {dataset_name} data already exists at {target_dir}")
        print("      Use --force to re-download")
        return True

    try:
        from huggingface_hub import snapshot_download

        print(f"\nDownloading {dataset_name} dataset from Hugging Face...")
        print(f"  Description: {dataset_config['description']}")
        print(f"  Repository: {dataset_config['repo_id']}")
        print(f"  Size: {dataset_config['size']}")
        print(f"  Destination: {target_dir}")

        snapshot_download(
            repo_id=dataset_config["repo_id"],
            repo_type=dataset_config["repo_type"],
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )

        print(f"\n  [OK] Downloaded {dataset_name} dataset")

    except ImportError:
        print("\n  [!] huggingface_hub not installed")
        print("      Install with: uv pip install huggingface-hub")
        print("\n  Alternative: Download manually from:")
        print(f"      https://huggingface.co/datasets/{dataset_config['repo_id']}")
        print(f"      Extract to: {target_dir}")
        return False
    except Exception as e:
        print(f"\n  [!] Error downloading from Hugging Face: {e}")
        print("\n  Manual download:")
        print(f"      1. Go to: https://huggingface.co/datasets/{dataset_config['repo_id']}")
        print("      2. Download files")
        print(f"      3. Extract to: {target_dir}")
        return False

    return True


def validate_data_structure(data_dir: Path, dataset_name: str) -> bool:
    """Validate the downloaded data has expected structure.

    Args:
        data_dir: Base data directory
        dataset_name: Name of dataset to validate

    Returns:
        True if data structure is valid, False otherwise
    """
    print("\nValidating data structure...")

    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    target_dir = data_dir / DATASETS[dataset_name]["target_dir"]

    if not target_dir.exists():
        print(f"  [!] Dataset directory not found: {target_dir}")
        return False

    # Look for CU3S files (hyperspectral data)
    cu3s_files = list(target_dir.rglob("*.cu3s"))
    if not cu3s_files:
        print(f"  [!] No .cu3s files found in {target_dir}")
        return False

    print(f"  [OK] Found {len(cu3s_files)} CU3S files")

    # Look for annotations
    json_files = list(target_dir.rglob("*.json"))
    if json_files:
        print(f"  [OK] Found {len(json_files)} JSON files (annotations)")

    # Show sample structure
    print("\n  Sample files:")
    for f in cu3s_files[:3]:
        print(f"    - {f.relative_to(data_dir)}")

    return True


def update_documentation_paths(repo_root: Path) -> None:
    """Update documentation to reference correct paths.

    Args:
        repo_root: Repository root directory
    """
    # This could update documentation files to use Lentils instead of lentils
    # For now, just inform the user
    pass


def print_summary(data_dir: Path, dataset_name: str, status: dict) -> None:
    """Print summary of data setup.

    Args:
        data_dir: Base data directory
        dataset_name: Name of dataset
        status: Status dictionary from check_existing_data()
    """
    print("\n" + "=" * 70)
    print("DATA SETUP SUMMARY")
    print("=" * 70)

    print(f"\nData directory: {data_dir.absolute()}")
    print(f"Dataset: {dataset_name}")
    print("\nAvailable datasets:")
    for name, exists in status.items():
        symbol = "[OK]" if exists else "[ ]"
        print(f"  {symbol} {name}/")

    # Count total files
    target_dir = data_dir / DATASETS[dataset_name]["target_dir"]
    if target_dir.exists():
        cu3s_count = len(list(target_dir.rglob("*.cu3s")))
        print(f"\nTotal CU3S files: {cu3s_count}")

    print("\nValidation command:")
    print(
        f"  uv run python -c \"from pathlib import Path; print(Path('data/{DATASETS[dataset_name]['target_dir']}').exists())\""
    )

    print("\nRun documentation tests:")
    print("  uv run pytest tests/docs/ -v")


def main() -> None:
    """Main download and setup function."""
    parser = argparse.ArgumentParser(
        description="Download hyperspectral datasets from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets:
{chr(10).join(f"  {name}: {config['description']}" for name, config in DATASETS.items())}

Examples:
  # Download default dataset (lentils)
  uv run python scripts/download_data.py

  # Download specific dataset
  uv run python scripts/download_data.py --dataset lentils

  # Force re-download
  uv run python scripts/download_data.py --dataset lentils --force
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="lentils",
        choices=list(DATASETS.keys()),
        help="Dataset to download (default: lentils)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Data directory (default: <repo>/data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data exists",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of symlink for case-insensitive access",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, only setup links/structure",
    )

    args = parser.parse_args()

    dataset_config = DATASETS[args.dataset]
    target_dir_name = dataset_config["target_dir"]

    print("=" * 70)
    print("CUVIS-AI DATA DOWNLOADER")
    print("=" * 70)
    print(f"\nDataset: {args.dataset}")
    print(f"Description: {dataset_config['description']}")
    print(f"Repository: {dataset_config['repo_id']}")
    print(f"Size: {dataset_config['size']}\n")

    # Create data directory if needed
    args.data_dir.mkdir(parents=True, exist_ok=True)

    # Check existing data
    print("Checking existing data...")
    status = check_existing_data(args.data_dir, args.dataset)
    for name, exists in status.items():
        symbol = "[OK]" if exists else "[ ]"
        print(f"  {symbol} {name}/")

    # Download if needed
    if not args.skip_download:
        if not status[target_dir_name] or args.force:
            success = download_from_huggingface(args.data_dir, args.dataset, args.force)
            if not success and not args.force:
                print("\n[!] Download failed. Continuing with existing data...")
        else:
            print(f"\n  [OK] {args.dataset} data already exists")

    # Update status after download
    status = check_existing_data(args.data_dir, args.dataset)

    # Create symlink or copy for case-insensitive access (for lentils dataset)
    if args.dataset == "lentils" and status[target_dir_name]:
        lowercase_path = args.data_dir / "lentils"
        if not lowercase_path.exists():
            print("\nSetting up case-insensitive access...")
            if args.copy:
                copy_if_symlink_fails(args.data_dir)
            else:
                success = create_symlink_if_needed(args.data_dir)
                if not success and not args.copy:
                    print("\n  Tip: Use --copy flag to copy instead of symlink")

    # Validate
    if status[target_dir_name]:
        validate_data_structure(args.data_dir, args.dataset)

    # Update status one final time
    status = check_existing_data(args.data_dir, args.dataset)

    # Print summary
    print_summary(args.data_dir, args.dataset, status)

    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
