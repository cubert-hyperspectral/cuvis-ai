"""SAM3 video object tracking example.

Demonstrates HSI video tracking using SAM3 with false RGB projection.

Usage:
    uv run python examples/sam3/sam3_tracking_example.py --video path/to/video.cu3s
"""


def main() -> None:
    """Run the SAM3 tracking example workflow scaffold."""
    # 1. Load plugin manifest.
    # 2. Build pipeline from configs/pipeline/sam3/sam3_naive_false_rgb.yaml.
    # 3. Run tracking on input video.
    # 4. Export results as COCO JSON.
    raise NotImplementedError("Requires T1.2 (SAM3VideoTracker node) and T1.3 (pipeline).")


if __name__ == "__main__":
    main()
