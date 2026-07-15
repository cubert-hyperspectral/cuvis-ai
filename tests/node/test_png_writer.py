from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torchvision.io import read_image

from cuvis_ai.node.image_file import PngWriter

pytestmark = pytest.mark.unit


def _expected_u8(frame: torch.Tensor) -> torch.Tensor:
    """[H, W, 3] float in [0, 1] -> [3, H, W] uint8, matching the writer."""
    return (frame.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).permute(2, 0, 1)


@torch.no_grad()
def test_single_frame_roundtrip(tmp_path: Path) -> None:
    img = torch.rand(1, 5, 7, 3)
    out = tmp_path / "strip.png"
    result = PngWriter(output_path=str(out)).forward(rgb_image=img)
    assert result == {}  # sink
    assert out.exists()
    loaded = read_image(str(out))
    assert loaded.shape == (3, 5, 7)
    assert torch.equal(loaded, _expected_u8(img[0]))


@torch.no_grad()
def test_multi_frame_batch_names_per_index(tmp_path: Path) -> None:
    img = torch.rand(3, 4, 4, 3)
    out = tmp_path / "frame.png"
    PngWriter(output_path=str(out)).forward(rgb_image=img)
    for i in range(3):
        p = tmp_path / f"frame_{i:06d}.png"
        assert p.exists()
        assert torch.equal(read_image(str(p)), _expected_u8(img[i]))
    assert not out.exists()  # no bare path for a multi-frame batch


@torch.no_grad()
def test_frame_id_names_single_frame(tmp_path: Path) -> None:
    img = torch.rand(1, 4, 4, 3)
    out = tmp_path / "frame.png"
    PngWriter(output_path=str(out)).forward(
        rgb_image=img, frame_id=torch.tensor([42], dtype=torch.int64)
    )
    assert (tmp_path / "frame_000042.png").exists()


@torch.no_grad()
def test_parent_dir_created_on_construction(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "out.png"
    PngWriter(output_path=str(nested))
    assert nested.parent.is_dir()


def test_bad_compression_level(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="compression_level"):
        PngWriter(output_path=str(tmp_path / "x.png"), compression_level=99)
