"""Tests for the dense patch-inference pair: PatchSampler and ClassMapAccumulator."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cuvis_ai.node.patch_inference import ClassMapAccumulator, PatchSampler

IGNORE = -100


class TestPatchSampler:
    def test_even_patch_size_rejected(self):
        with pytest.raises(ValueError, match="positive odd"):
            PatchSampler(patch_size=4)

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="mode must be"):
            PatchSampler(mode="bogus")

    def test_train_mode_shapes_and_skips_ignore(self):
        """Train mode draws samples_per_frame patches with the right shape and never the ignore label."""
        b, h, w, c, p = 2, 8, 8, 5, 3
        cube = torch.rand(b, h, w, c)
        targets = torch.zeros(b, h, w, dtype=torch.int64)
        targets[:, :4, :] = 1  # two classes present
        targets[:, 0, 0] = IGNORE
        out = PatchSampler(patch_size=p, samples_per_frame=16, mode="train").forward(
            cube=cube, targets=targets
        )
        assert out["patches"].shape[1:] == (p, p, c)
        assert out["patches"].dtype == torch.float32
        assert out["labels"].dtype == torch.int64
        assert out["patches"].shape[0] == out["labels"].shape[0]
        assert (out["labels"] != IGNORE).all()

    def test_eval_mode_round_trips_center_pixel(self):
        """patch_size=1 eval mode yields exactly the labeled pixels, in row-major order."""
        cube = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(1, 2, 2, 3)
        targets = torch.tensor([[[0, IGNORE], [2, 1]]], dtype=torch.int64)  # 3 labeled pixels
        out = PatchSampler(patch_size=1, mode="eval").forward(cube=cube, targets=targets)

        assert out["patches"].shape == (3, 1, 1, 3)
        # Labeled coords in row-major order: (0,0)=0, (1,0)=2, (1,1)=1.
        assert out["labels"].tolist() == [0, 2, 1]
        assert torch.equal(out["patches"][0, 0, 0], cube[0, 0, 0])
        assert torch.equal(out["patches"][1, 0, 0], cube[0, 1, 0])
        assert torch.equal(out["patches"][2, 0, 0], cube[0, 1, 1])

    def test_eval_mode_stride_cap(self):
        """max_per_frame strides the labeled pixels down to at most that many."""
        cube = torch.rand(1, 10, 10, 4)
        targets = torch.zeros(1, 10, 10, dtype=torch.int64)  # 100 labeled
        out = PatchSampler(patch_size=1, mode="eval", max_per_frame=10).forward(
            cube=cube, targets=targets
        )
        assert out["patches"].shape[0] <= 10

    def test_all_ignored_yields_empty(self):
        """A batch with no labeled pixel returns empty patches/labels of the right rank."""
        cube = torch.rand(1, 4, 4, 6)
        targets = torch.full((1, 4, 4), IGNORE, dtype=torch.int64)
        out = PatchSampler(patch_size=3, mode="eval").forward(cube=cube, targets=targets)
        assert out["patches"].shape == (0, 3, 3, 6)
        assert out["labels"].shape == (0,)


class TestClassMapAccumulator:
    @staticmethod
    def _logits(preds: list[int], num_classes: int) -> torch.Tensor:
        """One-hot logits whose argmax is each requested class."""
        return F.one_hot(torch.tensor(preds), num_classes=num_classes).to(torch.float32)

    def test_scatter_reconstructs_known_map(self):
        """Predictions land at their (y, x); untouched pixels stay at background_value."""
        node = ClassMapAccumulator(background_value=-1)
        node.reset()
        node.forward(
            logits=self._logits([2, 0], num_classes=3),
            frame_id=torch.tensor([0, 0]),
            y=torch.tensor([0, 1]),
            x=torch.tensor([0, 1]),
            height=torch.tensor([2, 2]),
            width=torch.tensor([2, 2]),
        )
        cmap = node.class_maps[0]
        expected = torch.tensor([[2, -1], [-1, 0]], dtype=torch.long)
        assert torch.equal(cmap, expected)

    def test_accumulates_across_batches(self):
        """Successive forward calls for the same frame keep filling the same map."""
        node = ClassMapAccumulator()
        node.reset()
        common = {"height": torch.tensor([2]), "width": torch.tensor([2])}
        node.forward(
            logits=self._logits([1], 3),
            frame_id=torch.tensor([0]),
            y=torch.tensor([0]),
            x=torch.tensor([0]),
            **common,
        )
        node.forward(
            logits=self._logits([2], 3),
            frame_id=torch.tensor([0]),
            y=torch.tensor([1]),
            x=torch.tensor([1]),
            **common,
        )
        assert node.class_maps[0].tolist() == [[1, -1], [-1, 2]]

    def test_reset_clears_state(self):
        node = ClassMapAccumulator()
        node.reset()
        node.forward(
            logits=self._logits([0], 2),
            frame_id=torch.tensor([0]),
            y=torch.tensor([0]),
            x=torch.tensor([0]),
            height=torch.tensor([1]),
            width=torch.tensor([1]),
        )
        assert node.class_maps  # non-empty
        node.reset()
        assert node.class_maps == {}

    def test_out_of_bounds_coords_raise(self):
        """A patch whose (y, x) exceeds the frame size is a wiring error, not a silent overwrite."""
        node = ClassMapAccumulator()
        node.reset()
        with pytest.raises(IndexError, match="out of bounds"):
            node.forward(
                logits=self._logits([0], 2),
                frame_id=torch.tensor([0]),
                y=torch.tensor([5]),
                x=torch.tensor([0]),
                height=torch.tensor([2]),
                width=torch.tensor([2]),
            )

    def test_class_maps_returns_copies(self):
        """class_maps hands back clones so callers can't mutate the accumulator's state."""
        node = ClassMapAccumulator()
        node.reset()
        node.forward(
            logits=self._logits([1], 2),
            frame_id=torch.tensor([0]),
            y=torch.tensor([0]),
            x=torch.tensor([0]),
            height=torch.tensor([1]),
            width=torch.tensor([1]),
        )
        node.class_maps[0][0, 0] = 99
        assert node.class_maps[0][0, 0].item() == 1
