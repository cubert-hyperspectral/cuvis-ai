from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.spectrum_plot import SpectrumPlotNode


def test_spectrum_plot_renders_fixed_size_rgb_frames() -> None:
    node = SpectrumPlotNode(
        wavelengths=[500.0, 600.0, 700.0],
        reference_wavelengths=[500.0, 700.0],
        plot_width=64,
        plot_height=48,
        dpi=32,
        tracked_label="tracked",
        reference_label="reference",
        y_fixed_range=None,
        tracked_hold_frames=0,
    )
    tracked = torch.tensor([[0.1, 0.5, 0.9], [0.2, 0.4, 0.6]], dtype=torch.float32)
    reference = torch.tensor([[[[0.25, 0.75]]]], dtype=torch.float32)
    valid = torch.tensor([1, 0], dtype=torch.int32)
    frame_id = torch.tensor([10, 11], dtype=torch.int64)

    out = node.forward(
        tracked_spectrum=tracked,
        reference_spectrum=reference,
        valid=valid,
        frame_id=frame_id,
    )["rgb_image"]

    assert out.shape == (2, 48, 64, 3)
    assert out.dtype == torch.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_spectrum_plot_holds_last_tracked_spectrum(monkeypatch: pytest.MonkeyPatch) -> None:
    node = SpectrumPlotNode(
        wavelengths=[1.0, 2.0, 3.0],
        reference_wavelengths=[1.0, 3.0],
        plot_width=32,
        plot_height=32,
        tracked_hold_frames=1,
    )
    calls: list[tuple[np.ndarray, bool, int | None]] = []

    def fake_render(
        tracked: np.ndarray,
        reference: np.ndarray,  # noqa: ARG001
        tracked_valid: bool,
        frame_id: int | None,
    ) -> np.ndarray:
        calls.append((tracked.copy(), tracked_valid, frame_id))
        return np.zeros((node.plot_height, node.plot_width, 3), dtype=np.uint8)

    monkeypatch.setattr(node, "_render_frame", fake_render)

    tracked = torch.tensor(
        [[1.0, 2.0, 3.0], [9.0, 9.0, 9.0], [5.0, 5.0, 5.0]],
        dtype=torch.float32,
    )
    reference = torch.tensor([[[[0.5, 0.6]]]], dtype=torch.float32)
    valid = torch.tensor([1, 0, 0], dtype=torch.int32)
    frame_id = torch.tensor([1, 2, 3], dtype=torch.int64)

    node.forward(
        tracked_spectrum=tracked,
        reference_spectrum=reference,
        valid=valid,
        frame_id=frame_id,
    )

    assert [call[1] for call in calls] == [True, True, False]
    np.testing.assert_allclose(calls[1][0], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(calls[2][0], np.array([5.0, 5.0, 5.0], dtype=np.float32))
    assert [call[2] for call in calls] == [1, 2, 3]


def test_spectrum_plot_valid_defaults_to_showing_tracked(monkeypatch: pytest.MonkeyPatch) -> None:
    node = SpectrumPlotNode(
        wavelengths=[1.0, 2.0],
        reference_wavelengths=[1.0, 2.0],
        plot_width=32,
        plot_height=32,
    )
    tracked_flags: list[bool] = []

    def fake_render(
        tracked: np.ndarray,  # noqa: ARG001
        reference: np.ndarray,  # noqa: ARG001
        tracked_valid: bool,
        frame_id: int | None,  # noqa: ARG001
    ) -> np.ndarray:
        tracked_flags.append(tracked_valid)
        return np.zeros((node.plot_height, node.plot_width, 3), dtype=np.uint8)

    monkeypatch.setattr(node, "_render_frame", fake_render)

    node.forward(
        tracked_spectrum=torch.ones((2, 2), dtype=torch.float32),
        reference_spectrum=torch.ones((1, 1, 1, 2), dtype=torch.float32),
    )

    assert tracked_flags == [True, True]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"plot_width": 31}, "plot dimensions"),
        ({"plot_height": 31}, "plot dimensions"),
        ({"dpi": 0}, "dpi"),
        ({"y_num_ticks": 1}, "y_num_ticks"),
        ({"tracked_hold_frames": -1}, "tracked_hold_frames"),
        ({"wavelengths": []}, "wavelengths"),
        ({"reference_wavelengths": []}, "reference_wavelengths"),
    ],
)
def test_spectrum_plot_validates_constructor_inputs(
    kwargs: dict[str, object],
    match: str,
) -> None:
    base = {
        "wavelengths": [1.0, 2.0],
        "reference_wavelengths": [1.0, 2.0],
        "plot_width": 32,
        "plot_height": 32,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        SpectrumPlotNode(**base)
