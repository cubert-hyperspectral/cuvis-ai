"""Unit tests for pipeline_visualization nodes."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

from cuvis_ai.node.pipeline_visualization import (
    CubeRGBVisualizer,
    PCAVisualization,
    PipelineComparisonVisualizer,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# CubeRGBVisualizer
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_cube_rgb_visualizer_returns_artifacts() -> None:
    """CubeRGBVisualizer should produce Artifact objects for each batch element."""
    B, H, W, C = 2, 8, 8, 10
    cube = torch.rand(B, H, W, C)
    weights = torch.rand(C)
    wavelengths = np.linspace(450, 900, C).astype(np.int32)
    ctx = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

    node = CubeRGBVisualizer(up_to=5)
    result = node.forward(cube=cube, weights=weights, wavelengths=wavelengths, context=ctx)

    artifacts = result["artifacts"]
    assert len(artifacts) == B
    for art in artifacts:
        assert art.value is not None
        assert art.value.ndim == 3  # H, W, C (RGB image)


@torch.no_grad()
def test_cube_rgb_visualizer_respects_up_to() -> None:
    """up_to should cap the number of artifacts produced."""
    B, H, W, C = 5, 4, 4, 8
    cube = torch.rand(B, H, W, C)
    weights = torch.rand(C)
    wavelengths = np.linspace(450, 900, C).astype(np.int32)
    ctx = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

    node = CubeRGBVisualizer(up_to=2)
    result = node.forward(cube=cube, weights=weights, wavelengths=wavelengths, context=ctx)
    assert len(result["artifacts"]) == 2


# ---------------------------------------------------------------------------
# PCAVisualization
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_pca_visualization_returns_artifacts() -> None:
    """PCAVisualization should produce Artifact objects for each batch element."""
    B, H, W, C = 2, 6, 6, 4
    data = torch.randn(B, H, W, C)
    ctx = Context(stage=ExecutionStage.VAL, epoch=1, batch_idx=0)

    node = PCAVisualization(up_to=5)
    result = node.forward(data=data, context=ctx)

    artifacts = result["artifacts"]
    assert len(artifacts) == B
    for art in artifacts:
        assert art.value is not None
        assert art.value.ndim == 3


@torch.no_grad()
def test_pca_visualization_respects_up_to() -> None:
    """up_to should cap the number of artifacts produced."""
    B, H, W, C = 4, 4, 4, 3
    data = torch.randn(B, H, W, C)
    ctx = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

    node = PCAVisualization(up_to=2)
    result = node.forward(data=data, context=ctx)
    assert len(result["artifacts"]) == 2


@torch.no_grad()
def test_pca_visualization_up_to_none() -> None:
    """When up_to is None, all batch elements should be visualized."""
    B, H, W, C = 3, 4, 4, 3
    data = torch.randn(B, H, W, C)
    ctx = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

    node = PCAVisualization(up_to=None)
    result = node.forward(data=data, context=ctx)
    assert len(result["artifacts"]) == B


@torch.no_grad()
def test_pca_visualization_rejects_3d_input() -> None:
    """Non-4D input should raise ValueError."""
    data = torch.randn(6, 6, 4)  # 3D
    ctx = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

    node = PCAVisualization()
    with pytest.raises(ValueError, match="Expected 4D input"):
        node.forward(data=data, context=ctx)


@torch.no_grad()
def test_pca_visualization_rejects_single_component() -> None:
    """Input with < 2 components should raise ValueError."""
    data = torch.randn(2, 4, 4, 1)
    ctx = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

    node = PCAVisualization()
    with pytest.raises(ValueError, match="at least 2 components"):
        node.forward(data=data, context=ctx)


# ---------------------------------------------------------------------------
# PipelineComparisonVisualizer
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_pipeline_comparison_returns_artifacts() -> None:
    """PipelineComparisonVisualizer should produce 4 artifacts per sample."""
    B, H, W, C = 2, 8, 8, 10
    hsi_cube = torch.rand(B, H, W, C)
    mixer_output = torch.rand(B, H, W, 3)
    gt_mask = torch.randint(0, 2, (B, H, W, 1), dtype=torch.bool)
    scores = torch.rand(B, H, W, 1)
    ctx = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

    node = PipelineComparisonVisualizer(hsi_channels=[0, 3, 6], max_samples=4)
    result = node.forward(
        hsi_cube=hsi_cube,
        mixer_output=mixer_output,
        ground_truth_mask=gt_mask,
        adaclip_scores=scores,
        context=ctx,
    )

    artifacts = result["artifacts"]
    # 4 artifacts per sample (hsi, mixer, mask, scores) * min(B, max_samples)
    assert len(artifacts) == B * 4


@torch.no_grad()
def test_pipeline_comparison_max_samples() -> None:
    """max_samples should cap artifacts."""
    B, H, W, C = 5, 4, 4, 8
    hsi_cube = torch.rand(B, H, W, C)
    mixer_output = torch.rand(B, H, W, 3)
    gt_mask = torch.randint(0, 2, (B, H, W, 1), dtype=torch.bool)
    scores = torch.rand(B, H, W, 1)
    ctx = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

    node = PipelineComparisonVisualizer(max_samples=2)
    result = node.forward(
        hsi_cube=hsi_cube,
        mixer_output=mixer_output,
        ground_truth_mask=gt_mask,
        adaclip_scores=scores,
        context=ctx,
    )
    assert len(result["artifacts"]) == 2 * 4


@torch.no_grad()
def test_pipeline_comparison_log_every_n_batches() -> None:
    """log_every_n_batches should skip intermediate batches."""
    B, H, W, C = 1, 4, 4, 8
    hsi_cube = torch.rand(B, H, W, C)
    mixer_output = torch.rand(B, H, W, 3)
    gt_mask = torch.zeros(B, H, W, 1, dtype=torch.bool)
    scores = torch.rand(B, H, W, 1)
    ctx = Context(stage=ExecutionStage.TRAIN, epoch=0, batch_idx=0)

    node = PipelineComparisonVisualizer(log_every_n_batches=3, max_samples=1)

    # First call (batch_counter=1): should log (counter-1=0, 0%3==0)
    r1 = node.forward(
        hsi_cube=hsi_cube,
        mixer_output=mixer_output,
        ground_truth_mask=gt_mask,
        adaclip_scores=scores,
        context=ctx,
    )
    assert len(r1["artifacts"]) == 4

    # Second call (batch_counter=2): should skip (counter-1=1, 1%3!=0)
    r2 = node.forward(
        hsi_cube=hsi_cube,
        mixer_output=mixer_output,
        ground_truth_mask=gt_mask,
        adaclip_scores=scores,
        context=ctx,
    )
    assert len(r2["artifacts"]) == 0

    # Third call (batch_counter=3): should skip
    r3 = node.forward(
        hsi_cube=hsi_cube,
        mixer_output=mixer_output,
        ground_truth_mask=gt_mask,
        adaclip_scores=scores,
        context=ctx,
    )
    assert len(r3["artifacts"]) == 0

    # Fourth call (batch_counter=4): should log (counter-1=3, 3%3==0)
    r4 = node.forward(
        hsi_cube=hsi_cube,
        mixer_output=mixer_output,
        ground_truth_mask=gt_mask,
        adaclip_scores=scores,
        context=ctx,
    )
    assert len(r4["artifacts"]) == 4


@torch.no_grad()
def test_pipeline_comparison_default_context() -> None:
    """When context is None, should use default Context."""
    B, H, W, C = 1, 4, 4, 6
    node = PipelineComparisonVisualizer(max_samples=1)
    result = node.forward(
        hsi_cube=torch.rand(B, H, W, C),
        mixer_output=torch.rand(B, H, W, 3),
        ground_truth_mask=torch.zeros(B, H, W, 1, dtype=torch.bool),
        adaclip_scores=torch.rand(B, H, W, 1),
        context=None,
    )
    assert len(result["artifacts"]) == 4


@torch.no_grad()
def test_pipeline_comparison_hsi_channel_clamping() -> None:
    """HSI channels beyond cube range should be clamped to valid indices."""
    B, H, W, C = 1, 4, 4, 5
    node = PipelineComparisonVisualizer(hsi_channels=[0, 100, 200], max_samples=1)
    result = node.forward(
        hsi_cube=torch.rand(B, H, W, C),
        mixer_output=torch.rand(B, H, W, 3),
        ground_truth_mask=torch.zeros(B, H, W, 1, dtype=torch.bool),
        adaclip_scores=torch.rand(B, H, W, 1),
    )
    assert len(result["artifacts"]) == 4


@torch.no_grad()
def test_pipeline_comparison_normalize_image_constant() -> None:
    """Constant-value image should produce zeros after normalization."""
    node = PipelineComparisonVisualizer()
    constant = np.ones((4, 4, 3), dtype=np.float32) * 5.0
    result = node._normalize_image(constant)
    assert np.allclose(result, 0.0)


@torch.no_grad()
def test_pipeline_comparison_mask_visualization() -> None:
    """Mask visualization should produce [H, W, 3] with red channel for anomalies."""
    node = PipelineComparisonVisualizer()
    mask = np.zeros((4, 4, 1), dtype=np.float32)
    mask[1, 1, 0] = 1.0
    rgb = node._create_mask_visualization(mask)
    assert rgb.shape == (4, 4, 3)
    assert rgb[1, 1, 0] == 1.0  # red channel
    assert rgb[0, 0, 0] == 0.0  # no anomaly


@torch.no_grad()
def test_pipeline_comparison_scores_heatmap() -> None:
    """Scores heatmap should produce [H, W, 3] with R=high, B=low pattern."""
    node = PipelineComparisonVisualizer()
    scores = np.zeros((4, 4, 1), dtype=np.float32)
    scores[0, 0, 0] = 1.0  # high score pixel
    rgb = node._create_scores_heatmap(scores)
    assert rgb.shape == (4, 4, 3)
    # High-score pixel should have high red, low blue
    assert rgb[0, 0, 0] > rgb[0, 0, 2]
