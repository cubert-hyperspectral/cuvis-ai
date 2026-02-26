"""Tests for statistical node initialization."""

import pytest
import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

from cuvis_ai.anomaly.deep_svdd import DeepSVDDCenterTracker, ZScoreNormalizerGlobal
from cuvis_ai.anomaly.lad_detector import LADGlobal
from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.node.channel_mixer import LearnableChannelMixer
from cuvis_ai.node.channel_selector import SoftChannelSelector
from cuvis_ai.node.conversion import ScoreToLogit
from cuvis_ai.node.dimensionality_reduction import TrainablePCA
from cuvis_ai.node.normalization import MinMaxNormalizer

pytestmark = pytest.mark.unit

EMPTY_INIT_ERRORS = (RuntimeError, ValueError, StopIteration)


def test_rxglobal_requires_initial_fit():
    """Test that RXGlobal requires initial fit."""
    rx = RXGlobal(num_channels=61)
    assert rx.requires_initial_fit is True


def test_minmax_normalizer_requires_initial_fit():
    """Test that MinMaxNormalizer requires initial fit when using running stats."""
    normalizer = MinMaxNormalizer(use_running_stats=True)
    assert normalizer.requires_initial_fit is True

    normalizer_no_stats = MinMaxNormalizer(use_running_stats=False)
    assert normalizer_no_stats.requires_initial_fit is False


def test_rxglobal_fit():
    """Test RXGlobal statistical initialization from data."""
    rx = RXGlobal(num_channels=5, eps=1e-6)

    # Create mock data iterator - fit() expects dicts with port names as keys
    def data_iterator():
        for _ in range(2):
            x = torch.randn(2, 10, 10, 5)  # B,H,W,C
            yield {"data": x}

    # Initialize
    rx.statistical_initialization(data_iterator())

    # Check mu and cov were created
    assert rx.mu is not None
    assert rx.cov is not None
    assert rx.mu.shape == torch.Size([5])  # C channels
    assert rx.cov.shape == torch.Size([5, 5])  # CxC
    assert rx._statistically_initialized is True


def test_minmax_normalizer_fit():
    """Test MinMaxNormalizer statistical initialization from data."""
    normalizer = MinMaxNormalizer(use_running_stats=True)

    # Create mock data iterator - fit() expects dicts with port names as keys
    def data_iterator():
        for _ in range(2):
            x = torch.randn(2, 10, 10, 1) + 5.0  # Shift to positive
            yield {"data": x}

    # Initialize
    normalizer.statistical_initialization(data_iterator())

    # Check running stats were created
    assert normalizer.running_min is not None
    assert normalizer.running_max is not None


def test_graph_identifies_statistical_nodes():
    """Test that graph identifies nodes requiring initialization."""
    pipeline = CuvisPipeline("test_graph")
    rx = RXGlobal(num_channels=61)
    normalizer = MinMaxNormalizer(use_running_stats=True)

    # Use port namespace access
    pipeline.connect(rx.scores, normalizer.data)

    # Find nodes requiring initialization
    stat_nodes = [node for node in pipeline.nodes() if node.requires_initial_fit]

    assert len(stat_nodes) == 2


def test_rxglobal_empty_initialization_raises_and_blocks_forward():
    rx = RXGlobal(num_channels=5, eps=1e-6)

    with pytest.raises(RuntimeError, match="insufficient samples"):
        rx.statistical_initialization(iter(()))

    assert rx._statistically_initialized is False
    with pytest.raises(RuntimeError, match="not initialized"):
        rx.forward(torch.randn(1, 4, 4, 5))


def test_rxglobal_single_pixel_initialization_raises():
    rx = RXGlobal(num_channels=5, eps=1e-6)

    def data_iterator():
        yield {"data": torch.randn(1, 1, 1, 5)}

    with pytest.raises(RuntimeError, match="insufficient samples"):
        rx.statistical_initialization(data_iterator())

    assert rx._statistically_initialized is False


def test_minmax_running_stats_empty_initialization_raises_and_blocks_forward():
    node = MinMaxNormalizer(use_running_stats=True)

    with pytest.raises(RuntimeError, match="did not receive any data"):
        node.statistical_initialization(iter(()))

    assert node._statistically_initialized is False
    with pytest.raises(RuntimeError, match="requires statistical_initialization"):
        node.forward(torch.randn(1, 4, 4, 3))


def test_score_to_logit_empty_initialization_raises_and_blocks_forward():
    node = ScoreToLogit()

    with pytest.raises(RuntimeError, match="insufficient samples"):
        node.statistical_initialization(iter(()))

    assert node._statistically_initialized is False
    with pytest.raises(RuntimeError, match="not initialized"):
        node.forward(torch.randn(1, 4, 4, 1))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RXGlobal(num_channels=5),
        lambda: LADGlobal(num_channels=5),
        lambda: ZScoreNormalizerGlobal(num_channels=5),
        lambda: TrainablePCA(num_channels=5, n_components=3),
    ],
)
def test_critical_statistical_nodes_reject_empty_initialization(factory):
    node = factory()
    assert node.requires_initial_fit is True

    with pytest.raises(EMPTY_INIT_ERRORS):
        node.statistical_initialization(iter(()))

    assert node._statistically_initialized is False


@pytest.mark.parametrize(
    ("node_name", "factory"),
    [
        ("RXGlobal", lambda: RXGlobal(num_channels=5)),
        ("LADGlobal", lambda: LADGlobal(num_channels=5)),
        ("ZScoreNormalizerGlobal", lambda: ZScoreNormalizerGlobal(num_channels=5)),
        ("TrainablePCA", lambda: TrainablePCA(num_channels=5, n_components=3)),
        ("DeepSVDDCenterTracker", lambda: DeepSVDDCenterTracker(rep_dim=4)),
        (
            "LearnableChannelMixer[pca]",
            lambda: LearnableChannelMixer(input_channels=5, output_channels=3, init_method="pca"),
        ),
        (
            "SoftChannelSelector[variance]",
            lambda: SoftChannelSelector(n_select=3, input_channels=5, init_method="variance"),
        ),
        ("MinMaxNormalizer[running]", lambda: MinMaxNormalizer(use_running_stats=True)),
        ("ScoreToLogit", lambda: ScoreToLogit()),
    ],
)
def test_constructable_requires_initial_fit_nodes_reject_empty_stream(node_name, factory):
    node = factory()
    assert node.requires_initial_fit is True, f"{node_name} must require initial fit in this sweep."

    with pytest.raises(EMPTY_INIT_ERRORS):
        node.statistical_initialization(iter(()))

    assert node._statistically_initialized is False
