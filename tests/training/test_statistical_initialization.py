"""Tests for statistical node initialization."""

import torch

from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline


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
