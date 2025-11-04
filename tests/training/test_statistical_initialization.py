"""Tests for statistical node initialization."""

import pytest
import torch

from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.normalization.normalization import MinMaxNormalizer
from cuvis_ai.pipeline.graph import Graph


def test_rxglobal_requires_initial_fit():
    """Test that RXGlobal requires initial fit."""
    rx = RXGlobal()
    assert rx.requires_initial_fit is True


def test_minmax_normalizer_requires_initial_fit():
    """Test that MinMaxNormalizer requires initial fit when using running stats."""
    normalizer = MinMaxNormalizer(use_running_stats=True)
    assert normalizer.requires_initial_fit is True
    
    normalizer_no_stats = MinMaxNormalizer(use_running_stats=False)
    assert normalizer_no_stats.requires_initial_fit is False


def test_rxglobal_initialize_from_data():
    """Test RXGlobal statistical initialization from data."""
    rx = RXGlobal(eps=1e-6)
    
    # Create mock data iterator
    def data_iterator():
        for _ in range(2):
            x = torch.randn(2, 10, 10, 5)  # B,H,W,C
            yield (x, None, {})
    
    # Initialize
    rx.initialize_from_data(data_iterator())
    
    # Check mu and cov were created
    assert rx.mu is not None
    assert rx.cov is not None
    assert rx.mu.shape == torch.Size([5])  # C channels
    assert rx.cov.shape == torch.Size([5, 5])  # CxC


def test_minmax_normalizer_initialize_from_data():
    """Test MinMaxNormalizer statistical initialization from data."""
    normalizer = MinMaxNormalizer(use_running_stats=True)
    
    # Create mock data iterator
    def data_iterator():
        for _ in range(2):
            x = torch.randn(2, 10, 10, 1) + 5.0  # Shift to positive
            yield (x, None, {})
    
    # Initialize
    normalizer.initialize_from_data(data_iterator())
    
    # Check running stats were created
    assert normalizer.running_min is not None
    assert normalizer.running_max is not None


def test_rxglobal_trainable_stats():
    """Test RXGlobal trainable_stats property."""
    rx_frozen = RXGlobal(trainable_stats=False)
    assert rx_frozen.is_trainable is False
    
    rx_trainable = RXGlobal(trainable_stats=True)
    assert rx_trainable.is_trainable is True


def test_graph_identifies_statistical_nodes():
    """Test that graph identifies nodes requiring initialization."""
    graph = Graph("test_graph")
    rx = RXGlobal()
    normalizer = MinMaxNormalizer(use_running_stats=True)
    
    graph.add_node(rx)
    graph.add_node(normalizer, parent=rx)
    
    # Find nodes requiring initialization
    stat_nodes = [
        (node_id, node) for node_id, node in graph.nodes.items()
        if node.requires_initial_fit
    ]
    
    assert len(stat_nodes) == 2
