"""Tests for WelfordAccumulator against numpy reference implementations."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.utils.welford import WelfordAccumulator

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _random_data(n: int, c: int, *, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, c)).astype(np.float64)


# ------------------------------------------------------------------
# Mean / Variance (no covariance tracking)
# ------------------------------------------------------------------


class TestMeanVar:
    def test_mean_var_vs_numpy(self) -> None:
        data = _random_data(200, 10)
        acc = WelfordAccumulator(10)
        acc.update(torch.from_numpy(data))

        np.testing.assert_allclose(acc.mean.numpy(), data.mean(axis=0), atol=1e-5)
        np.testing.assert_allclose(acc.var.numpy(), data.var(axis=0, ddof=1), atol=1e-5)

    def test_std_vs_numpy(self) -> None:
        data = _random_data(200, 10)
        acc = WelfordAccumulator(10)
        acc.update(torch.from_numpy(data))

        np.testing.assert_allclose(acc.std.numpy(), data.std(axis=0, ddof=1), atol=1e-5)

    def test_multi_batch_streaming(self) -> None:
        """Feeding data in 5 chunks must give same results as all-at-once."""
        data = _random_data(500, 8)
        chunks = np.array_split(data, 5)

        acc_stream = WelfordAccumulator(8)
        for chunk in chunks:
            acc_stream.update(torch.from_numpy(chunk))

        acc_full = WelfordAccumulator(8)
        acc_full.update(torch.from_numpy(data))

        np.testing.assert_allclose(acc_stream.mean.numpy(), acc_full.mean.numpy(), atol=1e-10)
        np.testing.assert_allclose(acc_stream.var.numpy(), acc_full.var.numpy(), atol=1e-10)
        assert acc_stream.count == acc_full.count == 500

    def test_single_sample_at_a_time(self) -> None:
        """One-by-one updates must match batch update."""
        data = _random_data(50, 4)

        acc_single = WelfordAccumulator(4)
        for row in data:
            acc_single.update(torch.from_numpy(row.reshape(1, -1)))

        acc_batch = WelfordAccumulator(4)
        acc_batch.update(torch.from_numpy(data))

        np.testing.assert_allclose(acc_single.mean.numpy(), acc_batch.mean.numpy(), atol=1e-10)
        np.testing.assert_allclose(acc_single.var.numpy(), acc_batch.var.numpy(), atol=1e-10)


# ------------------------------------------------------------------
# Covariance / Correlation
# ------------------------------------------------------------------


class TestCovariance:
    def test_cov_vs_numpy(self) -> None:
        data = _random_data(300, 6)
        acc = WelfordAccumulator(6, track_covariance=True)
        acc.update(torch.from_numpy(data))

        np.testing.assert_allclose(acc.cov.numpy(), np.cov(data, rowvar=False), atol=1e-5)

    def test_corr_vs_numpy(self) -> None:
        data = _random_data(300, 6)
        acc = WelfordAccumulator(6, track_covariance=True)
        acc.update(torch.from_numpy(data))

        expected = np.abs(np.corrcoef(data, rowvar=False))
        np.testing.assert_allclose(acc.corr.numpy(), expected, atol=1e-5)

    def test_cov_multi_batch(self) -> None:
        """Covariance from streamed chunks matches all-at-once."""
        data = _random_data(400, 5)
        chunks = np.array_split(data, 4)

        acc_stream = WelfordAccumulator(5, track_covariance=True)
        for chunk in chunks:
            acc_stream.update(torch.from_numpy(chunk))

        acc_full = WelfordAccumulator(5, track_covariance=True)
        acc_full.update(torch.from_numpy(data))

        np.testing.assert_allclose(acc_stream.cov.numpy(), acc_full.cov.numpy(), atol=1e-10)

    def test_var_from_cov_diagonal(self) -> None:
        """When tracking covariance, .var should equal the diagonal of .cov."""
        data = _random_data(100, 7)
        acc = WelfordAccumulator(7, track_covariance=True)
        acc.update(torch.from_numpy(data))

        np.testing.assert_allclose(acc.var.numpy(), np.diag(acc.cov.numpy()), atol=1e-6)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    def test_single_sample_mean_ok(self) -> None:
        acc = WelfordAccumulator(3)
        acc.update(torch.tensor([[1.0, 2.0, 3.0]]))
        np.testing.assert_allclose(acc.mean.numpy(), [1.0, 2.0, 3.0])
        assert acc.count == 1

    def test_single_sample_var_raises(self) -> None:
        acc = WelfordAccumulator(3)
        acc.update(torch.tensor([[1.0, 2.0, 3.0]]))
        with pytest.raises(RuntimeError, match="at least 2 samples"):
            _ = acc.var

    def test_constant_data(self) -> None:
        data = torch.ones(100, 4)
        acc = WelfordAccumulator(4)
        acc.update(data)
        np.testing.assert_allclose(acc.var.numpy(), [0.0, 0.0, 0.0, 0.0], atol=1e-15)

    def test_single_feature(self) -> None:
        """C=1 works correctly."""
        data = _random_data(100, 1)
        acc = WelfordAccumulator(1)
        acc.update(torch.from_numpy(data))

        np.testing.assert_allclose(acc.mean.numpy(), data.mean(axis=0), atol=1e-5)
        np.testing.assert_allclose(acc.var.numpy(), data.var(axis=0, ddof=1), atol=1e-5)

    def test_1d_input_with_single_feature(self) -> None:
        """1-D tensor input is accepted when n_features=1."""
        data = torch.randn(50)
        acc = WelfordAccumulator(1)
        acc.update(data)
        assert acc.count == 50

    def test_empty_batch_ignored(self) -> None:
        acc = WelfordAccumulator(3)
        acc.update(torch.empty(0, 3))
        assert acc.count == 0


# ------------------------------------------------------------------
# Reset
# ------------------------------------------------------------------


class TestReset:
    def test_reset_clears_state(self) -> None:
        acc = WelfordAccumulator(5)
        acc.update(torch.randn(100, 5))
        assert acc.count == 100

        acc.reset()
        assert acc.count == 0
        with pytest.raises(RuntimeError):
            _ = acc.mean

    def test_reset_allows_fresh_accumulation(self) -> None:
        data = _random_data(100, 3)
        acc = WelfordAccumulator(3)
        acc.update(torch.randn(50, 3))  # junk data

        acc.reset()
        acc.update(torch.from_numpy(data))

        np.testing.assert_allclose(acc.mean.numpy(), data.mean(axis=0), atol=1e-5)


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------


class TestErrors:
    def test_mean_before_update_raises(self) -> None:
        acc = WelfordAccumulator(3)
        with pytest.raises(RuntimeError, match="No samples"):
            _ = acc.mean

    def test_var_before_update_raises(self) -> None:
        acc = WelfordAccumulator(3)
        with pytest.raises(RuntimeError, match="at least 2 samples"):
            _ = acc.var

    def test_cov_without_tracking_raises(self) -> None:
        acc = WelfordAccumulator(3)
        acc.update(torch.randn(10, 3))
        with pytest.raises(RuntimeError, match="Covariance tracking was not enabled"):
            _ = acc.cov

    def test_corr_without_tracking_raises(self) -> None:
        acc = WelfordAccumulator(3)
        acc.update(torch.randn(10, 3))
        with pytest.raises(RuntimeError, match="Covariance tracking was not enabled"):
            _ = acc.corr


# ------------------------------------------------------------------
# nn.Module behaviour
# ------------------------------------------------------------------


class TestModuleBehaviour:
    def test_persistent_false(self) -> None:
        """Accumulator buffers must NOT appear in state_dict."""
        acc = WelfordAccumulator(4, track_covariance=True)
        acc.update(torch.randn(10, 4))
        assert len(acc.state_dict()) == 0

    def test_device_propagation(self) -> None:
        """Calling .to() on the accumulator moves its buffers."""
        acc = WelfordAccumulator(3)
        acc.to("cpu")  # should not raise
        assert acc._mean.device == torch.device("cpu")

    def test_submodule_in_parent(self) -> None:
        """When registered as a submodule, .to() on the parent propagates."""

        class Parent(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.welford = WelfordAccumulator(3)

        parent = Parent()
        parent.to("cpu")
        assert parent.welford._mean.device == torch.device("cpu")

    def test_state_dict_excludes_accumulator_in_parent(self) -> None:
        """Accumulator buffers must not leak into the parent's state_dict."""

        class Parent(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("mu", torch.zeros(3))
                self.welford = WelfordAccumulator(3)

        parent = Parent()
        parent.welford.update(torch.randn(10, 3))
        sd = parent.state_dict()
        assert "mu" in sd
        assert all("welford" not in k for k in sd)
