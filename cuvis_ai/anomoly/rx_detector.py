import torch
import torch.nn as nn

from cuvis_ai.normalization.normalization import resolve_score_normalizer


class RXDetectorTorch(nn.Module):
    """
    Reed-Xiaoli (RX) anomaly detector in pure PyTorch.

    Modes
    -----
    1) Accumulate global μ, Σ over a training dataset via `update()` (streaming)
       and `finalize()`. Then call `score(..., per_batch=False)` to apply those
       fixed stats on new/test data.

    2) Per-batch statistics on-the-fly via `score(..., per_batch=True)`, which
       computes μ and Σ from each *batch element* independently.

    Input / Output
    --------------
    - Input:  B × H × W × C (channels-last)
    - Output: B × H × W   (squared Mahalanobis distance per pixel)

    Notes
    -----
    - Uses Moore–Penrose pseudoinverse by default. You can add diagonal jitter
      via `eps` to stabilize Σ before inversion.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        score_normalization: str = "minmax",
    ):
        super().__init__()

        self.register_buffer("mu", None)  # (C,)
        self.register_buffer("cov", None)  # (C, C)
        self.register_buffer("cov_inv", None)  # (C, C)
        self.register_buffer("_mean", None)  # (C,)
        self.register_buffer("_M2", None)  # (C, C)

        self._fitted: bool = False

        # Online (Welford/parallel) accumulators for global stats
        self._n: int = 0  # total sample count

        self.eps = eps

        # Validate score normalization method
        valid_methods = ["identity", "minmax", "sigmoid"]
        if score_normalization not in valid_methods:
            raise ValueError(
                f"score_normalization must be one of {valid_methods}, got '{score_normalization}'"
            )
        self.score_normalization = score_normalization
        self._normalizer = resolve_score_normalizer(score_normalization)

    @staticmethod
    def _flatten_bhwc(x: torch.Tensor):
        # x: B,H,W,C  ->  (B, N, C), with N = H*W
        B, H, W, C = x.shape
        return x.view(B, H * W, C)

    # ---------------------------
    # Accumulation (streaming)
    # ---------------------------
    @torch.no_grad()
    def update(self, batch_bhwc: torch.Tensor):
        """
        Incorporate a training batch into global statistics (μ, Σ) using
        parallel-Welford accumulation across *all pixels of all images*.

        Call repeatedly across the training set, then call `finalize()`.
        """
        x = batch_bhwc
        assert x.dim() == 4, "Expected B×H×W×C tensor"
        B, H, W, C = x.shape
        X = self._flatten_bhwc(x).reshape(-1, C)  # (M, C), M=B*H*W

        # Batch stats
        m = X.shape[0]
        if m <= 1:
            return

        batch_mean = X.mean(dim=0)  # (C,)
        centered = X - batch_mean  # (M, C)
        # Sum of outer products: M2_b = centered^T @ centered
        M2_b = centered.transpose(0, 1) @ centered  # (C, C)

        # Merge with running stats
        if self._n == 0:
            self._n = m
            self._mean = batch_mean
            self._M2 = M2_b
        else:
            n = self._n
            delta = batch_mean - self._mean  # (C,)
            tot = n + m
            # Update mean
            new_mean = self._mean + delta * (m / tot)  # (C,)
            # Between-mean covariance contribution
            # n*m/(n+m) * outer(delta, delta)
            outer = torch.outer(delta, delta) * (n * m / tot)  # (C, C)
            # Update M2
            new_M2 = self._M2 + M2_b + outer

            self._n = tot
            self._mean = new_mean
            self._M2 = new_M2

        self._fitted = False  # since accumulators changed

    @torch.no_grad()
    def finalize(self):
        """
        Convert accumulated (_n, _mean, _M2) into μ, Σ, Σ^{-1}.
        After calling this, the detector is ready for `score(..., per_batch=False)`.
        """
        if self._n <= 1:
            raise ValueError("Not enough samples to finalize (need at least 2).")

        self.mu = self._mean.clone()
        # Unbiased covariance: M2 / (n - 1)
        cov = self._M2 / (self._n - 1)

        # Optional diagonal jitter
        if self.eps > 0:
            cov = cov + self.eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)

        self.cov = cov
        self.cov_inv = torch.linalg.pinv(cov)  # (C, C)
        self._fitted = True
        return self

    def reset(self):
        """Clear all accumulated/global stats."""
        self.mu = None
        self.cov = None
        self.cov_inv = None
        self._fitted = False
        self._n = 0
        self._mean = None
        self._M2 = None

    # ---------------------------
    # Score Normalization
    # ---------------------------
    @torch.no_grad()
    def run_postprocessing(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply score normalization per batch element.

        Parameters
        ----------
        scores : torch.Tensor
            Raw anomaly scores of shape (B, H, W)

        Returns
        -------
        normalized_scores : torch.Tensor
            Normalized scores of shape (B, H, W)
        """
        norm_module = self._normalizer.to(scores.device, scores.dtype)
        normalized, _, _ = norm_module(scores.unsqueeze(-1))
        return normalized.squeeze(-1)

    # ---------------------------
    # Batch-wise fit (single tensor)
    # ---------------------------
    @torch.no_grad()
    def fit(self, data_bhwc: torch.Tensor):
        """
        Compute global μ and Σ from a single tensor (no streaming).
        Equivalent to calling `update(data)` once then `finalize()`.
        """
        self.reset()
        self.update(data_bhwc)
        self.finalize()
        return self

    # ---------------------------
    # Scoring
    # ---------------------------
    @torch.no_grad()
    def score(self, data_bhwc: torch.Tensor, per_batch: bool = False) -> torch.Tensor:
        """
        Compute squared Mahalanobis distance per pixel.

        - per_batch = False (default): use finalized global μ and Σ.
          Requires `finalize()` or `fit()` beforehand.

        - per_batch = True: ignore stored stats and compute μ/Σ per batch
          element from the *current* input only (each image uses its own stats).

        Returns
        -------
        scores : B × H × W
        """
        x = data_bhwc
        assert x.dim() == 4, "Expected B×H×W×C"
        B, H, W, C = x.shape
        N = H * W

        if per_batch:
            # Compute μ/Σ per image in the batch
            X = self._flatten_bhwc(x)  # (B, N, C)
            mu_b = X.mean(dim=1, keepdim=True)  # (B, 1, C)
            Xc = X - mu_b  # (B, N, C)
            # Cov per-batch: (B, C, C) = Xc^T @ Xc / (N-1)
            cov_b = torch.matmul(Xc.transpose(1, 2), Xc) / max(N - 1, 1)  # (B, C, C)
            if self.eps > 0:
                eye = torch.eye(C, device=x.device, dtype=x.dtype).expand(B, C, C)
                cov_b = cov_b + self.eps * eye
            cov_inv_b = torch.linalg.pinv(cov_b)  # (B, C, C)
            # Mahalanobis per pixel: (B,N)
            md2 = torch.einsum("bnc,bcd,bnd->bn", Xc, cov_inv_b, Xc)
            scores = md2.view(B, H, W)
            return self.run_postprocessing(scores)

        # Fixed (global) stats
        if not self._fitted or self.mu is None or self.cov_inv is None:
            raise RuntimeError(
                "Global stats not available. Call fit()/finalize() or use per_batch=True."
            )

        # Broadcast-global μ and Σ^{-1}
        mu = self.mu.to(x.device, x.dtype)  # (C,)
        cov_inv = self.cov_inv.to(x.device, x.dtype)  # (C, C)
        X = x.view(B, N, C)  # (B, N, C)
        Xc = X - mu  # broadcasting over C
        md2 = torch.einsum("bnc,cd,bnd->bn", Xc, cov_inv, Xc)  # (B, N)
        scores = md2.view(B, H, W)
        return self.run_postprocessing(scores)
