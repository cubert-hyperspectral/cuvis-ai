"""Laplacian Anomaly Detector (LAD) for hyperspectral anomaly detection.

This module implements the Laplacian Anomaly Detector, a graph-based approach for
detecting spectral anomalies in hyperspectral images. LAD constructs a spectral graph
using Cauchy similarity weights and computes anomaly scores based on the graph Laplacian.

The LAD algorithm identifies anomalies by measuring how unusual a pixel's spectral
signature is within the spectral manifold learned from background data. Unlike RX
detectors that assume Gaussian distributions, LAD captures nonlinear manifold structures
through graph construction.

Reference:
    Gu, Y., Liu, Y., & Zhang, Y. (2008). "A selective KPCA algorithm based on high-order
    statistics for anomaly detection in hyperspectral imagery." IEEE Geoscience and
    Remote Sensing Letters, 5(1), 43-47.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec

## This node is not approeved
# missing unfreeze/freeze
# missing approved documentation and alignment with current API
# could we use streaming accumulators for mean and Laplacian construction


class LADGlobal(Node):
    """Laplacian Anomaly Detector (global), variant 'C' (Cauchy), port-based.

    This is the new cuvis.ai v3 implementation of the LAD detector. It follows the
    same mathematical definition as the legacy v2 `LADGlobal`, but exposes a
    port-based interface compatible with `CuvisPipeline`, `StatisticalTrainer`,
    and `GradientTrainer`.

    Ports
    -----
    INPUT_SPECS
        ``data`` : float32, shape (-1, -1, -1, -1)
            Input hyperspectral cube in BHWC format.
    OUTPUT_SPECS
        ``scores`` : float32, shape (-1, -1, -1, 1)
            Per pixel anomaly scores in BHW1 format.

    Parameters
    ----------
    eps : float, default 1e-8
        Small epsilon value for numerical stability in Laplacian construction.
    normalize_laplacian : bool, default True
        If True, applies symmetric normalization: L = D^{-1/2} (D - A) D^{-1/2}.
        If False, uses unnormalized Laplacian: L = D - A.
    use_numpy_laplacian : bool, default True
        If True, constructs the Laplacian matrix using NumPy (float64, 1e-12 eps)
        for parity with reference implementations. If False, uses pure PyTorch.

    Training
    --------
    After statistical initialization via `statistical_initialization()`, the node can be made trainable
    by calling `unfreeze()`. This converts the mean `M` and Laplacian `L` buffers
    to trainable `nn.Parameter` objects, enabling gradient-based fine-tuning.

    Example
    -------
    >>> lad = LADGlobal(num_channels=61)
    >>> stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    >>> stat_trainer.fit()  # Statistical initialization
    >>> lad.unfreeze()  # Enable gradient training
    >>> grad_trainer = GradientTrainer(pipeline=pipeline, datamodule=datamodule, ...)
    >>> grad_trainer.fit()  # Gradient-based fine-tuning
    """

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube (BHWC format)",
        )
    }

    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="LAD anomaly scores per pixel (BHW1 format)",
        )
    }

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-8,
        normalize_laplacian: bool = True,
        use_numpy_laplacian: bool = True,
        **kwargs: Any,
    ) -> None:
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.normalize_laplacian = bool(normalize_laplacian)
        self.use_numpy_laplacian = bool(use_numpy_laplacian)

        super().__init__(
            num_channels=self.num_channels,
            eps=self.eps,
            normalize_laplacian=self.normalize_laplacian,
            use_numpy_laplacian=self.use_numpy_laplacian,
            **kwargs,
        )

        # Streaming accumulators (float64 for numerical stability)
        self.register_buffer(
            "_mean_run", torch.zeros(self.num_channels, dtype=torch.float64)
        )  # (C,)
        self._count: int = 0
        # Model buffers
        self.register_buffer("M", torch.zeros(self.num_channels, dtype=torch.float64))  # (C,)
        self.register_buffer(
            "L", torch.zeros(self.num_channels, self.num_channels, dtype=torch.float64)
        )  # (C, C)
        self._statistically_initialized = False

    # ------------------------------------------------------------------
    # Statistical initialization API
    # ------------------------------------------------------------------
    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Compute global mean M and Laplacian L from a port-based input stream.

        Parameters
        ----------
        input_stream : InputStream
            Iterator yielding dicts matching INPUT_SPECS.
            Expected format: ``{"data": tensor}`` where tensor is BHWC.
        """
        self.reset()

        for batch_data in input_stream:
            x = batch_data.get("data")
            if x is not None:
                self.update(x)

        if self._count <= 0:
            raise RuntimeError("No samples provided to LADGlobal.statistical_initialization()")

        self.finalize()
        self._statistically_initialized = True

    @torch.no_grad()
    def update(self, batch_bhwc: torch.Tensor) -> None:
        """Update running mean statistics from a BHWC batch."""
        B, H, W, C = batch_bhwc.shape
        X = batch_bhwc.reshape(B * H * W, C).to(dtype=torch.float64)
        m = X.shape[0]
        if m <= 0:
            return

        mean_b = X.mean(dim=0)

        if self._count == 0:
            self._mean_run = mean_b
            self._count = m
        else:
            tot = self._count + m
            delta = mean_b - self._mean_run
            self._mean_run = self._mean_run + delta * (m / tot)
            self._count = tot

        self._statistically_initialized = False

    @torch.no_grad()
    def finalize(self) -> None:
        """Finalize mean and Laplacian from accumulated statistics."""
        if self._count <= 0 or self._mean_run.numel() == 0:
            raise RuntimeError("No samples accumulated for LADGlobal.finalize()")

        M = self._mean_run.clone().to(dtype=torch.float64)
        C = M.shape[0]
        a = M.mean()

        if self.use_numpy_laplacian:
            # NumPy implementation for exact parity with legacy version
            M_np = M.detach().cpu().numpy()
            A_abs = np.abs(M_np[:, None] - M_np[None, :])
            a_np = float(M_np.mean())
            A_np = 1.0 / (1.0 + (A_abs / (a_np + 1e-12)) ** 2)
            np.fill_diagonal(A_np, 0.0)
            D_np = np.diag(A_np.sum(axis=1))
            L_np = D_np - A_np

            if self.normalize_laplacian:
                d_np = np.diag(D_np)
                d_inv_sqrt_np = np.where(d_np > 0, 1.0 / (np.sqrt(d_np) + 1e-12), 0.0)
                D_inv_sqrt_np = np.diag(d_inv_sqrt_np)
                L_np = D_inv_sqrt_np @ L_np @ D_inv_sqrt_np

            L = torch.from_numpy(L_np).to(dtype=torch.float64, device=M.device)
        else:
            Mi = M.view(C, 1)
            Mj = M.view(1, C)
            denom = a + torch.tensor(1e-12, dtype=torch.float64, device=M.device)
            diff = torch.abs(Mi - Mj) / denom
            A = 1.0 / (1.0 + diff.pow(2))
            A.fill_diagonal_(0.0)

            D = torch.diag(A.sum(dim=1))
            L = D - A

            if self.normalize_laplacian:
                d = torch.diag(D)
                d_inv_sqrt = torch.where(
                    d > 0,
                    1.0 / torch.sqrt(d + torch.tensor(1e-12, dtype=torch.float64, device=M.device)),
                    torch.zeros_like(d),
                )
                D_inv_sqrt = torch.diag(d_inv_sqrt)
                L = D_inv_sqrt @ L @ D_inv_sqrt

        self.M = M
        self.L = L
        self._statistically_initialized = True

    @torch.no_grad()
    def reset(self) -> None:
        """Reset all statistics and model parameters to initial state.

        Clears the streaming mean accumulator (_mean_run), sample count (_count),
        global mean (M), and Laplacian matrix (L). After reset, the detector must
        be re-initialized via statistical_initialization() before inference.

        Notes
        -----
        Use this method to re-initialize the detector with different training data
        or when switching between different spectral distributions.
        """
        self.register_buffer(
            "_mean_run", torch.zeros(self.num_channels, dtype=torch.float64)
        )  # (C,)
        self._count: int = 0
        # Model buffers
        self.register_buffer("M", torch.zeros(self.num_channels, dtype=torch.float64))  # (C,)
        self.register_buffer(
            "L", torch.zeros(self.num_channels, self.num_channels, dtype=torch.float64)
        )  # (C, C)
        self._statistically_initialized = False

    def unfreeze(self) -> None:
        """Convert M and L buffers to trainable nn.Parameters."""
        if self.M.numel() > 0 and self.L.numel() > 0:
            device = self.M.device
            # Store current values
            M_data = self.M.clone()
            L_data = self.L.clone()

            # Remove buffer registrations
            delattr(self, "M")
            delattr(self, "L")

            # Register as parameters
            self.M = nn.Parameter(M_data, requires_grad=True)
            self.L = nn.Parameter(L_data, requires_grad=True)
            self.M.to(device=device)
            self.L.to(device=device)

        # Call parent to enable requires_grad
        super().unfreeze()

    # ------------------------------------------------------------------
    # Forward & serialization
    # ------------------------------------------------------------------
    def forward(self, data: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Compute LAD anomaly scores for a BHWC cube.

        Parameters
        ----------
        data : torch.Tensor
            Input tensor in BHWC format.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with key ``"scores"`` containing BHW1 anomaly scores.
        """
        if self.M.numel() == 0 or self.L.numel() == 0 or not self._statistically_initialized:
            raise RuntimeError(
                "LADGlobal not finalized. Call statistical_initialization() before forward()."
            )

        B, H, W, C = data.shape
        N = H * W

        X = data.view(B, N, C)

        Xc = X - self.M.to(dtype=X.dtype)
        L = self.L.to(dtype=X.dtype)

        scores = torch.einsum("bnc,cd,bnd->bn", Xc, L, Xc).view(B, H, W).unsqueeze(-1)
        return {"scores": scores}


__all__ = ["LADGlobal"]
