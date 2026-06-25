"""Batched non-negative least squares solver shared by the unmixing nodes.

Both :class:`~cuvis_ai.node.unmixing.nnls.NNLSUnmixing` and
:class:`~cuvis_ai.node.unmixing.nmf.NMFUnmixing` resolve per-pixel abundances by
solving ``min_{x >= 0} ||A x - b||`` for a stack of right-hand sides ``b``. This
module factors that solve out so both nodes share one pure-torch implementation
that runs on whatever device the inputs live on.

The solver uses accelerated projected gradient descent (FISTA). Its core step is
``x <- relu(y - lr * (AtA @ y - At @ b))`` with a fixed step ``lr`` equal to the
reciprocal of the spectral norm (largest eigenvalue) of ``AtA``; that step size
guarantees descent of the convex quadratic and the ``relu`` projection keeps every
iterate feasible. The Nesterov momentum extrapolation ``y`` accelerates
convergence from O(1/k) to O(1/k^2), which the ill-conditioned mixing matrices that
arise from overlapping endmember spectra need to reach a tight optimum in a
practical iteration budget.
"""

from __future__ import annotations

import torch


def nnls_batch(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Solve ``min_{x >= 0} ||A x - b||`` for a batch of right-hand sides.

    Parameters
    ----------
    a : torch.Tensor
        Mixing matrix ``A`` of shape ``[C, K]`` (channels by components).
    b : torch.Tensor
        Stacked targets of shape ``[P, C]`` (one spectrum per pixel).
    max_iter : int, optional
        Maximum number of projected-gradient iterations (default: 200).
    tol : float, optional
        Stop early once the per-iteration update norm falls below this value
        (default: 1e-6).

    Returns
    -------
    torch.Tensor
        Non-negative coefficients ``x`` of shape ``[P, K]`` solving the problem
        for every row of ``b``.
    """
    channels, components = a.shape
    pixels = b.shape[0]
    dtype = a.dtype
    device = a.device

    if pixels == 0:
        return torch.zeros(0, components, dtype=dtype, device=device)

    ata = a.transpose(0, 1) @ a  # [K, K]
    atb = b @ a  # [P, K]  ==  (At @ b^T)^T

    # Step size = 1 / spectral_norm(AtA); guards a zero/degenerate operator.
    op_norm = torch.linalg.matrix_norm(ata, ord=2)
    if op_norm <= 0:
        return torch.zeros(pixels, components, dtype=dtype, device=device)
    lr = 1.0 / op_norm

    x = torch.zeros(pixels, components, dtype=dtype, device=device)
    y = x.clone()  # momentum-extrapolated iterate
    t = 1.0
    for _ in range(max_iter):
        grad = y @ ata - atb  # [P, K]
        x_next = torch.relu(y - lr * grad)
        t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        y = x_next + ((t - 1.0) / t_next) * (x_next - x)
        update = torch.linalg.vector_norm(x_next - x)
        x = x_next
        t = t_next
        if update < tol:
            break
    return x


def reconstruction_residual(a: torch.Tensor, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the per-row Euclidean residual ``||A x - b||``.

    Parameters
    ----------
    a : torch.Tensor
        Mixing matrix ``A`` of shape ``[C, K]``.
    x : torch.Tensor
        Coefficients of shape ``[P, K]``.
    b : torch.Tensor
        Targets of shape ``[P, C]``.

    Returns
    -------
    torch.Tensor
        Residual norms of shape ``[P]``.
    """
    recon = x @ a.transpose(0, 1)  # [P, C]
    return torch.linalg.vector_norm(recon - b, dim=1)
