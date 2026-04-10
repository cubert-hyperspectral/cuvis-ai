"""GPU/CPU Poisson inpainting for channel-last images."""

from __future__ import annotations

import torch


def _cg_solve(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    *,
    max_iter: int,
    tol: float,
    x0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Solve ``matrix @ x = rhs`` with batched Conjugate Gradient.

    Parameters
    ----------
    matrix
        Sparse square matrix ``[N, N]`` (COO).
    rhs
        Dense right-hand side ``[N, C]``.
    max_iter
        Maximum iterations.
    tol
        Absolute residual tolerance on the worst channel.
    x0
        Optional initial guess ``[N, C]``.
    """
    if rhs.ndim != 2:
        raise ValueError(f"rhs must be 2D [N, C], got shape {tuple(rhs.shape)}")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")
    if tol <= 0:
        raise ValueError("tol must be > 0")

    x = torch.zeros_like(rhs) if x0 is None else x0.clone()
    r = rhs - torch.sparse.mm(matrix, x)
    p = r.clone()
    rs_old = (r * r).sum(dim=0)

    if torch.sqrt(rs_old.max()).item() <= tol:
        return x

    eps = torch.finfo(rhs.dtype).eps

    for _ in range(max_iter):
        ap = torch.sparse.mm(matrix, p)
        p_ap = (p * ap).sum(dim=0)

        alpha = torch.zeros_like(rs_old)
        valid_alpha = p_ap.abs() > eps
        alpha[valid_alpha] = rs_old[valid_alpha] / p_ap[valid_alpha]

        x = x + p * alpha.unsqueeze(0)
        r = r - ap * alpha.unsqueeze(0)

        rs_new = (r * r).sum(dim=0)
        if torch.sqrt(rs_new.max()).item() <= tol:
            return x

        beta = torch.zeros_like(rs_old)
        valid_beta = rs_old.abs() > eps
        beta[valid_beta] = rs_new[valid_beta] / rs_old[valid_beta]

        p = r + p * beta.unsqueeze(0)
        rs_old = rs_new

    return x


def poisson_inpaint(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Inpaint masked pixels by solving the Laplace equation.

    Parameters
    ----------
    image
        Tensor ``[H, W, C]`` on CPU or GPU.
    mask
        Tensor ``[H, W]`` where ``True`` marks unknown pixels to inpaint.
    max_iter
        Maximum CG iterations.
    tol
        CG residual tolerance.
    """
    if image.ndim != 3:
        raise ValueError(f"image must be [H, W, C], got shape {tuple(image.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be [H, W], got shape {tuple(mask.shape)}")
    if image.shape[:2] != tuple(mask.shape):
        raise ValueError(
            "image spatial shape and mask shape must match, got "
            f"{tuple(image.shape[:2])} vs {tuple(mask.shape)}"
        )
    if not image.is_floating_point():
        raise TypeError(f"image must be floating point, got {image.dtype}")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")
    if tol <= 0:
        raise ValueError("tol must be > 0")

    original_dtype = image.dtype
    compute_dtype = image.dtype
    if image.dtype in {torch.float16, torch.bfloat16}:
        compute_dtype = torch.float32

    image_work = image.to(dtype=compute_dtype)
    mask_bool = mask.to(device=image.device, dtype=torch.bool)

    h, w, channels = image_work.shape
    flat_mask = mask_bool.reshape(-1)
    unknown_flat = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
    n_unknown = int(unknown_flat.numel())

    if n_unknown == 0:
        return image.clone()

    if n_unknown == h * w:
        raise ValueError("mask covers the whole image; Poisson inpainting has no known boundary")

    rows = torch.div(unknown_flat, w, rounding_mode="floor")
    cols = unknown_flat % w

    idx_map = torch.full((h * w,), -1, dtype=torch.long, device=image.device)
    idx_map[unknown_flat] = torch.arange(n_unknown, dtype=torch.long, device=image.device)

    diag = torch.zeros(n_unknown, dtype=compute_dtype, device=image.device)
    rhs = torch.zeros((n_unknown, channels), dtype=compute_dtype, device=image.device)
    image_flat = image_work.reshape(-1, channels)

    row_parts: list[torch.Tensor] = []
    col_parts: list[torch.Tensor] = []
    val_parts: list[torch.Tensor] = []
    has_known_boundary = False

    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        n_rows = rows + dr
        n_cols = cols + dc
        valid = (n_rows >= 0) & (n_rows < h) & (n_cols >= 0) & (n_cols < w)
        if not torch.any(valid):
            continue

        diag = diag + valid.to(dtype=compute_dtype)

        local_rows = torch.nonzero(valid, as_tuple=False).squeeze(1)
        neigh_flat = n_rows[valid] * w + n_cols[valid]
        neigh_unknown = idx_map[neigh_flat]

        is_unknown_neighbor = neigh_unknown >= 0
        if torch.any(is_unknown_neighbor):
            off_rows = local_rows[is_unknown_neighbor]
            off_cols = neigh_unknown[is_unknown_neighbor]
            off_vals = torch.full(
                (off_rows.numel(),),
                -1.0,
                dtype=compute_dtype,
                device=image.device,
            )
            row_parts.append(off_rows)
            col_parts.append(off_cols)
            val_parts.append(off_vals)

        is_known_neighbor = ~is_unknown_neighbor
        if torch.any(is_known_neighbor):
            has_known_boundary = True
            known_rows = local_rows[is_known_neighbor]
            known_flat = neigh_flat[is_known_neighbor]
            rhs.index_add_(0, known_rows, image_flat[known_flat])

    if not has_known_boundary:
        raise ValueError("mask has no known neighboring pixels; Poisson system is singular")

    diag_rows = torch.arange(n_unknown, dtype=torch.long, device=image.device)
    row_parts.append(diag_rows)
    col_parts.append(diag_rows)
    val_parts.append(diag)

    mat_rows = torch.cat(row_parts, dim=0)
    mat_cols = torch.cat(col_parts, dim=0)
    mat_vals = torch.cat(val_parts, dim=0)

    matrix = torch.sparse_coo_tensor(
        indices=torch.stack((mat_rows, mat_cols), dim=0),
        values=mat_vals,
        size=(n_unknown, n_unknown),
        dtype=compute_dtype,
        device=image.device,
    ).coalesce()

    solved = _cg_solve(matrix, rhs, max_iter=max_iter, tol=tol)

    result = image_work.clone()
    result.reshape(-1, channels)[unknown_flat] = solved
    return result.to(dtype=original_dtype)


__all__ = ["poisson_inpaint"]
