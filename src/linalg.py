# src/linalg.py
from __future__ import annotations

import numpy as np
import torch
from numba import njit


def laplacian_4nbrs(df, step: int = 10, xcol: str = "X", ycol: str = "Y"):
    """
    Dense 4-neighbor combinatorial Laplacian on a rectilinear grid.

    Neighbor criterion: |dx| + |dy| == step
    L = D - A (symmetric)
    """
    X = df[xcol].to_numpy()
    Y = df[ycol].to_numpy()

    dL1 = np.abs(X[:, None] - X[None, :]) + np.abs(Y[:, None] - Y[None, :])
    A = (dL1 == step).astype(np.float64)
    deg = A.sum(axis=1)

    L = -A
    np.fill_diagonal(L, deg)
    return torch.from_numpy(L)


@njit
def minevec_iter(H: np.ndarray, x: np.ndarray, iters: int = 100_000) -> np.ndarray:
    """
    Power iteration on H (largest |eig|). Use on H^{-1} or (H^{-1})^k
    to approximate smallest eigenvector of original H.
    """
    for _ in range(iters):
        x = H @ x
        x = x / np.linalg.norm(x)
    return x


def ground_state_symmetric(H: np.ndarray):
    """Compute the normalized ground state of a real symmetric matrix exactly by dense diagonalization."""
    H = np.asarray(H, dtype=np.float64)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square matrix.")

    if not np.allclose(H, H.T, rtol=1e-12, atol=1e-12):
        raise ValueError("H must be symmetric.")

    eigenvalues, eigenvectors = np.linalg.eigh(H)

    lam = float(eigenvalues[0])
    phi = eigenvectors[:, 0].astype(np.float64, copy=False)
    phi /= np.linalg.norm(phi)

    # Fix the arbitrary global sign for reproducible output.
    if phi.sum() < 0:
        phi = -phi

    residual = H @ phi - lam * phi
    res = float(np.linalg.norm(residual))

    opnorm = float(np.max(np.abs(eigenvalues)))
    relres = float(res / (opnorm + abs(lam) + 1e-60))

    gap = (
        float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else float("nan")
    )

    return phi, lam, res, relres, gap
