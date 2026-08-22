# src/featurize.py
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.linalg import laplacian_4nbrs


def _resolve_cols(df: pd.DataFrame, n_grey: int = 25) -> dict:
    """Tolerant column name resolution."""
    cols = {c.lower(): c for c in df.columns}
    vcol = cols.get(
        "numpoints_smooth", cols.get("smoothed_numpoints", "NUMPOINTS_SMOOTH")
    )

    greys = []
    for k in range(1, n_grey + 1):
        greys.append(cols.get(f"grey{k}", f"Grey{k}"))

    return {
        "id": cols.get("id", "id"),
        "x": cols.get("x", "X"),
        "y": cols.get("y", "Y"),
        # prefer Zdrio if present; else Z
        "z": cols.get("zdrio", cols.get("z", "Z")),
        "drio": cols.get("drio", "drio"),
        "greys": greys,  # Grey1..Grey25
        "v": vcol,
    }


def build_Tdata_grey25(
    df_region: pd.DataFrame, df_total: pd.DataFrame, n_grey: int = 25
) -> torch.Tensor:
    """
    T columns (normalized):
      [id, v_smooth, X, Y, Z, drio, Grey1..Grey{n_grey}]
    Normalization:
      id,v,X,Y -> 1
      Z -> max(Z_total)
      drio -> max(drio_total)
      Grey_k -> max(Grey_k_total) per k
    """
    C_r = _resolve_cols(df_region, n_grey=n_grey)
    C_t = _resolve_cols(df_total, n_grey=n_grey)

    zmax = float(df_total[C_t["z"]].max())
    dmax = float(df_total[C_t["drio"]].max())
    if zmax == 0 or dmax == 0:
        raise ValueError("Invalid normalization: max(Z) or max(drio) is 0.")

    m_list = [1.0, 1.0, 1.0, 1.0, zmax, dmax]
    for colk in C_t["greys"]:
        if colk not in df_total.columns:
            raise ValueError(f"Missing required column in df_total: {colk}")
        gmax = float(df_total[colk].max())
        if gmax == 0:
            gmax = 1.0
        m_list.append(gmax)

    m = torch.tensor(m_list, dtype=torch.float64)

    rows = []
    for _, r in df_region.iterrows():
        row = [
            float(r.get(C_r["id"], 0.0)),
            float(r[C_r["v"]]),
            float(r[C_r["x"]]),
            float(r[C_r["y"]]),
            float(r[C_r["z"]]),
            float(r[C_r["drio"]]),
        ]
        for colk in C_r["greys"]:
            if colk not in df_region.columns:
                raise ValueError(f"Missing required column in df_region: {colk}")
            row.append(float(r[colk]))
        rows.append(row)

    T = torch.tensor(np.asarray(rows, dtype=np.float64))
    return T / m


def phiin_from_T(T: torch.Tensor) -> torch.Tensor:
    """Normalized smoothed counts (column 1)."""
    phiin = T[:, 1].clone()
    s = torch.sum(phiin)
    if s <= 0:
        raise ValueError("Sum of smoothed counts is non-positive.")
    return phiin / s


def compute_V(
    df_region: pd.DataFrame, phiin: torch.Tensor, step: int = 10
) -> torch.Tensor:
    """
    V = -(L·Phiin)/Phiin with Phiin = sqrt(phiin). Shift to be >= 0.
    """
    Phiin = torch.sqrt(phiin).to(dtype=torch.float64)
    L = laplacian_4nbrs(df_region, step=step).to(dtype=torch.float64)
    y = L.matmul(Phiin)

    eps = 1e-30
    phi_safe = torch.where(Phiin.abs() < eps, eps, Phiin)
    V = -y / phi_safe
    if V.min() < 0:
        V = V - V.min()
    return V


def build_T_and_V(
    df_region: pd.DataFrame,
    df_total: pd.DataFrame,
    step: int = 10,
):
    """Return float32 feature and target tensors aligned with ``df_region``."""
    T = build_Tdata_grey25(df_region, df_total, n_grey=25)

    phiin = phiin_from_T(T)
    V = compute_V(df_region, phiin, step=step)
    return T.to(torch.float32), V.to(torch.float32)
