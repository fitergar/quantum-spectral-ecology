# src/neighborhoods.py
from __future__ import annotations

import argparse
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd

from src.featurize import _resolve_cols as _cols
from src.utils import dataset_paths, die, ensure_dir, save_stage_record, step


def build_neighborhoods(
    df_all: pd.DataFrame,
    C: dict,
    *,
    half_size: float = 60.0,
    drio_center: float = 0.0,
    drio_tol: float = 1e-9,
):
    xcol, ycol, dcol, idcol = C["x"], C["y"], C["drio"], C["id"]

    centers = df_all[np.abs(df_all[dcol] - drio_center) <= drio_tol].copy()
    if centers.empty:
        raise ValueError("No river-center cells found (drio ~= drio_center).")

    neighborhoods = []
    for _, c in centers.iterrows():
        dx = (df_all[xcol] - c[xcol]).abs()
        dy = (df_all[ycol] - c[ycol]).abs()
        mask = np.maximum(dx, dy) <= half_size

        nb = df_all.loc[mask].copy().reset_index(drop=True)

        # Store the center metadata once per neighborhood for prediction diagnostics.
        nb["Centroid"] = pd.Series("", index=nb.index, dtype=object)
        nb["CentroX"] = pd.Series("", index=nb.index, dtype=object)
        nb["CentroY"] = pd.Series("", index=nb.index, dtype=object)
        nb.at[0, "Centroid"] = c[idcol]
        nb.at[0, "CentroX"] = c[xcol]
        nb.at[0, "CentroY"] = c[ycol]

        neighborhoods.append(nb)

    return neighborhoods


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate Chebyshev neighborhoods around river cells."
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--half-size", type=float, default=60.0)
    p.add_argument("--drio-center", type=float, default=0.0)
    p.add_argument("--drio-tol", type=float, default=1e-9)
    p.add_argument("--outfile", default="VecindadesLinf.pkl")
    return p.parse_args()


def main():
    args = parse_args()
    paths = dataset_paths(args.dataset)

    prepared = paths["prepared_csv"]
    if not prepared.exists():
        die(f"Prepared CSV not found: {prepared}. Run prepare.py first.")

    step(f"Loading {prepared}")
    df_all = pd.read_csv(prepared)
    C = _cols(df_all)

    for key in ("id", "x", "y", "drio"):
        if C[key] not in df_all.columns:
            die(f"Required column {C[key]!r} (for {key}) not found in prepared.csv.")

    step(
        f"Building neighborhoods (half_size={args.half_size} m, drio~={args.drio_center})"
    )
    neighborhoods = build_neighborhoods(
        df_all,
        C,
        half_size=args.half_size,
        drio_center=args.drio_center,
        drio_tol=args.drio_tol,
    )

    out_dir = Path(paths["out_dir"])
    ensure_dir(out_dir)

    out_pkl = out_dir / args.outfile
    step(f"Saving {len(neighborhoods)} neighborhoods -> {out_pkl}")
    with open(out_pkl, "wb") as f:
        pkl.dump(neighborhoods, f)

    save_stage_record(
        out_dir,
        "neighborhoods",
        args,
        inputs={"prepared": prepared},
        outputs={"neighborhoods": out_pkl},
    )

    step("Done.")


if __name__ == "__main__":
    main()
