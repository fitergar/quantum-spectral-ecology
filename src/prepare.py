"""Compute greyscale features and the smoothed training signal."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import ndimage

from src.utils import dataset_paths, ensure_dir, save_stage_record, step

# ----------------- CLI -----------------


def parse_args():
    p = argparse.ArgumentParser("Prepare data (Grey + smoothing)")

    p.add_argument("--dataset", required=True)

    p.add_argument("--gamma", type=float, default=3.0)
    p.add_argument("--grey-subcells", action="store_true")

    p.add_argument("--smooth", choices=["none", "step"], default="step")
    p.add_argument("--sigma-meters", type=float, default=8.0)
    p.add_argument("--cellsize", type=float, default=10.0)
    p.add_argument("--s", type=int, default=100)

    p.add_argument("--xcol", default="X")
    p.add_argument("--ycol", default="Y")
    p.add_argument("--vcol", default="NUMPOINTS")
    p.add_argument("--max-drio", type=float, default=80.0)

    return p.parse_args()


# ----------------- Grey computation -----------------


def find_rgb_triplets(cols, nmax=25) -> int:
    for k in range(1, nmax + 1):
        if not all(f"{c}{k}" in cols for c in ("R", "G", "B")):
            return k - 1
    return nmax


def compute_grey(df, gamma: float, n_triplets: int) -> pd.Series:
    G = np.stack(
        [
            0.299 * df[f"R{k}"].astype(float)
            + 0.587 * df[f"G{k}"].astype(float)
            + 0.114 * df[f"B{k}"].astype(float)
            for k in range(1, n_triplets + 1)
        ],
        axis=1,
    )

    gmin = G.min(axis=1, keepdims=True)
    gmax = G.max(axis=1, keepdims=True)
    gnorm = (G - gmin) / np.clip(gmax - gmin, 1e-12, None)

    w = (1 - gnorm) ** gamma
    grey = (gnorm * w).sum(axis=1) / (w.sum(axis=1) + 1e-12)

    return pd.Series((grey * 100).clip(0, 100), index=df.index, name="Grey")


def compute_grey_triplets(df, n_triplets: int) -> pd.DataFrame:
    out = {}
    for k in range(1, n_triplets + 1):
        out[f"Grey{k}"] = (
            0.299 * df[f"R{k}"] + 0.587 * df[f"G{k}"] + 0.114 * df[f"B{k}"]
        )
    return pd.DataFrame(out, index=df.index)


# ----------------- Smoothing -----------------


def smooth_step_embedding(df, *, sigma_meters, cellsize, s, xcol, ycol, vcol, outcol):

    df = df.copy()
    df[xcol] = np.rint(df[xcol]).astype(int)
    df[ycol] = np.rint(df[ycol]).astype(int)

    xs = np.sort(df[xcol].unique())
    ys = np.sort(df[ycol].unique())
    xi = {x: i for i, x in enumerate(xs)}
    yi = {y: i for i, y in enumerate(ys)}

    u = np.zeros((len(ys), len(xs)))
    for _, r in df.iterrows():
        u[yi[r[ycol]], xi[r[xcol]]] = r[vcol]

    fine = np.repeat(np.repeat(u, s, 0), s, 1)
    sigma_px = sigma_meters * (s / cellsize)
    fine = ndimage.gaussian_filter(fine, sigma_px, truncate=10, mode="nearest")

    Ny, Nx = u.shape
    coarse = fine.reshape(Ny, s, Nx, s).mean(axis=(1, 3))

    rows = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            rows.append({xcol: x, ycol: y, outcol: coarse[j, i]})

    return pd.DataFrame(rows)


# ----------------- Region mask -----------------


def training_region_mask(df, *, xcol, ycol, max_drio):
    X, Y = df[xcol], df[ycol]

    region = (
        ((327634.1 < X) & (X < 327945) & (1856536.44 < Y) & (Y < 1856805.7))
        | ((1856805.7 <= Y) & (Y < 1856881) & (327727 < X) & (X < 327946))
        | ((1856881 <= Y) & (Y < 1856903) & (327805 < X) & (X < 327946))
        | ((1856903 <= Y) & (Y < 1856923) & (327807 < X) & (X < 327946))
    )

    if "drio" in df:
        region &= df["drio"] <= max_drio

    return region


# ----------------- main -----------------


def main(args):
    paths = dataset_paths(args.dataset)

    step(f"Reading {paths['raw_csv']}")
    df = pd.read_csv(paths["raw_csv"])

    # Resolve user-selectable column names once.
    cols = {c.lower(): c for c in df.columns}
    xcol = cols.get(args.xcol.lower(), args.xcol)
    ycol = cols.get(args.ycol.lower(), args.ycol)
    vcol = cols.get(args.vcol.lower(), args.vcol)

    df[xcol] = np.rint(df[xcol]).astype(int)
    df[ycol] = np.rint(df[ycol]).astype(int)

    # ---- GREY ----
    n_trip = find_rgb_triplets(df.columns)
    if n_trip > 0:
        step(f"Computing Grey from {n_trip} RGB triplets (gamma={args.gamma})")
        df["Grey"] = compute_grey(df, args.gamma, n_trip)
        if args.grey_subcells:
            df = pd.concat([df, compute_grey_triplets(df, n_trip)], axis=1)
    else:
        step("Grey skipped (no RGB triplets found)")

    # ---- SMOOTHING ----
    if args.smooth == "step":
        reg_mask = training_region_mask(
            df, xcol=xcol, ycol=ycol, max_drio=args.max_drio
        )
        df_reg = df.loc[reg_mask, [xcol, ycol, vcol]].copy()

        if not df_reg.empty:
            step(f"Smoothing {vcol} on {reg_mask.sum()} training cells")
            smooth_col = f"{vcol}_SMOOTH"

            sm = smooth_step_embedding(
                df_reg,
                sigma_meters=args.sigma_meters,
                cellsize=args.cellsize,
                s=args.s,
                xcol=xcol,
                ycol=ycol,
                vcol=vcol,
                outcol=smooth_col,
            )

            # restrict smoothing to true training cells only
            sm = sm.merge(
                df_reg[[xcol, ycol]].drop_duplicates(),
                on=[xcol, ycol],
                how="inner",
            )

            # renormalize to preserve total mass on the training region
            scale = df_reg[vcol].sum() / sm[smooth_col].sum()
            sm[smooth_col] *= scale

            # diagnostics BEFORE quantization
            vsm_raw = sm[smooth_col].to_numpy(float)
            step(
                f"{smooth_col} (raw): min={vsm_raw.min():.6e}, max={vsm_raw.max():.6e}"
            )

            # merge into full df (cells outside region stay NaN)
            df = df.merge(sm, on=[xcol, ycol], how="left")

            # ---- precision control + soft floor (preserves decay to ~0) ----
            if smooth_col in df.columns:
                v = df[smooth_col].to_numpy(dtype=float, copy=True)
                m = ~np.isnan(v)
                vp = v[m]

                vp = np.maximum(vp, 0.0)

                q = 1e-30
                vp = np.round(vp / q) * q

                floor = q
                vp = np.where((vp > 0) & (vp < floor), floor, vp)

                v[m] = vp
                df[smooth_col] = v

                # diagnostics AFTER quantization
                step(
                    f"{smooth_col} (quantized): min={vp.min():.6e}, max={vp.max():.6e}"
                )

    # ---- SAVE ----
    out_dir = ensure_dir(paths["out_dir"])

    step(f"Saving {paths['prepared_csv']}")
    df.to_csv(paths["prepared_csv"], index=False)
    save_stage_record(
        out_dir,
        "prepare",
        args,
        inputs={"dataset": paths["raw_csv"]},
        outputs={"prepared": paths["prepared_csv"]},
    )
    step("Done.")


if __name__ == "__main__":
    main(parse_args())
