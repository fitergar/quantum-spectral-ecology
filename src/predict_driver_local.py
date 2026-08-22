# src/predict_driver_local.py
from __future__ import annotations

import argparse
import pickle as pkl
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.featurize import _resolve_cols as _cols
from src.featurize import build_Tdata_grey25
from src.linalg import ground_state_symmetric, laplacian_4nbrs
from src.models import MLPDriver
from src.utils import (
    dataset_paths,
    die,
    ensure_dir,
    save_stage_record,
    select_torch_device,
    sha256_file,
    step,
)

# ----------------- CLI -----------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Local driver-only predictions with neighborhood averaging"
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--cuda", action="store_true")

    p.add_argument("--step", type=int, default=10)

    p.add_argument("--neigh-file", default="VecindadesLinf.pkl")
    p.add_argument("--out-csv", default="pred_driver_localavg.csv")
    p.add_argument(
        "--conv-print-every",
        type=int,
        default=25,
        help="Print ground-state diagnostics every N neighborhoods (0 disables).",
    )
    p.add_argument(
        "--conv-warn-relres",
        type=float,
        default=1e-10,
        help="Warn if the relative residual exceeds this threshold.",
    )
    p.add_argument("--metrics-csv", default="ground_state_metrics_driver.csv")

    return p.parse_args()


# ----------------- main -----------------


def main():
    args = parse_args()
    device = select_torch_device(cuda=args.cuda)
    step(f"Device: {device}")

    paths = dataset_paths(args.dataset)
    out_dir = Path(paths["out_dir"])
    ensure_dir(out_dir)

    prepared = paths["prepared_csv"]
    if not prepared.exists():
        die("prepared.csv not found. Run prepare.py first.")

    df_all = pd.read_csv(prepared)
    C = _cols(df_all)

    idcol = C["id"]
    geom_cols = [c for c in (C["id"], C["x"], C["y"], C["drio"]) if c in df_all.columns]

    # ---------- neighborhoods ----------
    neigh_path = out_dir / args.neigh_file
    if not neigh_path.exists():
        die("Neighborhoods file not found. Run neighborhoods.py first.")

    with open(neigh_path, "rb") as f:
        neighborhoods = pkl.load(f)

    if not neighborhoods:
        die("Neighborhood list is empty.")

    step(f"Loaded {len(neighborhoods)} neighborhoods")

    # ---------- driver ----------
    dstate = torch.load(
        out_dir / "model_driver.pt", map_location=device, weights_only=True
    )
    dmeta = dstate["meta"]
    if dmeta.get("prepared_sha256") != sha256_file(prepared):
        die("Driver checkpoint was not trained from the current prepared.csv.")

    poly_degree = int(dmeta["poly_degree"])
    hidden = list(dmeta["hidden"])

    driver = MLPDriver(poly_degree, *hidden).to(device)
    driver.load_state_dict(dstate["state_dict"])
    driver.eval()

    for p in driver.parameters():
        p.requires_grad = False

    step(f"Loaded driver: polynomial degree={poly_degree}, hidden={hidden}")

    # ---------- accumulation ----------
    sum_prob = defaultdict(float)
    count = defaultdict(int)
    metrics_rows = []

    for k, nb in enumerate(neighborhoods):
        step(f"[{k + 1}/{len(neighborhoods)}] Neighborhood N={len(nb)}")

        T_nb = build_Tdata_grey25(nb, df_all, n_grey=25).to(torch.float32).to(device)

        with torch.no_grad():
            V_nb = driver(T_nb).cpu().numpy().astype(np.float64)

        L = laplacian_4nbrs(nb, step=args.step).numpy().astype(np.float64)
        H = L + np.diag(V_nb)

        Phi, lam, res, relres, gap = ground_state_symmetric(H)
        Prob = Phi**2
        centroid = (
            nb["Centroid"].iloc[0] if "Centroid" in nb.columns and len(nb) else np.nan
        )
        centrox = (
            nb["CentroX"].iloc[0] if "CentroX" in nb.columns and len(nb) else np.nan
        )
        centroy = (
            nb["CentroY"].iloc[0] if "CentroY" in nb.columns and len(nb) else np.nan
        )

        metrics_rows.append(
            {
                "k": k,
                "neigh_size": len(nb),
                "centroid_id": centroid,
                "centro_x": centrox,
                "centro_y": centroy,
                "ground_state_energy": float(lam),
                "spectral_gap": float(gap),
                "residual_norm": float(res),
                "relative_residual": float(relres),
            }
        )

        if args.conv_print_every and ((k + 1) % args.conv_print_every == 0 or k == 0):
            msg = f"  ground state: lambda={lam:.6e} | gap={gap:.3e} | res={res:.3e} | relres={relres:.3e}"
            if relres > args.conv_warn_relres:
                msg += "  [WARN]"
            step(msg)

        for cid, p in zip(nb[idcol].to_numpy(), Prob):
            sum_prob[int(cid)] += float(p)
            count[int(cid)] += 1

    # ---------- global average ----------
    df_geom = df_all[geom_cols].drop_duplicates(subset=idcol)
    df_geom = df_geom[df_geom[idcol].isin(sum_prob)]

    df_geom["Probav"] = df_geom[idcol].map(sum_prob) / df_geom[idcol].map(count)
    df_geom["Probav"] /= df_geom["Probav"].sum()

    # ---------- population calibration ----------
    vcol = C["v"]
    if vcol in df_all.columns:
        train = df_all[[idcol, vcol]].dropna()
        if not train.empty:
            m = df_geom.merge(train, on=idcol)
            scale = m[vcol].sum() / m["Probav"].sum()
            df_geom["Pop_est"] = df_geom["Probav"] * scale
            step(f"Population calibration scale={scale:.6e}")

    # ---------- save ----------
    out_csv = out_dir / args.out_csv
    cols = (
        geom_cols + ["Probav"] + (["Pop_est"] if "Pop_est" in df_geom.columns else [])
    )
    df_geom[cols].to_csv(out_csv, index=False)
    metrics_path = out_dir / args.metrics_csv
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    step(f"Saved ground-state metrics: {metrics_path}")
    step(f"Saved: {out_csv}")

    save_stage_record(
        out_dir,
        "predict_driver_local",
        args,
        inputs={
            "prepared": prepared,
            "neighborhoods": neigh_path,
            "driver_model": out_dir / "model_driver.pt",
        },
        outputs={"prediction": out_csv, "ground_state_metrics": metrics_path},
        device=device,
    )


if __name__ == "__main__":
    main()
