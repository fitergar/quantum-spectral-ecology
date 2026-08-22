# src/train_driver.py
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import torch

from src.featurize import _resolve_cols as _cols
from src.featurize import build_T_and_V
from src.models import MLPDriver
from src.utils import (
    configure_torch,
    dataset_paths,
    die,
    ensure_dir,
    git_provenance,
    runtime_provenance,
    save_stage_record,
    sha256_file,
    step,
)

# ----------------- CLI -----------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Train driver model (depends ONLY on drio)."
    )

    p.add_argument("--dataset", required=True)

    p.add_argument("--driver-poly", type=int, default=5)
    p.add_argument("--driver-h1", type=int, default=400)
    p.add_argument("--driver-h2", type=int, default=500)
    p.add_argument("--driver-h3", type=int, default=450)

    p.add_argument("--epochs", type=int, default=100_000)
    p.add_argument("--lr0-inv", type=float, default=100.0)
    p.add_argument("--step-size", type=float, default=1.0)

    p.add_argument("--log-every", type=int, default=5000)
    p.add_argument("--metrics-csv", default="train_driver_metrics.csv")

    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--cuda", action="store_true")

    return p.parse_args()


# ----------------- metrics -----------------


def nrmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((y_pred - y_true) ** 2)
    return torch.sqrt(mse) / (torch.max(y_true) - torch.min(y_true) + 1e-12)


# ----------------- main -----------------


def main():
    args = parse_args()
    paths = dataset_paths(args.dataset)

    device = configure_torch(args.seed, cuda=args.cuda)
    step(f"Device: {device}")

    if not paths["prepared_csv"].exists():
        die("prepared.csv not found. Run prepare.py first.")

    df = pd.read_csv(paths["prepared_csv"])

    # Training region = rows with valid smoothed driver signal
    C = _cols(df)
    vcol = C["v"]
    train_mask = df[vcol].notna()
    df_region = df.loc[train_mask].copy()

    if df_region.empty:
        die("Training region is empty (all driver values are NaN).")

    step(f"Training samples: {len(df_region)}")

    # Build tensors (driver reads only drio internally, but V_target comes from the inversion)
    T, V_target = build_T_and_V(df_region, df, step=10)
    T = T.to(device)
    V_target = V_target.to(device)

    model = MLPDriver(
        args.driver_poly,
        args.driver_h1,
        args.driver_h2,
        args.driver_h3,
    ).to(device)

    out_dir = Path(paths["out_dir"])
    ensure_dir(out_dir)

    model_path = out_dir / "model_driver.pt"
    if model_path.exists():
        die(f"Refusing to overwrite existing checkpoint: {model_path}")

    opt = torch.optim.Adam(model.parameters(), lr=1.0 / args.lr0_inv)
    n = float(args.lr0_inv)

    metrics_path = out_dir / args.metrics_csv
    with metrics_path.open("w", newline="") as metrics_stream:
        writer = csv.writer(metrics_stream, lineterminator="\n")
        writer.writerow(
            [
                "epoch",
                "se",
                "nrmse",
                "vmin_target",
                "vmax_target",
                "vmin_pred",
                "vmax_pred",
            ]
        )

        for epoch in range(1, args.epochs + 1):
            opt.zero_grad()
            V_pred = model(T)
            loss = torch.mean((V_pred - V_target) ** 2)
            loss.backward()
            opt.step()

            if (epoch == 1) or (epoch % args.log_every == 0) or (epoch == args.epochs):
                with torch.no_grad():
                    V_logged = model(T)
                    se = float(torch.sum((V_logged - V_target) ** 2).cpu())
                    err = float(nrmse(V_target, V_logged).cpu())
                    vmin_t = float(V_target.min().cpu())
                    vmax_t = float(V_target.max().cpu())
                    vmin_p = float(V_logged.min().cpu())
                    vmax_p = float(V_logged.max().cpu())

                step(f"Epoch {epoch}/{args.epochs} | SE={se:.6e} | NRMSE={err:.6e}")
                writer.writerow([epoch, se, err, vmin_t, vmax_t, vmin_p, vmax_p])
                metrics_stream.flush()

            n += args.step_size
            opt.param_groups[0]["lr"] = 1.0 / n

    # ---- SAVE MODEL ----
    meta = {
        "model": "MLPDriver",
        "input": "drio",
        "poly_degree": args.driver_poly,
        "hidden": [args.driver_h1, args.driver_h2, args.driver_h3],
        "prepared_sha256": sha256_file(paths["prepared_csv"]),
        "git": git_provenance(),
        "runtime": runtime_provenance(device),
        "train_args": vars(args),
    }

    torch.save({"state_dict": model.state_dict(), "meta": meta}, model_path)
    step(f"Saved model: {model_path}")

    with (out_dir / "model_driver_meta.json").open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    # ---- TRAINING PREDICTIONS ----
    with torch.no_grad():
        V_pred = model(T).detach().cpu().numpy()
    V_true = V_target.detach().cpu().numpy()

    out = df_region[[C["x"], C["y"]]].copy()
    out["V_target"] = V_true
    out["V_pred"] = V_pred

    pred_csv = out_dir / "train_driver_predictions.csv"
    out.to_csv(pred_csv, index=False)
    step(f"Saved training predictions: {pred_csv}")

    step(f"Saved metrics: {metrics_path}")

    save_stage_record(
        out_dir,
        "train_driver",
        args,
        inputs={"prepared": paths["prepared_csv"]},
        outputs={
            "model": model_path,
            "model_metadata": out_dir / "model_driver_meta.json",
            "metrics": metrics_path,
            "training_predictions": pred_csv,
        },
        device=device,
    )

    step(
        f"V_target range: [{V_true.min():.3e}, {V_true.max():.3e}] | "
        f"V_pred range: [{V_pred.min():.3e}, {V_pred.max():.3e}]"
    )


if __name__ == "__main__":
    main()
