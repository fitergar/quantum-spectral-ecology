# src/train_grey.py
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import torch

from src.featurize import _resolve_cols as _cols
from src.featurize import build_T_and_V
from src.models import MLPDriver, MLPModulatedGrey25Local
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
        description="Train local grey modulator (Grey1..25 + Z) with frozen driver"
    )

    p.add_argument("--dataset", required=True)

    p.add_argument("--mod-poly", type=int, default=1)
    p.add_argument("--mod-h1", type=int, default=300)
    p.add_argument("--mod-h2", type=int, default=400)
    p.add_argument("--mod-h3", type=int, default=650)

    p.add_argument(
        "--d0", type=float, default=30.0, help="Local influence distance (meters)"
    )
    p.add_argument("--k", type=float, default=0.5, help="Mask sharpness (per meter)")

    p.add_argument("--a", type=float, default=0.07)
    p.add_argument("--b", type=float, default=5.0)

    p.add_argument("--epochs", type=int, default=100_000)
    p.add_argument("--lr0-inv", type=float, default=100.0)
    p.add_argument("--step-size", type=float, default=1)

    p.add_argument("--log-every", type=int, default=5000)
    p.add_argument("--metrics-csv", default="train_grey_metrics.csv")

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

    # Training region = rows with valid smoothed signal
    C = _cols(df)
    vcol = C["v"]
    train_mask = df[vcol].notna()
    df_region = df.loc[train_mask].copy()
    if df_region.empty:
        die("Training region is empty.")
    step(f"Training samples: {len(df_region)}")

    # Build tensors (Grey25 + Z + drio are already in T by convention)
    T, V_target = build_T_and_V(df_region, df, step=10)
    T = T.to(device)
    V_target = V_target.to(device)

    # -------- Load frozen driver --------
    out_dir = Path(paths["out_dir"])
    ensure_dir(out_dir)

    driver_path = out_dir / "model_driver.pt"
    if not driver_path.exists():
        die("Driver model not found. Train driver first.")

    state = torch.load(driver_path, map_location=device, weights_only=True)
    driver_meta = state["meta"]
    prepared_sha256 = sha256_file(paths["prepared_csv"])
    if driver_meta.get("prepared_sha256") != prepared_sha256:
        die("Driver checkpoint was not trained from the current prepared.csv.")

    driver = MLPDriver(
        driver_meta["poly_degree"],
        *driver_meta["hidden"],
    ).to(device)
    driver.load_state_dict(state["state_dict"])
    driver.eval()
    for p in driver.parameters():
        p.requires_grad = False

    # -------- Normalize d0, k into T-space --------
    drio_max = float(df[C["drio"]].max())
    if drio_max <= 0:
        die("Invalid drio_max.")

    d0_T = args.d0 / drio_max
    k_T = args.k * drio_max

    # drio is the sixth column in the documented feature tensor.
    near_mask = T[:, 5] <= d0_T
    step(f"Near-river mask: {int(near_mask.sum())}/{len(T)} samples")

    # -------- Build grey modulator --------
    model = MLPModulatedGrey25Local(
        args.mod_poly,
        args.mod_h1,
        args.mod_h2,
        args.mod_h3,
        driver,
        a=args.a,
        b=args.b,
        d0=d0_T,
        k=k_T,
        ngrey=25,
        grey_start=6,
        use_z=True,
        z_index=4,
    ).to(device)

    model_path = out_dir / "model_grey.pt"
    if model_path.exists():
        die(f"Refusing to overwrite existing checkpoint: {model_path}")

    # -------- Optimize --------
    opt = torch.optim.Adam(model.parameters(), lr=1.0 / args.lr0_inv)
    n = float(args.lr0_inv)

    metrics_path = out_dir / args.metrics_csv
    with metrics_path.open("w", newline="") as metrics_stream:
        writer = csv.writer(metrics_stream, lineterminator="\n")
        writer.writerow(["epoch", "nrmse_near", "nrmse_all"])

        for epoch in range(1, args.epochs + 1):
            opt.zero_grad()
            V_pred = model(T)
            loss = torch.mean((V_pred - V_target) ** 2)
            loss.backward()
            opt.step()

            if (epoch == 1) or (epoch % args.log_every == 0) or (epoch == args.epochs):
                with torch.no_grad():
                    V_logged = model(T)
                    nrmse_near = float(
                        nrmse(V_target[near_mask], V_logged[near_mask]).cpu()
                    )
                    nrmse_all = float(nrmse(V_target, V_logged).cpu())

                step(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"NRMSE(d<=d0)={nrmse_near:.6e} | NRMSE(all)={nrmse_all:.6e}"
                )
                writer.writerow([epoch, nrmse_near, nrmse_all])
                metrics_stream.flush()

            n += args.step_size
            opt.param_groups[0]["lr"] = 1.0 / n

    # -------- Save --------
    meta = {
        "model": "MLPModulatedGrey25Local",
        "grey_features": "Grey1..Grey25",
        "uses_Z": True,
        "a": args.a,
        "b": args.b,
        "d0_meters": args.d0,
        "k_per_meter": args.k,
        "prepared_sha256": prepared_sha256,
        "driver_checkpoint_sha256": sha256_file(driver_path),
        "git": git_provenance(),
        "runtime": runtime_provenance(device),
        "driver_meta": driver_meta,
        "train_args": vars(args),
    }

    torch.save({"state_dict": model.state_dict(), "meta": meta}, model_path)
    step(f"Saved model: {model_path}")

    with (out_dir / "model_grey_meta.json").open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    # -------- Training predictions --------
    with torch.no_grad():
        V_pred = model(T).detach().cpu().numpy()

    out = df_region[[C["x"], C["y"]]].copy()
    out["V_target"] = V_target.detach().cpu().numpy()
    out["V_pred"] = V_pred

    pred_csv = out_dir / "train_grey_predictions.csv"
    out.to_csv(pred_csv, index=False)

    step(f"Saved metrics: {metrics_path}")
    step(f"Saved training predictions: {pred_csv}")

    save_stage_record(
        out_dir,
        "train_grey",
        args,
        inputs={"prepared": paths["prepared_csv"], "driver_model": driver_path},
        outputs={
            "model": model_path,
            "model_metadata": out_dir / "model_grey_meta.json",
            "metrics": metrics_path,
            "training_predictions": pred_csv,
        },
        device=device,
    )


if __name__ == "__main__":
    main()
