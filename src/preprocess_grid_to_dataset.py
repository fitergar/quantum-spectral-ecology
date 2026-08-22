"""
Preprocess Gridvaluesraw + RGB tiles into a dataset CSV usable by the other scripts.

Inputs (default in data/):
  - Gridvaluesraw.csv
  - R2GL.csv, G2GL.csv, B2GL.csv

Output:
  - data/{dataset}.csv

Adds:
  - drio  : L1 distance to nearest river cell (Esrio != 0)
  - Zdrio : Z_cell - Z_nearest_river_cell
  - R1..R25, G1..G25, B1..B25: 25 RGB subpixels per channel per grid cell
Optional:
  - Xsub1..25, Ysub1..25 (shared aligned subpixel coords per cell)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import DATA, dataset_paths, die, ensure_dir, save_stage_record, step

# -------------------- core computations --------------------


def compute_drio_and_Zdrio(
    df: pd.DataFrame,
    *,
    xcol: str = "X",
    ycol: str = "Y",
    zcol: str = "Z",
    esrio_col: str = "Esrio",
) -> pd.DataFrame:
    """
    drio  = min L1 distance to any cell with Esrio != 0 (river)
    Zdrio = Z_cell - Z_nearest_river_cell

    River cells: Esrio != 0 (NaNs treated as 0)
    For river cells: drio = 0 and Zdrio = 0.
    """
    if esrio_col not in df.columns:
        die(f"Column '{esrio_col}' not found in grid dataframe.")

    coords = df[[xcol, ycol]].to_numpy(dtype=float)
    z_all = df[zcol].to_numpy(dtype=float)

    river_mask = df[esrio_col].fillna(0) != 0
    if not river_mask.any():
        die("No river cells found (Esrio != 0). Cannot compute drio/Zdrio.")

    river_coords = coords[river_mask.values]
    river_z = z_all[river_mask.values]

    # Brute-force L1 distance to the nearest river cell.
    diff = np.abs(coords[:, None, :] - river_coords[None, :, :])  # (N, Nr, 2)
    l1 = diff.sum(axis=2)  # (N, Nr)

    drio = l1.min(axis=1)
    nearest_idx = l1.argmin(axis=1)
    nearest_z = river_z[nearest_idx]
    zdrio = z_all - nearest_z

    out = df.copy()
    out["drio"] = drio
    out["Zdrio"] = zdrio
    return out


def infer_value_column(
    df: pd.DataFrame, expected_xy: tuple[str, str] = ("X", "Y")
) -> str:
    """
    Given a colour CSV with X/Y and one value column, infer the value column.
    """
    ignore = set(expected_xy) | {"id", "ID", "fid", "FID"}
    candidates = [c for c in df.columns if c not in ignore]
    if len(candidates) != 1:
        raise ValueError(
            f"Could not infer unique value column in {list(df.columns)} "
            f"(candidates: {candidates})"
        )
    return candidates[0]


def build_grid_index(df: pd.DataFrame, *, xcol="X", ycol="Y", tol: float = 1e-6):
    """
    Normalize grid centers to integer centers via rounding (not truncation).
    Returns:
      xs_int, ys_int : sorted unique integer centers
      coord_to_idx   : dict[(Xc, Yc)] -> df.index
      cellsize       : approx spacing (for logging)
    """
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)

    x_round = np.rint(x).astype(int)
    y_round = np.rint(y).astype(int)

    if not np.allclose(x, x_round, atol=tol):
        die(f"Grid X centers are not integer-valued within tol={tol}.")
    if not np.allclose(y, y_round, atol=tol):
        die(f"Grid Y centers are not integer-valued within tol={tol}.")

    xs_int = np.sort(np.unique(x_round))
    ys_int = np.sort(np.unique(y_round))
    if len(xs_int) < 2 or len(ys_int) < 2:
        die("Not enough distinct X or Y values to define grid spacing.")

    dx = int(np.min(np.diff(xs_int)))
    dy = int(np.min(np.diff(ys_int)))
    cellsize = min(dx, dy)

    coord_to_idx: dict[tuple[int, int], int] = {}
    for idx, (xc, yc) in zip(df.index, zip(x_round, y_round)):
        coord_to_idx[(int(xc), int(yc))] = idx

    return xs_int, ys_int, coord_to_idx, cellsize


def assign_subpixels_to_grid(
    sub_df: pd.DataFrame,
    *,
    value_col: str,
    xs_int: np.ndarray,
    ys_int: np.ndarray,
    coord_to_idx: dict,
    radius: float = 5.0,
    xcol: str = "X",
    ycol: str = "Y",
    label: str = "R",
    x_offset: float = 0.47,
    y_offset: float = 0.25,
    key_precision: int = 4,
):
    """
    Returns dict:
      grid_idx -> dict(rel_key -> (value, x, y, linf))
    where rel_key is (dx,dy) rounded to align RGB channels.
    """
    bucket = defaultdict(dict)
    misses_outside = 0
    misses_no_center = 0

    xs = xs_int
    ys = ys_int

    for _, row in sub_df.iterrows():
        x = float(row[xcol]) + x_offset
        y = float(row[ycol]) + y_offset
        v = float(row[value_col])

        # Snap to nearest VALID grid center
        cx = int(xs[np.argmin(np.abs(xs - x))])
        cy = int(ys[np.argmin(np.abs(ys - y))])

        linf = max(abs(cx - x), abs(cy - y))
        if linf > radius + 1e-9:
            misses_outside += 1
            continue

        idx = coord_to_idx.get((cx, cy))
        if idx is None:
            misses_no_center += 1
            continue

        dx = x - cx
        dy = y - cy
        rel_key = (round(dx, key_precision), round(dy, key_precision))

        prev = bucket[idx].get(rel_key)
        if (prev is None) or (linf < prev[3]):
            bucket[idx][rel_key] = (v, x, y, linf)

    if misses_outside or misses_no_center:
        print(
            f"[{label}] WARNING: {misses_outside} subpixels outside radius={radius} m, "
            f"{misses_no_center} mapped to non-existing centers."
        )

    return bucket


def attach_rgb_columns(
    df: pd.DataFrame,
    *,
    R_dict: dict,
    G_dict: dict,
    B_dict: dict,
    n_per_cell: int = 25,
    save_subcoords: bool = False,
) -> tuple[pd.DataFrame, tuple[tuple[float, float], ...]]:
    """
    Add R1..Rn, G1..Gn, B1..Bn columns to df using dicts:
      grid_idx -> dict(rel_key -> (value, x, y, linf))

    Alignment: intersect relative keys so (Rk,Gk,Bk) share the same (dx,dy).
    Positions are ordered north-to-south and then west-to-east. Consequently,
    index 1 is the upper-left subcell in UTM coordinates.
    """
    out = df.copy()

    for prefix in ("R", "G", "B"):
        for k in range(1, n_per_cell + 1):
            out[f"{prefix}{k}"] = np.nan

    if save_subcoords:
        for k in range(1, n_per_cell + 1):
            out[f"Xsub{k}"] = np.nan
            out[f"Ysub{k}"] = np.nan

    reference_layout: tuple[tuple[float, float], ...] | None = None

    for idx in out.index:
        R_map = R_dict.get(idx, {})
        G_map = G_dict.get(idx, {})
        B_map = B_dict.get(idx, {})

        common = set(R_map) & set(G_map) & set(B_map)
        # dx increases west-to-east; dy increases south-to-north.
        keys_take = tuple(sorted(common, key=lambda key: (-key[1], key[0])))

        if len(keys_take) != n_per_cell:
            raise ValueError(
                f"Grid row {idx} has {len(keys_take)} aligned RGB positions; "
                f"expected exactly {n_per_cell}."
            )
        if reference_layout is None:
            reference_layout = keys_take
        elif keys_take != reference_layout:
            raise ValueError(
                f"Grid row {idx} does not have the same relative RGB layout "
                "as the first grid cell."
            )

        for k, key in enumerate(keys_take, start=1):
            rv, rx, ry, _ = R_map[key]
            gv, _, _, _ = G_map[key]
            bv, _, _, _ = B_map[key]

            out.at[idx, f"R{k}"] = rv
            out.at[idx, f"G{k}"] = gv
            out.at[idx, f"B{k}"] = bv

            if save_subcoords:
                out.at[idx, f"Xsub{k}"] = rx
                out.at[idx, f"Ysub{k}"] = ry

    if reference_layout is None:
        raise ValueError("The grid contains no RGB subcell layout.")

    print(
        f"\n[RGB] OK: all {len(out)} cells share one explicit "
        f"{n_per_cell}-position layout."
    )
    return out, reference_layout


def layout_table(layout: tuple[tuple[float, float], ...]) -> pd.DataFrame:
    """Return the spatial meaning of each R/G/B/Grey feature index."""
    xs = sorted({dx for dx, _ in layout})
    ys = sorted({dy for _, dy in layout}, reverse=True)
    x_to_column = {x: column for column, x in enumerate(xs, start=1)}
    y_to_row = {y: row for row, y in enumerate(ys, start=1)}
    return pd.DataFrame(
        [
            {
                "feature_index": index,
                "grey_feature": f"Grey{index}",
                "rgb_features": f"R{index}/G{index}/B{index}",
                "row_north_to_south": y_to_row[dy],
                "column_west_to_east": x_to_column[dx],
                "dx_meters": dx,
                "dy_meters": dy,
            }
            for index, (dx, dy) in enumerate(layout, start=1)
        ]
    )


# -------------------- CLI --------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess grid + RGB tiles into a dataset CSV with drio/Zdrio and aligned RGB subpixels."
    )

    p.add_argument(
        "--dataset", required=True, help="Output dataset name (e.g. 'Test_4')."
    )

    p.add_argument("--grid-csv", default="Gridvaluesraw.csv")
    p.add_argument("--r-csv", default="R2GL.csv")
    p.add_argument("--g-csv", default="G2GL.csv")
    p.add_argument("--b-csv", default="B2GL.csv")

    p.add_argument("--radius", type=float, default=5.0)
    p.add_argument("--n-per-cell", type=int, default=25)
    p.add_argument("--esrio-col", type=str, default="Esrio")

    p.add_argument("--save-subcoords", action="store_true")

    # Offsets as args (reproducible + dataset-dependent)
    p.add_argument("--x-offset", type=float, default=0.47)
    p.add_argument("--y-offset", type=float, default=0.25)
    p.add_argument("--key-precision", type=int, default=4)

    return p.parse_args()


def resolve_in_data(name_or_path: str) -> Path:
    p = Path(name_or_path)
    return p if p.is_absolute() else (DATA / p)


def main(args: argparse.Namespace) -> None:
    grid_path = resolve_in_data(args.grid_csv)
    r_path = resolve_in_data(args.r_csv)
    g_path = resolve_in_data(args.g_csv)
    b_path = resolve_in_data(args.b_csv)

    for pth, label in [
        (grid_path, "Grid"),
        (r_path, "R"),
        (g_path, "G"),
        (b_path, "B"),
    ]:
        if not pth.exists():
            die(f"{label} CSV not found: {pth}")

    paths = dataset_paths(args.dataset)
    out_csv = paths["raw_csv"]
    ensure_dir(out_csv.parent)
    out_dir = ensure_dir(paths["out_dir"])

    step(f"Output dataset '{args.dataset}' -> {out_csv}")

    step(f"Reading grid: {grid_path}")
    grid = pd.read_csv(grid_path)

    step("Computing drio and Zdrio...")
    grid = compute_drio_and_Zdrio(grid, esrio_col=args.esrio_col)

    step("Building grid index...")
    xs_int, ys_int, coord_to_idx, cellsize = build_grid_index(grid)
    step(f"Grid spacing ~ {cellsize} m; RGB L∞ radius={args.radius} m")
    step(
        f"Offsets: x_offset={args.x_offset}, y_offset={args.y_offset}, key_precision={args.key_precision}"
    )

    step("Reading RGB tiles...")
    R_df = pd.read_csv(r_path)
    G_df = pd.read_csv(g_path)
    B_df = pd.read_csv(b_path)

    R_val = infer_value_column(R_df)
    G_val = infer_value_column(G_df)
    B_val = infer_value_column(B_df)
    step(f"Value columns: R='{R_val}', G='{G_val}', B='{B_val}'")

    step("Assigning subpixels to grid cells...")
    R_dict = assign_subpixels_to_grid(
        R_df,
        value_col=R_val,
        xs_int=xs_int,
        ys_int=ys_int,
        coord_to_idx=coord_to_idx,
        radius=args.radius,
        label="R",
        x_offset=args.x_offset,
        y_offset=args.y_offset,
        key_precision=args.key_precision,
    )
    G_dict = assign_subpixels_to_grid(
        G_df,
        value_col=G_val,
        xs_int=xs_int,
        ys_int=ys_int,
        coord_to_idx=coord_to_idx,
        radius=args.radius,
        label="G",
        x_offset=args.x_offset,
        y_offset=args.y_offset,
        key_precision=args.key_precision,
    )
    B_dict = assign_subpixels_to_grid(
        B_df,
        value_col=B_val,
        xs_int=xs_int,
        ys_int=ys_int,
        coord_to_idx=coord_to_idx,
        radius=args.radius,
        label="B",
        x_offset=args.x_offset,
        y_offset=args.y_offset,
        key_precision=args.key_precision,
    )

    step("Attaching RGB columns...")
    out_df, layout = attach_rgb_columns(
        grid,
        R_dict=R_dict,
        G_dict=G_dict,
        B_dict=B_dict,
        n_per_cell=args.n_per_cell,
        save_subcoords=args.save_subcoords,
    )

    step(f"Writing dataset CSV: {out_csv}")
    out_df.to_csv(out_csv, index=False)
    layout_path = out_dir / "subcell_layout.csv"
    layout_table(layout).to_csv(layout_path, index=False)
    step(f"Writing subcell layout: {layout_path}")
    save_stage_record(
        out_dir,
        "preprocess_grid_to_dataset",
        args,
        inputs={"grid": grid_path, "red": r_path, "green": g_path, "blue": b_path},
        outputs={"dataset": out_csv, "subcell_layout": layout_path},
    )
    step("Done.")


if __name__ == "__main__":
    main(parse_args())
