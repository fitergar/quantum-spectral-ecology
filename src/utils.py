"""Shared paths, diagnostics, and reproducibility helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUTPUTS = REPO / "outputs"


def dataset_paths(dataset: str) -> dict[str, Path]:
    """Return the canonical input and output paths for a dataset name."""
    raw_csv = DATA / f"{dataset}.csv"
    out_dir = OUTPUTS / dataset
    return {
        "raw_csv": raw_csv,
        "out_dir": out_dir,
        "prepared_csv": out_dir / "prepared.csv",
    }


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def step(msg: str) -> None:
    """Print a progress message."""
    print(f"• {msg}")


def die(msg: str, code: int = 1) -> None:
    """Print an error and stop execution."""
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance() -> dict:
    """Describe the repository revision used for a pipeline stage."""

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(REPO), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain", "--untracked-files=no")
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "tracked_files_dirty": None}
    return {"commit": commit, "tracked_files_dirty": bool(status)}


def runtime_provenance(device=None) -> dict:
    """Capture the software and hardware information relevant to a run."""
    packages = {}
    for name in ("numpy", "pandas", "scipy", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
    if device is not None:
        import torch

        payload.update(
            {
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "torch_cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            }
        )
        if device.type == "cuda":
            payload["gpu_name"] = torch.cuda.get_device_name(device)
    return payload


def configure_torch(seed: int, *, cuda: bool):
    """Seed PyTorch and select a strict CPU or CUDA device."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch

    if cuda and not torch.cuda.is_available():
        die("CUDA was requested, but torch.cuda.is_available() is false.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device("cuda" if cuda else "cpu")


def select_torch_device(*, cuda: bool):
    """Select a device and fail rather than silently ignoring --cuda."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch

    if cuda and not torch.cuda.is_available():
        die("CUDA was requested, but torch.cuda.is_available() is false.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device("cuda" if cuda else "cpu")


def _file_record(path: Path) -> dict:
    path = path.resolve()
    try:
        display_path = str(path.relative_to(REPO))
    except ValueError:
        display_path = str(path)
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def save_stage_record(
    out_dir: Path,
    stage: str,
    args,
    *,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
    device=None,
) -> Path:
    """Write a completed-stage record with arguments, provenance, and hashes."""
    ensure_dir(out_dir)
    payload = {
        "stage": stage,
        "status": "completed",
        "completed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "argv": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "args": vars(args),
        "git": git_provenance(),
        "runtime": runtime_provenance(device),
        "inputs": {name: _file_record(path) for name, path in inputs.items()},
        "outputs": {name: _file_record(path) for name, path in outputs.items()},
    }
    record_path = out_dir / f"args_{stage}.json"
    record_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return record_path
