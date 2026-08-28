#!/usr/bin/env python3
"""Verify imports and optional local model artifacts without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "cv2",
    "einops",
    "joblib",
    "kornia",
    "lightglue",
    "loguru",
    "matplotlib",
    "numpy",
    "scipy",
    "torch",
    "tqdm",
    "yaml",
    "yacs",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-xfeat", action="store_true")
    parser.add_argument("--require-loftr", action="store_true")
    parser.add_argument("--require-roma", action="store_true")
    parser.add_argument("--require-vggt", action="store_true")
    args = parser.parse_args()

    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(f"Missing Python modules: {', '.join(missing)}")

    expected = {
        "accelerated_features": (
            "e92685f57f8318b18725c5c8c0bd28c7fe188d9a",
            "weights/xfeat.pt",
            "0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b",
            args.require_xfeat,
        ),
        "EfficientLoFTR": (
            "ffd4a4644064354468eb1f0c7a3e732233cb732f",
            "weights/eloftr_outdoor.ckpt",
            "0af6291141c736e75e94b7f8aae4399b77c7731b3f08758212b2cfe370188878",
            args.require_loftr,
        ),
        "RoMa": (
            "77f8d68803526dcddfd9b7a46bc76125bdc25f15",
            ".torch/hub/checkpoints/roma_outdoor.pth",
            "c7a45c80d41ad788a63c641d1b686d7cb3f297f40097c6f4e75039889e5cc8ba",
            args.require_roma,
        ),
        "vggt": (
            "a288dd0f14786c93483e45524328726ab7b1b4ce",
            None,
            None,
            args.require_vggt,
        ),
    }
    failures: list[str] = []
    for directory, (revision, relative_weight, weight_hash, required) in expected.items():
        checkout = ROOT / "third_party" / directory
        actual = git_revision(checkout)
        if actual is None:
            print(f"optional: {directory} checkout not installed")
            if required:
                failures.append(f"missing {directory} checkout")
            continue
        if actual != revision:
            failures.append(f"{directory}: expected {revision}, found {actual}")
        if relative_weight is not None:
            weight = checkout / relative_weight
            if not weight.is_file():
                failures.append(f"missing {weight}")
            elif sha256(weight) != weight_hash:
                failures.append(f"checksum mismatch: {weight}")
        print(f"found: {directory} @ {actual}")

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Python {sys.version.split()[0]}: core environment OK")


if __name__ == "__main__":
    main()
