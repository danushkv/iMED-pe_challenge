#!/usr/bin/env python3
"""Verify optimized symmetric reuse against legacy RoMa cycle inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.io_pose import read_pose_txt


def _summary(root: Path, split: str) -> dict:
    path = root / f"run_summary_{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing run summary: {path}")
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--optimized-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    legacy_summary = _summary(args.legacy_root, args.split)
    optimized_summary = _summary(args.optimized_root, args.split)
    if legacy_summary.get("symmetric_reverse_reuse") is not False:
        raise RuntimeError("Legacy run did not disable symmetric reverse reuse")
    if optimized_summary.get("symmetric_reverse_reuse") is not True:
        raise RuntimeError("Optimized run did not enable symmetric reverse reuse")

    sequence = args.sequence
    legacy_rows = read_pose_txt(
        args.legacy_root / args.split / sequence / "pose.txt"
    )
    optimized_rows = read_pose_txt(
        args.optimized_root / args.split / sequence / "pose.txt"
    )
    legacy_ids = [row.frame_idx for row in legacy_rows]
    optimized_ids = [row.frame_idx for row in optimized_rows]
    if legacy_ids != optimized_ids:
        raise RuntimeError("Legacy and optimized frame IDs differ")

    translation_error = float(
        np.max(
            np.abs(
                np.stack([row.t for row in legacy_rows])
                - np.stack([row.t for row in optimized_rows])
            )
        )
    )
    quaternion_error = float(
        np.max(
            np.abs(
                np.stack([row.q_xyzw for row in legacy_rows])
                - np.stack([row.q_xyzw for row in optimized_rows])
            )
        )
    )
    maximum_error = max(translation_error, quaternion_error)

    legacy_sequence = legacy_summary["sequences"][sequence]
    optimized_sequence = optimized_summary["sequences"][sequence]
    legacy_seconds = float(legacy_sequence["total_runtime_seconds"])
    optimized_seconds = float(optimized_sequence["total_runtime_seconds"])
    speedup = legacy_seconds / optimized_seconds if optimized_seconds > 0 else np.nan

    result = {
        "passed": bool(maximum_error <= args.tolerance),
        "tolerance": args.tolerance,
        "sequence": sequence,
        "num_frames": len(legacy_rows),
        "maximum_translation_component_error": translation_error,
        "maximum_quaternion_component_error": quaternion_error,
        "maximum_pose_component_error": maximum_error,
        "legacy_total_runtime_seconds": legacy_seconds,
        "optimized_total_runtime_seconds": optimized_seconds,
        "measured_sequence_speedup": float(speedup),
        "legacy_median_reverse_seconds_per_pair": {
            "stereo": legacy_sequence.get("median_stereo_reverse_seconds"),
            "cross": legacy_sequence.get("median_cross_reverse_seconds"),
        },
        "optimized_median_reverse_seconds_per_pair": {
            "stereo": optimized_sequence.get("median_stereo_reverse_seconds"),
            "cross": optimized_sequence.get("median_cross_reverse_seconds"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps(result, indent=2, allow_nan=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
