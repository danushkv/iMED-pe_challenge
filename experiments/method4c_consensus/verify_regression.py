#!/usr/bin/env python3
"""Verify two-observer Method 4C reproduces frozen Method 4A poses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.io_pose import read_pose_txt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    maximum = 0.0
    sequences: dict[str, object] = {}
    for sequence in args.sequence:
        reference = read_pose_txt(
            args.reference_root / args.split / sequence / "pose.txt"
        )
        candidate = read_pose_txt(
            args.candidate_root / args.split / sequence / "pose.txt"
        )
        if [row.frame_idx for row in reference] != [row.frame_idx for row in candidate]:
            raise RuntimeError(f"Frame IDs differ for {sequence}")
        translation = float(
            np.max(
                np.abs(
                    np.stack([row.t for row in reference])
                    - np.stack([row.t for row in candidate])
                )
            )
        )
        quaternion = float(
            np.max(
                np.abs(
                    np.stack([row.q_xyzw for row in reference])
                    - np.stack([row.q_xyzw for row in candidate])
                )
            )
        )
        maximum = max(maximum, translation, quaternion)
        sequences[sequence] = {
            "maximum_translation_component_error": translation,
            "maximum_quaternion_component_error": quaternion,
        }
    result = {
        "passed": maximum <= args.tolerance,
        "tolerance": args.tolerance,
        "maximum_pose_component_error": maximum,
        "sequences": sequences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
