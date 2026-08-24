#!/usr/bin/env python3
"""Create post-evaluation diagnostic plots for one Method 1.5 sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from imcpe.methods.stereo_vo_anchored.diagnostics import create_sequence_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--method1-root", type=Path, required=True)
    parser.add_argument("--method1-5-root", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/method_1_5/plots"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = args.split
    sequence = args.sequence
    result = create_sequence_diagnostics(
        sequence=sequence,
        ground_truth_pose=args.data_root / split / sequence / "pose.txt",
        baseline_pose=args.baseline_root / split / sequence / "pose.txt",
        method_1_pose=args.method1_root / split / sequence / "pose.txt",
        method_1_5_pose=args.method1_5_root / split / sequence / "pose.txt",
        anchor_cache=args.anchor_root / split / f"{sequence}.npz",
        method_1_diagnostics=args.method1_root / split / sequence / "diagnostics.json",
        fusion_diagnostics=args.method1_5_root
        / split
        / sequence
        / "fusion_diagnostics.json",
        output_dir=args.output_root / split / sequence,
        thresholds_path=args.thresholds,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
