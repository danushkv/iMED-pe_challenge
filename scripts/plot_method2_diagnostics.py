#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.methods.cross_stereo_pnp.diagnostics import create_method2_sequence_plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--method1-root", type=Path, required=True)
    parser.add_argument("--method1-5-root", type=Path, required=True)
    parser.add_argument("--method2-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "method_2" / "plots",
    )
    args = parser.parse_args()
    sequence = args.sequence
    split = args.split
    result = create_method2_sequence_plots(
        sequence_name=sequence,
        ground_truth_pose=args.data_root / split / sequence / "pose.txt",
        baseline_pose=args.baseline_root / split / sequence / "pose.txt",
        method1_pose=args.method1_root / split / sequence / "pose.txt",
        method1_5_pose=args.method1_5_root / split / sequence / "pose.txt",
        method2_pose=args.method2_root / split / sequence / "pose.txt",
        method2_diagnostics=args.method2_root
        / split
        / sequence
        / "diagnostics.json",
        output_dir=args.output_root / split / sequence,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

