#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.methods.hybrid_pose_fusion.evaluate import compare_method4
from imcpe.pose_metrics import RPE_DELTAS


def main() -> None:
    parser = argparse.ArgumentParser(description="Official evaluation including Method 4A.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--baseline-root", type=Path, default=None)
    parser.add_argument("--method1-root", type=Path, default=None)
    parser.add_argument("--method1-5-root", type=Path, default=None)
    parser.add_argument("--method2a-root", type=Path, default=None)
    parser.add_argument("--method2b-root", type=Path, default=None)
    parser.add_argument("--method4a0-root", type=Path, default=None)
    parser.add_argument("--method4a1-root", type=Path, default=None)
    parser.add_argument("--sequence", action="append", default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    roots = {
        name: root for name, root in {
            "baseline": args.baseline_root,
            "method1": args.method1_root,
            "method1_5": args.method1_5_root,
            "method2a": args.method2a_root,
            "method2b": args.method2b_root,
            "method4a0": args.method4a0_root,
            "method4a1": args.method4a1_root,
        }.items() if root is not None
    }
    result = compare_method4(
        data_root=args.data_root,
        split=args.split,
        prediction_roots=roots,
        selected_sequences=args.sequence,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print("Method | mean ATE | median ATE | std ATE")
    for name, metrics in result["global"].items():
        print(
            f"{name:10s} | {metrics.get('mean_ate', float('nan')):.4f} | "
            f"{metrics.get('median_ate', float('nan')):.4f} | "
            f"{metrics.get('std_ate', float('nan')):.4f}"
        )
        for delta in RPE_DELTAS:
            print(
                f"  RPE d={delta:2d}: "
                f"{metrics.get(f'trans_rpe_{delta}frame', float('nan')):.4f} mm / "
                f"{metrics.get(f'rot_rpe_deg_{delta}frame', float('nan')):.4f} deg"
            )
    print(f"Saved: {args.json_out}")


if __name__ == "__main__":
    main()
