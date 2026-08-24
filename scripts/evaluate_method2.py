#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.methods.cross_stereo_pnp.evaluate import compare_four_methods
from imcpe.pose_metrics import RPE_DELTAS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Official comparison: baseline, Method 1, Method 1.5A, Method 2."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--method1-root", type=Path, required=True)
    parser.add_argument("--method1-5-root", type=Path, required=True)
    parser.add_argument("--method2-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    result = compare_four_methods(
        data_root=args.data_root,
        split=args.split,
        baseline_root=args.baseline_root,
        method1_root=args.method1_root,
        method1_5_root=args.method1_5_root,
        method2_root=args.method2_root,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    global_result = result["global"]
    print("Method | mean ATE | median ATE | std ATE")
    for name in ("baseline", "method1", "method1_5", "method2"):
        metrics = global_result[name]
        print(
            f"{name:10s} | {metrics['mean_ate']:.4f} | "
            f"{metrics['median_ate']:.4f} | {metrics['std_ate']:.4f}"
        )
        for delta in RPE_DELTAS:
            print(
                f"  RPE d={delta:2d}: "
                f"{metrics[f'trans_rpe_{delta}frame']:.4f} mm / "
                f"{metrics[f'rot_rpe_deg_{delta}frame']:.4f} deg"
            )
    diagnostics = global_result["method2_diagnostics"]
    print("METHOD 2 DIAGNOSTICS")
    print(json.dumps(diagnostics, indent=2))
    print(
        "Method 2 improved/worsened vs Method 1.5A: "
        f"{global_result['num_method2_improved_vs_method1_5']}/"
        f"{global_result['num_method2_worsened_vs_method1_5']}"
    )
    print(f"Saved: {args.json_out}")


if __name__ == "__main__":
    main()

