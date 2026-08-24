#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.methods.cross_stereo_pnp_reverse.evaluate import compare_five_methods
from imcpe.pose_metrics import RPE_DELTAS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Official comparison: baseline, M1, M1.5A, M2A, and M2B."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--method1-root", type=Path, required=True)
    parser.add_argument("--method1-5-root", type=Path, required=True)
    parser.add_argument("--method2a-root", type=Path, required=True)
    parser.add_argument("--method2b-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    result = compare_five_methods(
        data_root=args.data_root,
        split=args.split,
        baseline_root=args.baseline_root,
        method1_root=args.method1_root,
        method1_5_root=args.method1_5_root,
        method2a_root=args.method2a_root,
        method2b_root=args.method2b_root,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    global_result = result["global"]
    print("Method | mean ATE | median ATE | std ATE")
    for name in ("baseline", "method1", "method1_5", "method2a", "method2b"):
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
    print("METHOD 2B DIAGNOSTICS")
    print(json.dumps(global_result["method2b_diagnostics"], indent=2))
    print(
        "Method 2B improved/worsened vs Method 2A: "
        f"{global_result['num_method2b_improved_vs_method2a']}/"
        f"{global_result['num_method2b_worsened_vs_method2a']}"
    )
    print(f"Saved: {args.json_out}")


if __name__ == "__main__":
    main()
