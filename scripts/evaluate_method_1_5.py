from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.methods.stereo_vo_anchored.evaluate import compare_three_methods
from imcpe.pose_metrics import RPE_DELTAS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Official-metric comparison: baseline vs Method 1 vs Method 1.5."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--method1-root", type=Path, required=True)
    parser.add_argument("--method1-5-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    result = compare_three_methods(
        args.data_root,
        args.split,
        args.baseline_root,
        args.method1_root,
        args.method1_5_root,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")

    global_metrics = result["global"]
    print("Method | mean ATE | median ATE | std ATE")
    for name in ("baseline", "method1", "method1_5"):
        metrics = global_metrics[name]
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
    print(
        f"Improved/worsened vs Method 1: "
        f"{global_metrics['num_improved_vs_method1']}/"
        f"{global_metrics['num_worsened_vs_method1']}"
    )
    print(f"Saved: {args.json_out}")


if __name__ == "__main__":
    main()

