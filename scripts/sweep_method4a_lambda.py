#!/usr/bin/env python3
"""Small TRAIN-only lambda sweep; fusion subprocess never receives data-root/GT."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.methods.hybrid_pose_fusion.evaluate import compare_method4


def tag(value: float) -> str:
    return f"lambda_{value:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal TRAIN-only Method 4A lambda sweep.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--method1-root", type=Path, required=True)
    parser.add_argument("--method2a-root", type=Path, required=True)
    parser.add_argument("--method2b-root", type=Path, required=True)
    parser.add_argument("--rotation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 0.5, 1, 2, 5])
    parser.add_argument("--confidence-version", choices=["v0", "v1"], default="v1")
    args = parser.parse_args()
    results: list[dict[str, object]] = []
    for lambda_vo in args.lambdas:
        run_root = args.output_root / tag(lambda_vo)
        command = [
            sys.executable, str(ROOT / "scripts" / "run_method4a.py"),
            "--split", "train",
            "--method1-root", str(args.method1_root),
            "--method2a-root", str(args.method2a_root),
            "--method2b-root", str(args.method2b_root),
            "--rotation-root", str(args.rotation_root),
            "--output-root", str(run_root),
            "--mode", "4a1",
            "--confidence-version", args.confidence_version,
            "--lambda-vo", str(lambda_vo),
        ]
        for sequence in args.sequence:
            command.extend(["--sequence", sequence])
        subprocess.run(command, check=True)
        comparison = compare_method4(
            data_root=args.data_root,
            split="train",
            prediction_roots={"method4a1": run_root},
            selected_sequences=args.sequence,
        )
        metrics = comparison["global"]["method4a1"]
        record = {
            "lambda_vo": lambda_vo,
            "mean_ate": metrics["mean_ate"],
            "median_ate": metrics["median_ate"],
            "std_ate": metrics["std_ate"],
            "run_root": str(run_root),
        }
        results.append(record)
        print(
            f"lambda={lambda_vo:g}: ATE={metrics['mean_ate']:.4f} "
            f"median={metrics['median_ate']:.4f}"
        )
    best = min(results, key=lambda item: float(item["mean_ate"]))
    payload = {
        "selection_split": "train",
        "sequences": args.sequence,
        "confidence_version": args.confidence_version,
        "results": results,
        "best": best,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "grid_results.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    config_path = args.output_root / "best_config.json"
    config_path.write_text(json.dumps({
        "lambda_vo": best["lambda_vo"],
        "confidence_version": args.confidence_version,
        "selected_on": "train",
        "sequences": args.sequence,
    }, indent=2) + "\n")
    print(f"Best TRAIN lambda: {best['lambda_vo']}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
