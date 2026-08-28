#!/usr/bin/env python3
"""Official-metric comparison of completed RoMa-2A prediction variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.pose_metrics import RPE_DELTAS, evaluate_split


def _json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def parse_named_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Expected non-empty NAME=PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare completed variants using the repository evaluator."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument(
        "--prediction",
        type=parse_named_root,
        action="append",
        required=True,
        help="NAME=ROOT; repeat for each prediction root",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "split": args.split,
        "data_root": str(args.data_root),
        "methods": {},
    }
    for name, prediction_root in args.prediction:
        per_sequence, global_metrics = evaluate_split(
            args.data_root, prediction_root, args.split
        )
        report["methods"][name] = {
            "prediction_root": str(prediction_root),
            "global": global_metrics,
            "per_sequence": per_sequence,
        }

    print("Method | mean ATE | median ATE | std ATE")
    for name, payload in report["methods"].items():
        metrics = payload["global"]
        print(
            f"{name:18s} | {metrics.get('mean_ate', float('nan')):.4f} | "
            f"{metrics.get('median_ate', float('nan')):.4f} | "
            f"{metrics.get('std_ate', float('nan')):.4f}"
        )
        for delta in RPE_DELTAS:
            print(
                f"  RPE d={delta:2d}: "
                f"{metrics.get(f'trans_rpe_{delta}frame', float('nan')):.4f} mm / "
                f"{metrics.get(f'rot_rpe_deg_{delta}frame', float('nan')):.4f} deg"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, default=_json_default, allow_nan=True) + "\n"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
