#!/usr/bin/env python3
"""Official metrics plus aligned per-frame p95 error for saved Method 4C variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.alignment import horn_align_sim3
from imcpe.io_pose import read_pose_txt
from imcpe.pose_metrics import RPE_DELTAS, build_correspondence, evaluate_split

from .load_observers import parse_named_root


def _aggregate(per_sequence: dict[str, dict[str, float | int]]) -> dict[str, float]:
    """Aggregate selected sequences exactly like the repository evaluator."""
    valid = [
        result
        for result in per_sequence.values()
        if result["num_matched_poses"] > 0
    ]
    if not valid:
        return {
            key: float("nan")
            for key in ("mean_ate", "std_ate", "median_ate", "num_sequences")
        }
    aggregate = {
        "mean_ate": float(np.mean([result["mean_ate"] for result in valid])),
        "std_ate": float(np.mean([result["std_ate"] for result in valid])),
        "median_ate": float(np.mean([result["median_ate"] for result in valid])),
        "num_sequences": float(len(valid)),
    }
    for delta in RPE_DELTAS:
        aggregate[f"trans_rpe_{delta}frame"] = float(
            np.mean([result[f"trans_rpe_{delta}frame"] for result in valid])
        )
        aggregate[f"rot_rpe_deg_{delta}frame"] = float(
            np.mean([result[f"rot_rpe_deg_{delta}frame"] for result in valid])
        )
    aggregate["mean_trans_rpe_40frame"] = float(
        np.mean([result["mean_trans_rpe_40frame"] for result in valid])
    )
    aggregate["mean_rot_rpe_deg_40frame"] = float(
        np.mean([result["mean_rot_rpe_deg_40frame"] for result in valid])
    )
    return aggregate


def _aligned_errors(gt_path: Path, prediction_path: Path) -> np.ndarray:
    gt = read_pose_txt(gt_path)
    prediction = read_pose_txt(prediction_path)
    model, data, *_ = build_correspondence(gt, prediction)
    _, _, errors, _ = horn_align_sim3(model, data)
    return np.asarray(errors, dtype=np.float64)


def _json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument(
        "--prediction", type=parse_named_root, action="append", required=True
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="Restrict evaluation to this sequence; repeat as needed",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    methods: dict[str, Any] = {}
    for name, root in args.prediction:
        per_sequence, global_metrics = evaluate_split(args.data_root, root, args.split)
        if args.sequence:
            unknown = sorted(set(args.sequence) - set(per_sequence))
            if unknown:
                raise ValueError(f"Unknown {args.split} sequences: {unknown}")
            per_sequence = {
                sequence: per_sequence[sequence] for sequence in args.sequence
            }
            global_metrics = _aggregate(per_sequence)
        all_errors: list[np.ndarray] = []
        tail_by_sequence: dict[str, dict[str, float]] = {}
        for sequence in sorted(per_sequence):
            prediction_path = root / args.split / sequence / "pose.txt"
            if per_sequence[sequence]["num_matched_poses"] <= 0:
                continue
            errors = _aligned_errors(
                args.data_root / args.split / sequence / "pose.txt",
                prediction_path,
            )
            all_errors.append(errors)
            tail_by_sequence[sequence] = {
                "p95_aligned_position_error_mm": float(np.percentile(errors, 95)),
                "maximum_aligned_position_error_mm": float(np.max(errors)),
            }
        if not all_errors:
            raise RuntimeError(f"No valid predictions found for {name} under {root}")
        concatenated = np.concatenate(all_errors)
        methods[name] = {
            "prediction_root": str(root),
            "global": {
                **global_metrics,
                "p95_aligned_position_error_mm": float(
                    np.percentile(concatenated, 95)
                ),
                "maximum_aligned_position_error_mm": float(np.max(concatenated)),
            },
            "per_sequence": per_sequence,
            "tail_error_per_sequence": tail_by_sequence,
        }

    print("Method | mean ATE | session-unaware p95 | median ATE | std ATE")
    for name, payload in methods.items():
        metrics = payload["global"]
        print(
            f"{name:18s} | {metrics['mean_ate']:.4f} | "
            f"{metrics['p95_aligned_position_error_mm']:.4f} | "
            f"{metrics['median_ate']:.4f} | {metrics['std_ate']:.4f}"
        )
        for delta in RPE_DELTAS:
            print(
                f"  RPE d={delta:2d}: "
                f"{metrics[f'trans_rpe_{delta}frame']:.4f} mm / "
                f"{metrics[f'rot_rpe_deg_{delta}frame']:.4f} deg"
            )
    report = {
        "split": args.split,
        "data_root": str(args.data_root),
        "selected_sequences": args.sequence,
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, default=_json_default, allow_nan=True) + "\n"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
