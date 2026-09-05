#!/usr/bin/env python3
"""TRAIN-only post-inference observer error correlation and recoverability."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.alignment import horn_align_sim3
from imcpe.io_pose import read_pose_txt
from imcpe.pose_metrics import build_correspondence

from .load_observers import parse_named_root


def aligned_error_by_frame(gt_path: Path, prediction_path: Path) -> dict[int, float]:
    gt = read_pose_txt(gt_path)
    prediction = read_pose_txt(prediction_path)
    model, data, *_, matched_ids = build_correspondence(gt, prediction)
    _, _, errors, _ = horn_align_sim3(model, data)
    return {int(frame_id): float(error) for frame_id, error in zip(matched_ids, errors)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument(
        "--observer", type=parse_named_root, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    observers = dict(args.observer)
    if len(observers) != len(args.observer):
        raise ValueError("Observer names must be unique")

    errors_by_key: dict[str, dict[tuple[str, int], float]] = {
        name: {} for name in observers
    }
    sequences = sorted(
        path.name
        for path in (args.data_root / args.split).iterdir()
        if path.is_dir()
    )
    for sequence in sequences:
        gt = args.data_root / args.split / sequence / "pose.txt"
        for name, root in observers.items():
            values = aligned_error_by_frame(
                gt, root / args.split / sequence / "pose.txt"
            )
            errors_by_key[name].update(
                {(sequence, frame_id): error for frame_id, error in values.items()}
            )

    names = tuple(observers)
    common = sorted(set.intersection(*(set(errors_by_key[name]) for name in names)))
    matrix = np.stack(
        [[errors_by_key[name][key] for key in common] for name in names]
    )
    correlation = np.corrcoef(matrix)
    thresholds = np.percentile(matrix, 90, axis=1)
    recoverability: dict[str, object] = {}
    for index, name in enumerate(names):
        worst = matrix[index] >= thresholds[index]
        others = np.delete(matrix, index, axis=0)
        other_median = np.median(others[:, worst], axis=0)
        own = matrix[index, worst]
        ratio = other_median / np.maximum(own, 1.0e-12)
        recoverability[name] = {
            "worst_decile_threshold_mm": float(thresholds[index]),
            "num_worst_frames": int(np.count_nonzero(worst)),
            "median_other_to_own_error_ratio": float(np.median(ratio)),
            "other_median_lower_fraction": float(np.mean(other_median < own)),
            "other_median_at_most_75pct_fraction": float(np.mean(ratio <= 0.75)),
            "other_median_at_most_50pct_fraction": float(np.mean(ratio <= 0.50)),
        }

    worst_masks = matrix >= thresholds[:, None]
    worst_overlap = {
        first: {
            second: float(
                np.mean(worst_masks[i] & worst_masks[j])
                / max(np.mean(worst_masks[i]), 1.0e-12)
            )
            for j, second in enumerate(names)
        }
        for i, first in enumerate(names)
    }
    report = {
        "split": args.split,
        "ground_truth_use": (
            "post-inference TRAIN diagnostic only; never used by Method 4C "
            "alignment, weighting, optimization, or prediction"
        ),
        "observers": list(names),
        "prediction_roots": {name: str(root) for name, root in observers.items()},
        "num_common_sequence_frames": len(common),
        "error_correlation_matrix": {
            first: {second: float(correlation[i, j]) for j, second in enumerate(names)}
            for i, first in enumerate(names)
        },
        "per_observer_error_mm": {
            name: {
                "median": float(np.median(matrix[index])),
                "p90": float(np.percentile(matrix[index], 90)),
                "p95": float(np.percentile(matrix[index], 95)),
                "maximum": float(np.max(matrix[index])),
            }
            for index, name in enumerate(names)
        },
        "worst_decile_recoverability": recoverability,
        "conditional_worst_decile_overlap": worst_overlap,
    }
    print("TRAIN observer aligned-error correlation")
    print(" " * 14 + " ".join(f"{name:>10s}" for name in names))
    for first in names:
        print(
            f"{first:>14s} "
            + " ".join(
                f"{report['error_correlation_matrix'][first][second]:10.4f}"
                for second in names
            )
        )
    print("Worst-decile recoverability")
    for name, payload in recoverability.items():
        print(
            f"{name}: other<=75% {payload['other_median_at_most_75pct_fraction']:.1%}, "
            f"median ratio={payload['median_other_to_own_error_ratio']:.3f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

