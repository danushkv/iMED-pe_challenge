from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from imcpe.pose_metrics import RPE_DELTAS, evaluate_split


def compare_three_methods(
    data_root: Path,
    split: str,
    baseline_root: Path,
    method1_root: Path,
    method1_5_root: Path,
) -> dict[str, Any]:
    """Use the repository evaluator for all three methods."""

    baseline_per_sequence, baseline_global = evaluate_split(
        data_root, baseline_root, split
    )
    method1_per_sequence, method1_global = evaluate_split(
        data_root, method1_root, split
    )
    anchored_per_sequence, anchored_global = evaluate_split(
        data_root, method1_5_root, split
    )
    sequence_names = sorted(
        set(baseline_per_sequence)
        | set(method1_per_sequence)
        | set(anchored_per_sequence)
    )
    rows: dict[str, dict[str, float | int]] = {}
    differences: list[tuple[float, str]] = []
    for sequence_name in sequence_names:
        baseline = baseline_per_sequence[sequence_name]
        method1 = method1_per_sequence[sequence_name]
        anchored = anchored_per_sequence[sequence_name]
        row: dict[str, float | int] = {
            "baseline_mean_ate_mm": baseline["mean_ate"],
            "method1_mean_ate_mm": method1["mean_ate"],
            "method1_5_mean_ate_mm": anchored["mean_ate"],
            "method1_5_minus_method1_ate_mm": (
                float(anchored["mean_ate"]) - float(method1["mean_ate"])
            ),
        }
        for delta in RPE_DELTAS:
            for name, metrics in (
                ("baseline", baseline),
                ("method1", method1),
                ("method1_5", anchored),
            ):
                row[f"{name}_trans_rpe_{delta}frame_mm"] = metrics[
                    f"trans_rpe_{delta}frame"
                ]
                row[f"{name}_rot_rpe_{delta}frame_deg"] = metrics[
                    f"rot_rpe_deg_{delta}frame"
                ]
        rows[sequence_name] = row
        differences.append((float(row["method1_5_minus_method1_ate_mm"]), sequence_name))

    finite_differences = [item for item in differences if np.isfinite(item[0])]
    improved = [item for item in finite_differences if item[0] < 0.0]
    worsened = [item for item in finite_differences if item[0] > 0.0]
    best = min(finite_differences, default=(float("nan"), ""))
    worst = max(finite_differences, default=(float("nan"), ""))
    return {
        "split": split,
        "per_sequence": rows,
        "global": {
            "baseline": baseline_global,
            "method1": method1_global,
            "method1_5": anchored_global,
            "num_improved_vs_method1": len(improved),
            "num_worsened_vs_method1": len(worsened),
            "best_improvement": {
                "sequence": best[1],
                "method1_5_minus_method1_ate_mm": best[0],
            },
            "worst_regression": {
                "sequence": worst[1],
                "method1_5_minus_method1_ate_mm": worst[0],
            },
        },
    }

