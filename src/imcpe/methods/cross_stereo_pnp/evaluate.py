from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from imcpe.pose_metrics import RPE_DELTAS, evaluate_split


def _load_method2_diagnostics(
    method2_root: Path, split: str, sequence_name: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = method2_root / split / sequence_name / "diagnostics.json"
    if not path.is_file():
        return {}, []
    payload = json.loads(path.read_text())
    return payload.get("summary", {}), payload.get("frames", [])


def compare_four_methods(
    *,
    data_root: Path,
    split: str,
    baseline_root: Path,
    method1_root: Path,
    method1_5_root: Path,
    method2_root: Path,
) -> dict[str, Any]:
    """Evaluate every method through the repository's official evaluator."""

    evaluated = {
        "baseline": evaluate_split(data_root, baseline_root, split),
        "method1": evaluate_split(data_root, method1_root, split),
        "method1_5": evaluate_split(data_root, method1_5_root, split),
        "method2": evaluate_split(data_root, method2_root, split),
    }
    per_method = {name: result[0] for name, result in evaluated.items()}
    global_metrics = {name: result[1] for name, result in evaluated.items()}
    sequence_names = sorted(set().union(*(set(value) for value in per_method.values())))
    per_sequence: dict[str, dict[str, Any]] = {}
    method2_minus_method1_5: list[tuple[float, str]] = []
    total_frames = 0
    total_successes = 0
    shared_tracks: list[float] = []
    pnp_inliers: list[float] = []
    pnp_ratios: list[float] = []
    reprojection_errors: list[float] = []

    for sequence_name in sequence_names:
        metrics_by_name = {
            name: per_method[name][sequence_name] for name in per_method
        }
        summary, frames = _load_method2_diagnostics(
            method2_root, split, sequence_name
        )
        row: dict[str, Any] = {
            f"{name}_mean_ate_mm": metrics["mean_ate"]
            for name, metrics in metrics_by_name.items()
        }
        row["method2_minus_method1_5_ate_mm"] = float(
            metrics_by_name["method2"]["mean_ate"]
        ) - float(metrics_by_name["method1_5"]["mean_ate"])
        row["method2_pnp_success_rate"] = summary.get(
            "pnp_success_rate", float("nan")
        )
        row["method2_median_shared_tracks"] = summary.get(
            "median_shared_3d_2d_tracks", float("nan")
        )
        row["method2_median_pnp_inliers"] = summary.get(
            "median_pnp_inliers", float("nan")
        )
        row["method2_median_pnp_inlier_ratio"] = summary.get(
            "median_pnp_inlier_ratio", float("nan")
        )
        row["method2_median_reprojection_error_px"] = summary.get(
            "median_pnp_reprojection_error_px", float("nan")
        )
        for delta in RPE_DELTAS:
            for name, metrics in metrics_by_name.items():
                row[f"{name}_trans_rpe_{delta}frame_mm"] = metrics[
                    f"trans_rpe_{delta}frame"
                ]
                row[f"{name}_rot_rpe_{delta}frame_deg"] = metrics[
                    f"rot_rpe_deg_{delta}frame"
                ]
        per_sequence[sequence_name] = row
        difference = float(row["method2_minus_method1_5_ate_mm"])
        if np.isfinite(difference):
            method2_minus_method1_5.append((difference, sequence_name))
        total_frames += int(summary.get("num_frames", 0))
        total_successes += int(summary.get("num_pnp_successes", 0))
        for frame in frames:
            shared_tracks.append(float(frame.get("shared_e1_left_tracks", np.nan)))
            if frame.get("pnp_success", False):
                pnp_inliers.append(float(frame.get("pnp_inliers", np.nan)))
                pnp_ratios.append(float(frame.get("pnp_inlier_ratio", np.nan)))
                reprojection_errors.append(
                    float(frame.get("pnp_reprojection_error_median_px", np.nan))
                )

    def finite_median(values: list[float]) -> float:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        return float(np.median(array)) if array.size else float("nan")

    improved = [item for item in method2_minus_method1_5 if item[0] < 0.0]
    worsened = [item for item in method2_minus_method1_5 if item[0] > 0.0]
    best = min(method2_minus_method1_5, default=(float("nan"), ""))
    worst = max(method2_minus_method1_5, default=(float("nan"), ""))
    return {
        "split": split,
        "per_sequence": per_sequence,
        "global": {
            **global_metrics,
            "method2_diagnostics": {
                "total_frames": total_frames,
                "total_pnp_successes": total_successes,
                "frame_weighted_pnp_success_rate": (
                    total_successes / total_frames if total_frames else float("nan")
                ),
                "median_shared_3d_2d_tracks": finite_median(shared_tracks),
                "median_pnp_inliers": finite_median(pnp_inliers),
                "median_pnp_inlier_ratio": finite_median(pnp_ratios),
                "median_pnp_reprojection_error_px": finite_median(
                    reprojection_errors
                ),
            },
            "num_method2_improved_vs_method1_5": len(improved),
            "num_method2_worsened_vs_method1_5": len(worsened),
            "best_method2_improvement": {
                "sequence": best[1],
                "method2_minus_method1_5_ate_mm": best[0],
            },
            "worst_method2_regression": {
                "sequence": worst[1],
                "method2_minus_method1_5_ate_mm": worst[0],
            },
        },
    }

