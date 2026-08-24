from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from imcpe.pose_metrics import RPE_DELTAS, evaluate_split


def _load_diagnostics(
    root: Path, split: str, sequence: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = root / split / sequence / "diagnostics.json"
    if not path.is_file():
        return {}, []
    payload = json.loads(path.read_text())
    return payload.get("summary", {}), payload.get("frames", [])


def compare_five_methods(
    *,
    data_root: Path,
    split: str,
    baseline_root: Path,
    method1_root: Path,
    method1_5_root: Path,
    method2a_root: Path,
    method2b_root: Path,
) -> dict[str, Any]:
    evaluated = {
        "baseline": evaluate_split(data_root, baseline_root, split),
        "method1": evaluate_split(data_root, method1_root, split),
        "method1_5": evaluate_split(data_root, method1_5_root, split),
        "method2a": evaluate_split(data_root, method2a_root, split),
        "method2b": evaluate_split(data_root, method2b_root, split),
    }
    per_method = {name: value[0] for name, value in evaluated.items()}
    global_metrics = {name: value[1] for name, value in evaluated.items()}
    sequence_names = sorted(set().union(*(set(value) for value in per_method.values())))
    per_sequence: dict[str, dict[str, Any]] = {}
    differences: list[tuple[float, str]] = []
    total_frames = 0
    total_successes = 0
    shared: list[float] = []
    inliers: list[float] = []
    ratios: list[float] = []
    reprojection: list[float] = []
    for sequence in sequence_names:
        metrics = {name: per_method[name][sequence] for name in per_method}
        summary, frames = _load_diagnostics(method2b_root, split, sequence)
        row: dict[str, Any] = {
            f"{name}_mean_ate_mm": value["mean_ate"]
            for name, value in metrics.items()
        }
        difference = float(metrics["method2b"]["mean_ate"]) - float(
            metrics["method2a"]["mean_ate"]
        )
        row["method2b_minus_method2a_ate_mm"] = difference
        row["method2b_pnp_success_rate"] = summary.get("pnp_success_rate", np.nan)
        row["method2b_median_shared_tracks"] = summary.get(
            "median_shared_3d_2d_tracks", np.nan
        )
        row["method2b_median_pnp_inliers"] = summary.get("median_pnp_inliers", np.nan)
        row["method2b_median_pnp_inlier_ratio"] = summary.get(
            "median_pnp_inlier_ratio", np.nan
        )
        row["method2b_median_reprojection_error_px"] = summary.get(
            "median_pnp_reprojection_error_px", np.nan
        )
        row["method2b_calibration_valid"] = summary.get("calibration_valid")
        for delta in RPE_DELTAS:
            for name, value in metrics.items():
                row[f"{name}_trans_rpe_{delta}frame_mm"] = value[
                    f"trans_rpe_{delta}frame"
                ]
                row[f"{name}_rot_rpe_{delta}frame_deg"] = value[
                    f"rot_rpe_deg_{delta}frame"
                ]
        per_sequence[sequence] = row
        if np.isfinite(difference):
            differences.append((difference, sequence))
        total_frames += int(summary.get("num_frames", 0))
        total_successes += int(summary.get("num_pnp_successes", 0))
        for frame in frames:
            shared.append(float(frame.get("shared_e2_left_tracks", np.nan)))
            if frame.get("pnp_success", False):
                inliers.append(float(frame.get("pnp_inliers", np.nan)))
                ratios.append(float(frame.get("pnp_inlier_ratio", np.nan)))
                reprojection.append(
                    float(frame.get("pnp_reprojection_error_median_px", np.nan))
                )

    def finite_median(values: list[float]) -> float:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        return float(np.median(array)) if array.size else float("nan")

    improved = [item for item in differences if item[0] < 0.0]
    worsened = [item for item in differences if item[0] > 0.0]
    best = min(differences, default=(float("nan"), ""))
    worst = max(differences, default=(float("nan"), ""))
    return {
        "split": split,
        "per_sequence": per_sequence,
        "global": {
            **global_metrics,
            "method2b_diagnostics": {
                "total_frames": total_frames,
                "total_pnp_successes": total_successes,
                "frame_weighted_pnp_success_rate": (
                    total_successes / total_frames if total_frames else np.nan
                ),
                "median_shared_3d_2d_tracks": finite_median(shared),
                "median_pnp_inliers": finite_median(inliers),
                "median_pnp_inlier_ratio": finite_median(ratios),
                "median_reprojection_error_px": finite_median(reprojection),
            },
            "num_method2b_improved_vs_method2a": len(improved),
            "num_method2b_worsened_vs_method2a": len(worsened),
            "best_method2b_improvement": {
                "sequence": best[1],
                "method2b_minus_method2a_ate_mm": best[0],
            },
            "worst_method2b_regression": {
                "sequence": worst[1],
                "method2b_minus_method2a_ate_mm": worst[0],
            },
        },
    }
