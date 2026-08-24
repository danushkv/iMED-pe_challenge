from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from imcpe.pose_metrics import RPE_DELTAS, evaluate_split


def _load_summary(root: Path, split: str, sequence: str) -> dict[str, Any]:
    path = root / split / sequence / "diagnostics.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text()).get("summary", {})


def compare_method3(
    *,
    data_root: Path,
    split: str,
    prediction_roots: dict[str, Path],
    selected_sequences: list[str] | None = None,
) -> dict[str, Any]:
    evaluated = {
        name: evaluate_split(data_root, root, split)
        for name, root in prediction_roots.items()
    }
    per_method = {name: result[0] for name, result in evaluated.items()}
    if selected_sequences is not None:
        selected = set(selected_sequences)
        missing = sorted(
            selected - set().union(*(set(value) for value in per_method.values()))
        )
        if missing:
            raise KeyError(f"Unknown selected sequences: {missing}")
        per_method = {
            name: {sequence: metrics for sequence, metrics in values.items() if sequence in selected}
            for name, values in per_method.items()
        }

    def aggregate(values: dict[str, dict[str, float | int]]) -> dict[str, float]:
        valid = [item for item in values.values() if item["num_matched_poses"] > 0]
        if not valid:
            return {
                "mean_ate": float("nan"),
                "std_ate": float("nan"),
                "median_ate": float("nan"),
                "num_sequences": 0.0,
            }
        result = {
            "mean_ate": float(np.mean([item["mean_ate"] for item in valid])),
            "std_ate": float(np.mean([item["std_ate"] for item in valid])),
            "median_ate": float(np.mean([item["median_ate"] for item in valid])),
            "num_sequences": float(len(valid)),
        }
        for delta in RPE_DELTAS:
            result[f"trans_rpe_{delta}frame"] = float(
                np.mean([item[f"trans_rpe_{delta}frame"] for item in valid])
            )
            result[f"rot_rpe_deg_{delta}frame"] = float(
                np.mean([item[f"rot_rpe_deg_{delta}frame"] for item in valid])
            )
        return result

    global_metrics = {
        name: aggregate(values) for name, values in per_method.items()
    }
    sequence_names = sorted(set().union(*(set(value) for value in per_method.values())))
    per_sequence: dict[str, dict[str, Any]] = {}
    for sequence in sequence_names:
        row: dict[str, Any] = {}
        for name in per_method:
            metrics = per_method[name][sequence]
            row[f"{name}_mean_ate_mm"] = metrics["mean_ate"]
            row[f"{name}_median_ate_mm"] = metrics["median_ate"]
            for delta in RPE_DELTAS:
                row[f"{name}_trans_rpe_{delta}frame_mm"] = metrics[
                    f"trans_rpe_{delta}frame"
                ]
                row[f"{name}_rot_rpe_{delta}frame_deg"] = metrics[
                    f"rot_rpe_deg_{delta}frame"
                ]
        for method3_name in ("method3a", "method3b"):
            if method3_name in prediction_roots:
                summary = _load_summary(
                    prediction_roots[method3_name], split, sequence
                )
                row[f"{method3_name}_runtime_seconds"] = summary.get(
                    "total_camera_inference_seconds", np.nan
                )
                row[f"{method3_name}_peak_gpu_memory_bytes"] = summary.get(
                    "peak_gpu_memory_bytes", 0
                )
                row[f"{method3_name}_valid_fraction"] = summary.get(
                    "valid_fraction", np.nan
                )
        per_sequence[sequence] = row
    return {
        "split": split,
        "per_sequence": per_sequence,
        "global": global_metrics,
    }
