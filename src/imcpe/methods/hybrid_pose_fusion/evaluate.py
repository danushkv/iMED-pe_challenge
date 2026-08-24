from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from imcpe.pose_metrics import RPE_DELTAS, evaluate_split


def _aggregate(values: dict[str, dict[str, float | int]]) -> dict[str, float]:
    valid = [item for item in values.values() if item["num_matched_poses"] > 0]
    if not valid:
        return {
            "mean_ate": np.nan, "std_ate": np.nan,
            "median_ate": np.nan, "num_sequences": 0.0,
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


def compare_method4(
    *,
    data_root: Path,
    split: str,
    prediction_roots: dict[str, Path],
    selected_sequences: list[str] | None = None,
) -> dict[str, Any]:
    evaluated = {
        name: evaluate_split(data_root, root, split)[0]
        for name, root in prediction_roots.items()
    }
    if selected_sequences is not None:
        selected = set(selected_sequences)
        evaluated = {
            name: {key: value for key, value in per_sequence.items() if key in selected}
            for name, per_sequence in evaluated.items()
        }
    sequence_names = sorted(set().union(*(set(value) for value in evaluated.values())))
    per_sequence: dict[str, dict[str, Any]] = {}
    for sequence in sequence_names:
        row: dict[str, Any] = {}
        for name, values in evaluated.items():
            metrics = values[sequence]
            row[f"{name}_mean_ate_mm"] = metrics["mean_ate"]
            row[f"{name}_median_ate_mm"] = metrics["median_ate"]
            for delta in RPE_DELTAS:
                row[f"{name}_trans_rpe_{delta}frame_mm"] = metrics[
                    f"trans_rpe_{delta}frame"
                ]
                row[f"{name}_rot_rpe_{delta}frame_deg"] = metrics[
                    f"rot_rpe_deg_{delta}frame"
                ]
        for name in ("method4a0", "method4a1"):
            if name not in prediction_roots:
                continue
            path = prediction_roots[name] / split / sequence / "diagnostics.json"
            if path.is_file():
                summary = json.loads(path.read_text()).get("summary", {})
                row[f"{name}_optimization_seconds"] = summary.get(
                    "optimization_seconds", np.nan
                )
                row[f"{name}_downweighted_frames"] = summary.get(
                    "num_anchor_frames_downweighted_below_0_5", 0
                )
        per_sequence[sequence] = row
    return {
        "split": split,
        "per_sequence": per_sequence,
        "global": {name: _aggregate(values) for name, values in evaluated.items()},
    }
