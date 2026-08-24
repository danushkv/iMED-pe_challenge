"""Evaluate one held-out physical session with the official pose metrics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from imcpe.io_pose import read_pose_txt
from imcpe.pose_metrics import RPE_DELTAS, evaluate_sequence

from discover_sessions import SessionRecord


def aggregate_sequence_metrics(
    values: dict[str, dict[str, float | int]],
) -> dict[str, float | int]:
    valid = [item for item in values.values() if int(item["num_matched_poses"]) > 0]
    if not valid:
        result: dict[str, float | int] = {
            "mean_ate": np.nan,
            "median_ate": np.nan,
            "std_ate": np.nan,
            "num_sequences": 0,
            "num_matched_poses": 0,
            "frame_weighted_mean_ate": np.nan,
        }
        for delta in RPE_DELTAS:
            result[f"trans_rpe_{delta}frame"] = np.nan
            result[f"rot_rpe_deg_{delta}frame"] = np.nan
        return result

    counts = np.asarray([int(item["num_matched_poses"]) for item in valid], dtype=float)
    means = np.asarray([float(item["mean_ate"]) for item in valid], dtype=float)
    result = {
        "mean_ate": float(np.mean(means)),
        "median_ate": float(np.mean([float(item["median_ate"]) for item in valid])),
        "std_ate": float(np.mean([float(item["std_ate"]) for item in valid])),
        "num_sequences": len(valid),
        "num_matched_poses": int(np.sum(counts)),
        "frame_weighted_mean_ate": float(np.sum(means * counts) / np.sum(counts)),
    }
    for delta in RPE_DELTAS:
        result[f"trans_rpe_{delta}frame"] = float(
            np.nanmean([float(item[f"trans_rpe_{delta}frame"]) for item in valid])
        )
        result[f"rot_rpe_deg_{delta}frame"] = float(
            np.nanmean([float(item[f"rot_rpe_deg_{delta}frame"]) for item in valid])
        )
    return result


def validate_prediction_coverage(
    *,
    prediction_root: Path,
    split: str,
    sessions: tuple[SessionRecord, ...],
) -> dict[str, Any]:
    expected = {sequence.sequence for session in sessions for sequence in session.sequences}
    prediction_split = prediction_root / split
    if not prediction_split.is_dir():
        raise FileNotFoundError(f"prediction split does not exist: {prediction_split}")
    actual = {
        path.name for path in prediction_split.iterdir()
        if path.is_dir() and (path / "pose.txt").is_file()
    }
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(
            f"prediction coverage mismatch for {prediction_root}: missing={missing}, extra={extra}"
        )

    for session in sessions:
        for sequence in session.sequences:
            rows = read_pose_txt(prediction_split / sequence.sequence / "pose.txt")
            predicted_ids = tuple(int(row.frame_idx) for row in rows)
            if predicted_ids != sequence.frame_ids:
                raise RuntimeError(
                    f"frame IDs differ for {prediction_root}, {sequence.sequence}"
                )
    return {
        "prediction_root": str(prediction_root),
        "num_sequences": len(expected),
        "missing_sequences": [],
        "extra_sequences": [],
        "all_frame_ids_equal_ground_truth": True,
    }


def evaluate_session(
    *,
    session: SessionRecord,
    prediction_root: Path,
    split: str,
) -> dict[str, Any]:
    per_sequence: dict[str, dict[str, float | int]] = {}
    for sequence in session.sequences:
        # Ground truth is opened only inside this evaluation function. It is not
        # exposed to configuration selection or inference-time operations.
        gt_rows = read_pose_txt(sequence.path / "pose.txt")
        pred_rows = read_pose_txt(
            prediction_root / split / sequence.sequence / "pose.txt"
        )
        per_sequence[sequence.sequence] = evaluate_sequence(gt_rows, pred_rows)
    return {
        "session_id": session.session_id,
        "per_sequence": per_sequence,
        "aggregate": aggregate_sequence_metrics(per_sequence),
    }
