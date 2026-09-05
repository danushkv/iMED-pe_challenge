"""Pose/diagnostic serialization shared by LOSO and frozen final inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from imcpe.io_pose import PoseRow, write_pose_txt

from .config import EXPERT_NAMES
from .dataset import SequenceRecord
from .router import router_metrics
from .routed_optimizer import RoutedResult


def _json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_trajectory(
    root: Path,
    split: str,
    record: SequenceRecord,
    centers: np.ndarray,
    diagnostics: dict[str, Any],
) -> None:
    if not np.isfinite(centers).all():
        raise RuntimeError(f"Non-finite centers for {record.sequence}")
    rotations = record.rotation_source
    determinant_error = float(np.max(np.abs(np.linalg.det(rotations) - 1.0)))
    orthogonality_error = float(
        np.max(
            np.linalg.norm(
                np.transpose(rotations, (0, 2, 1)) @ rotations - np.eye(3),
                axis=(1, 2),
            )
        )
    )
    if determinant_error > 1.0e-7 or orthogonality_error > 1.0e-7:
        raise RuntimeError("Frozen rotation source is not valid SO(3)")
    rows = [
        PoseRow(
            frame_idx=int(frame_id),
            t=centers[index].copy(),
            q_xyzw=Rotation.from_matrix(rotations[index]).as_quat(),
        )
        for index, frame_id in enumerate(record.frame_ids)
    ]
    output = root / split / record.sequence
    write_pose_txt(output / "pose.txt", rows)
    diagnostics["maximum_rotation_determinant_error"] = determinant_error
    diagnostics["maximum_rotation_orthogonality_error"] = orthogonality_error
    diagnostics["held_session_ground_truth_used_for_routing"] = False
    (output / "diagnostics.json").write_text(
        json.dumps({"summary": diagnostics}, indent=2, default=_json_default) + "\n"
    )


def write_routed_outputs(
    *,
    output_root: Path,
    record: SequenceRecord,
    probabilities: np.ndarray,
    routed: RoutedResult,
    fold: str,
) -> dict[str, Any]:
    metrics = (
        router_metrics(probabilities, record.expert_errors)
        if record.expert_errors is not None
        else {
            "num_frames": len(record.frame_ids),
            "mean_entropy": float(np.mean(routed.entropy)),
            "highest_probability_fraction": {
                name: float(np.mean(np.argmax(probabilities, axis=1) == index))
                for index, name in enumerate(EXPERT_NAMES)
            },
            "mean_probability": {
                name: float(np.mean(probabilities[:, index]))
                for index, name in enumerate(EXPERT_NAMES)
            },
        }
    )
    common = {
        "sequence": record.sequence,
        "session": record.session,
        "outer_fold": fold,
        "router": metrics,
        "entropy_strength_mean": float(np.mean(routed.entropy_strength)),
    }
    _write_trajectory(
        output_root / "method6a", record.split, record, routed.centers_6a, {**common, "method": "6a"}
    )
    _write_trajectory(
        output_root / "method6b",
        record.split,
        record,
        routed.centers_6b,
        {**common, "method": "6b", "optimization_seconds": routed.optimization_b_seconds},
    )
    _write_trajectory(
        output_root / "method6c",
        record.split,
        record,
        routed.centers_6c,
        {**common, "method": "6c", "optimization_seconds": routed.optimization_c_seconds},
    )
    selected = np.argmax(probabilities, axis=1)
    hard = record.expert_centers[selected, np.arange(len(selected))]
    hard -= hard[0]
    _write_trajectory(
        output_root / "router_hard",
        record.split,
        record,
        hard,
        {**common, "method": "router_hard"},
    )
    probability_path = (
        output_root / "router_probabilities" / record.split / f"{record.sequence}.npz"
    )
    probability_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        probability_path,
        frame_ids=record.frame_ids,
        probabilities=probabilities,
        entropy=routed.entropy,
        entropy_strength=routed.entropy_strength,
    )
    return metrics
