"""Prediction/diagnostic loading only; this module never reads ground truth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from imcpe.io_pose import PoseRow, read_pose_txt


@dataclass(frozen=True)
class PoseTrajectory:
    frame_ids: np.ndarray
    centers: np.ndarray
    rotations_initial_from_camera: np.ndarray
    rows: list[PoseRow]


@dataclass(frozen=True)
class PredictionBundle:
    method1: PoseTrajectory
    method2a: PoseTrajectory
    method2b: PoseTrajectory
    rotation_source: PoseTrajectory
    method1_summary: dict[str, Any]
    method1_frames: dict[int, dict[str, Any]]
    method2a_summary: dict[str, Any]
    method2a_frames: dict[int, dict[str, Any]]
    method2b_summary: dict[str, Any]
    method2b_frames: dict[int, dict[str, Any]]


def load_pose_trajectory(path: Path) -> PoseTrajectory:
    rows = read_pose_txt(path)
    if not rows:
        raise RuntimeError(f"Empty prediction: {path}")
    return PoseTrajectory(
        frame_ids=np.asarray([row.frame_idx for row in rows], dtype=np.int64),
        # pose.txt stores T_0_from_t, so this translation is camera center C_t
        # expressed in frame zero. It is not camera-from-world [R|t].
        centers=np.stack([np.asarray(row.t, dtype=np.float64) for row in rows]),
        rotations_initial_from_camera=np.stack(
            [Rotation.from_quat(row.q_xyzw).as_matrix() for row in rows]
        ),
        rows=rows,
    )


def _load_diagnostics(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    if not path.is_file():
        return {}, {}
    payload = json.loads(path.read_text())
    frames = {
        int(item["frame_idx"]): item
        for item in payload.get("frames", [])
        if "frame_idx" in item
    }
    return payload.get("summary", {}), frames


def load_prediction_bundle(
    *,
    split: str,
    sequence: str,
    method1_root: Path,
    method2a_root: Path,
    method2b_root: Path,
    rotation_root: Path,
) -> PredictionBundle:
    roots = {
        "method1": method1_root,
        "method2a": method2a_root,
        "method2b": method2b_root,
        "rotation": rotation_root,
    }
    trajectories = {
        name: load_pose_trajectory(root / split / sequence / "pose.txt")
        for name, root in roots.items()
    }
    reference_ids = trajectories["method2b"].frame_ids
    for name, trajectory in trajectories.items():
        if not np.array_equal(trajectory.frame_ids, reference_ids):
            raise RuntimeError(f"Frame IDs differ between Method 2B and {name}")
        if not np.allclose(trajectory.centers[0], 0.0, atol=1e-7, rtol=0.0):
            raise RuntimeError(f"{name} frame-zero center is not zero")
        if not np.allclose(
            trajectory.rotations_initial_from_camera[0], np.eye(3),
            atol=1e-7, rtol=0.0,
        ):
            raise RuntimeError(f"{name} frame-zero rotation is not identity")
    summaries: dict[str, dict[str, Any]] = {}
    frames: dict[str, dict[int, dict[str, Any]]] = {}
    for name in ("method1", "method2a", "method2b"):
        summaries[name], frames[name] = _load_diagnostics(
            roots[name] / split / sequence / "diagnostics.json"
        )
    return PredictionBundle(
        method1=trajectories["method1"],
        method2a=trajectories["method2a"],
        method2b=trajectories["method2b"],
        rotation_source=trajectories["rotation"],
        method1_summary=summaries["method1"],
        method1_frames=frames["method1"],
        method2a_summary=summaries["method2a"],
        method2a_frames=frames["method2a"],
        method2b_summary=summaries["method2b"],
        method2b_frames=frames["method2b"],
    )
