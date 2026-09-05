"""Load saved predictions only. This module never opens challenge ground truth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from imcpe.io_pose import PoseRow, read_pose_txt


@dataclass(frozen=True)
class LoadedTrajectory:
    name: str
    frame_ids: np.ndarray
    centers: np.ndarray
    rotations_initial_from_camera: np.ndarray
    valid: np.ndarray
    rows_by_id: dict[int, PoseRow]
    summary: dict[str, Any]
    frames: dict[int, dict[str, Any]]


@dataclass(frozen=True)
class ObserverBundle:
    frame_ids: np.ndarray
    observers: dict[str, LoadedTrajectory]
    method1: LoadedTrajectory
    rotation_source: LoadedTrajectory
    reference_name: str


def parse_named_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise ValueError("Expected non-empty NAME=PATH")
    return name, Path(raw_path)


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


def _read_rows(path: Path) -> dict[int, PoseRow]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved prediction: {path}")
    rows = read_pose_txt(path)
    if not rows:
        raise RuntimeError(f"Empty saved prediction: {path}")
    by_id = {int(row.frame_idx): row for row in rows}
    if len(by_id) != len(rows):
        raise RuntimeError(f"Duplicate frame IDs in {path}")
    return by_id


def load_trajectory_on_reference(
    *,
    name: str,
    sequence_root: Path,
    reference_ids: np.ndarray,
    require_complete: bool,
) -> LoadedTrajectory:
    rows = _read_rows(sequence_root / "pose.txt")
    centers = np.full((len(reference_ids), 3), np.nan, dtype=np.float64)
    rotations = np.full((len(reference_ids), 3, 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(reference_ids), dtype=bool)
    for index, frame_id in enumerate(reference_ids):
        row = rows.get(int(frame_id))
        if row is None:
            continue
        center = np.asarray(row.t, dtype=np.float64)
        quaternion = np.asarray(row.q_xyzw, dtype=np.float64)
        if not np.isfinite(center).all() or not np.isfinite(quaternion).all():
            continue
        centers[index] = center
        rotations[index] = Rotation.from_quat(quaternion).as_matrix()
        valid[index] = True
    missing = reference_ids[~valid]
    if require_complete and len(missing):
        raise RuntimeError(
            f"{name} is missing/non-finite at frame IDs {missing.tolist()}"
        )
    summary, frames = _load_diagnostics(sequence_root / "diagnostics.json")
    return LoadedTrajectory(
        name=name,
        frame_ids=reference_ids.copy(),
        centers=centers,
        rotations_initial_from_camera=rotations,
        valid=valid,
        rows_by_id=rows,
        summary=summary,
        frames=frames,
    )


def load_observer_bundle(
    *,
    split: str,
    sequence: str,
    observer_roots: dict[str, Path],
    reference_name: str,
    method1_root: Path,
    rotation_root: Path,
) -> ObserverBundle:
    if reference_name not in observer_roots:
        raise KeyError(f"Reference observer {reference_name!r} was not supplied")
    reference_sequence = observer_roots[reference_name] / split / sequence
    reference_rows = _read_rows(reference_sequence / "pose.txt")
    frame_ids = np.asarray(sorted(reference_rows), dtype=np.int64)
    if not len(frame_ids):
        raise RuntimeError(f"No reference frames for {sequence}")

    observers = {
        name: load_trajectory_on_reference(
            name=name,
            sequence_root=root / split / sequence,
            reference_ids=frame_ids,
            require_complete=(name == reference_name),
        )
        for name, root in observer_roots.items()
    }
    method1 = load_trajectory_on_reference(
        name="method1",
        sequence_root=method1_root / split / sequence,
        reference_ids=frame_ids,
        require_complete=True,
    )
    rotation_source = load_trajectory_on_reference(
        name="rotation",
        sequence_root=rotation_root / split / sequence,
        reference_ids=frame_ids,
        require_complete=True,
    )

    for name, trajectory in {
        **observers,
        "method1": method1,
        "rotation": rotation_source,
    }.items():
        if trajectory.valid[0]:
            if not np.allclose(trajectory.centers[0], 0.0, atol=1e-7, rtol=0.0):
                raise RuntimeError(f"{name} frame-zero center is not zero")
            if not np.allclose(
                trajectory.rotations_initial_from_camera[0],
                np.eye(3),
                atol=1e-7,
                rtol=0.0,
            ):
                raise RuntimeError(f"{name} frame-zero rotation is not identity")

    return ObserverBundle(
        frame_ids=frame_ids,
        observers=observers,
        method1=method1,
        rotation_source=rotation_source,
        reference_name=reference_name,
    )

