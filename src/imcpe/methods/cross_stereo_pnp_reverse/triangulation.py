from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import ReverseMethod2Config


@dataclass(frozen=True)
class E2StereoCalibration:
    session_id: str
    K_left: np.ndarray
    K_right: np.ndarray
    R_LR: np.ndarray
    t_LR_unit: np.ndarray
    T_R_from_L: np.ndarray
    suggested_min_depth: float
    suggested_max_depth: float
    valid: bool | None
    validity_failures: tuple[str, ...]
    source_path: Path


@dataclass(frozen=True)
class ReverseTriangulationResult:
    points_e2_left_3d: np.ndarray
    valid_mask: np.ndarray
    depth_left: np.ndarray
    depth_right: np.ndarray
    reprojection_error_px: np.ndarray
    triangulation_angle_deg: np.ndarray
    applied_min_depth: float
    applied_max_depth: float


def _diagnostics_from_archive(archive: np.lib.npyio.NpzFile) -> dict[str, object]:
    if "diagnostic_statistics_json" not in archive.files:
        return {}
    raw = archive["diagnostic_statistics_json"].item()
    try:
        return json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        return {}


def load_e2_stereo_calibration(
    calibration_root: Path, session_id: str
) -> E2StereoCalibration:
    """Load either a Method-1-compatible or native Method-2B archive."""

    path = calibration_root / f"session_{session_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing E2 stereo calibration: {path}")
    with np.load(path, allow_pickle=False) as archive:
        diagnostics = _diagnostics_from_archive(archive)
        archived_session = (
            str(archive["session_id"].item()).zfill(3)
            if "session_id" in archive.files
            else str(session_id).zfill(3)
        )
        if archived_session != str(session_id).zfill(3):
            raise RuntimeError(f"Calibration session mismatch in {path}")
        suggested_min = (
            float(archive["suggested_min_depth"].item())
            if "suggested_min_depth" in archive.files
            else float(diagnostics.get("suggested_min_depth", np.nan))
        )
        suggested_max = (
            float(archive["suggested_max_depth"].item())
            if "suggested_max_depth" in archive.files
            else float(diagnostics.get("suggested_max_depth", np.nan))
        )
        valid_value = diagnostics.get("fixed_transform_appears_valid")
        valid = bool(valid_value) if valid_value is not None else None
        failures = tuple(str(item) for item in diagnostics.get("validity_failures", []))
        return E2StereoCalibration(
            session_id=archived_session,
            K_left=np.asarray(archive["K_left"], dtype=np.float64),
            K_right=np.asarray(archive["K_right"], dtype=np.float64),
            R_LR=np.asarray(archive["R_LR"], dtype=np.float64),
            t_LR_unit=np.asarray(archive["t_LR_unit"], dtype=np.float64).reshape(3),
            T_R_from_L=np.asarray(archive["T_R_from_L"], dtype=np.float64),
            suggested_min_depth=suggested_min,
            suggested_max_depth=suggested_max,
            valid=valid,
            validity_failures=failures,
            source_path=path,
        )


def _normalise_points(points_px: np.ndarray, K: np.ndarray) -> np.ndarray:
    return cv2.undistortPoints(
        np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2), K, None
    ).reshape(-1, 2)


def _project(K: np.ndarray, points_camera: np.ndarray) -> np.ndarray:
    homogeneous = (K @ points_camera.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _triangulation_angles(
    points_left: np.ndarray, R_LR: np.ndarray, t_LR: np.ndarray
) -> np.ndarray:
    right_center_in_left = -R_LR.T @ t_LR.reshape(3)
    left_rays = points_left
    right_rays = points_left - right_center_in_left.reshape(1, 3)
    denominator = np.maximum(
        np.linalg.norm(left_rays, axis=1) * np.linalg.norm(right_rays, axis=1),
        np.finfo(np.float64).eps,
    )
    cosine = np.sum(left_rays * right_rays, axis=1) / denominator
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def triangulate_e2_stereo(
    points_left_px: np.ndarray,
    points_right_px: np.ndarray,
    calibration: E2StereoCalibration,
    config: ReverseMethod2Config,
) -> ReverseTriangulationResult:
    count = int(np.asarray(points_left_px).shape[0])
    if count == 0:
        empty = np.empty(0, dtype=np.float64)
        return ReverseTriangulationResult(
            points_e2_left_3d=np.empty((0, 3), dtype=np.float64),
            valid_mask=np.empty(0, dtype=bool),
            depth_left=empty,
            depth_right=empty,
            reprojection_error_px=empty,
            triangulation_angle_deg=empty,
            applied_min_depth=config.min_depth,
            applied_max_depth=config.max_depth,
        )

    left_normalized = _normalise_points(points_left_px, calibration.K_left)
    right_normalized = _normalise_points(points_right_px, calibration.K_right)
    P_left = np.hstack([np.eye(3), np.zeros((3, 1))])
    P_right = np.hstack(
        [calibration.R_LR, calibration.t_LR_unit.reshape(3, 1)]
    )
    homogeneous = cv2.triangulatePoints(
        P_left, P_right, left_normalized.T, right_normalized.T
    )
    finite_w = np.abs(homogeneous[3]) > np.finfo(np.float64).eps
    points_left = np.full((count, 3), np.nan, dtype=np.float64)
    points_left[finite_w] = (
        homogeneous[:3, finite_w] / homogeneous[3, finite_w]
    ).T
    points_right = (
        calibration.R_LR @ points_left.T
    ).T + calibration.t_LR_unit.reshape(1, 3)
    depth_left = points_left[:, 2]
    depth_right = points_right[:, 2]
    finite = (
        finite_w
        & np.isfinite(points_left).all(axis=1)
        & np.isfinite(points_right).all(axis=1)
    )

    left_error = np.full(count, np.inf, dtype=np.float64)
    right_error = np.full(count, np.inf, dtype=np.float64)
    average_error = np.full(count, np.inf, dtype=np.float64)
    if np.any(finite):
        left_error[finite] = np.linalg.norm(
            _project(calibration.K_left, points_left[finite])
            - np.asarray(points_left_px, dtype=np.float64)[finite],
            axis=1,
        )
        right_error[finite] = np.linalg.norm(
            _project(calibration.K_right, points_right[finite])
            - np.asarray(points_right_px, dtype=np.float64)[finite],
            axis=1,
        )
        average_error[finite] = 0.5 * (left_error[finite] + right_error[finite])

    angles = np.full(count, np.nan, dtype=np.float64)
    if np.any(finite):
        angles[finite] = _triangulation_angles(
            points_left[finite], calibration.R_LR, calibration.t_LR_unit
        )

    minimum_depth = config.min_depth
    maximum_depth = config.max_depth
    if config.use_calibration_depth_bounds:
        if np.isfinite(calibration.suggested_min_depth):
            minimum_depth = max(minimum_depth, calibration.suggested_min_depth)
        if np.isfinite(calibration.suggested_max_depth):
            maximum_depth = min(maximum_depth, calibration.suggested_max_depth)
    if maximum_depth <= minimum_depth:
        raise RuntimeError(
            f"Invalid applied depth interval: [{minimum_depth}, {maximum_depth}]"
        )

    valid = (
        finite
        & (depth_left > minimum_depth)
        & (depth_right > minimum_depth)
        & (depth_left < maximum_depth)
        & (depth_right < maximum_depth)
        & (left_error <= config.max_stereo_reprojection_error_px)
        & (right_error <= config.max_stereo_reprojection_error_px)
    )
    if config.min_triangulation_angle_deg > 0.0:
        valid &= angles >= config.min_triangulation_angle_deg

    return ReverseTriangulationResult(
        points_e2_left_3d=points_left,
        valid_mask=valid,
        depth_left=depth_left,
        depth_right=depth_right,
        reprojection_error_px=average_error,
        triangulation_angle_deg=angles,
        applied_min_depth=float(minimum_depth),
        applied_max_depth=float(maximum_depth),
    )
