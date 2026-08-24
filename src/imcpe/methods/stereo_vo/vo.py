from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from imcpe.data_io import load_sequence
from imcpe.io_pose import PoseRow

from .features import SharedFeatureMatcher, intersect_left_keypoint_tracks


@dataclass(frozen=True)
class StereoVOConfig:
    min_shared_tracks: int = 8
    min_depth: float = 0.05
    max_depth: float = 200.0
    max_stereo_reprojection_error_px: float = 2.0
    pnp_min_points: int = 8
    pnp_min_inliers: int = 8
    pnp_iterations: int = 200
    pnp_reprojection_error_px: float = 3.0
    pnp_confidence: float = 0.999


@dataclass(frozen=True)
class SessionStereoCalibration:
    session_id: str
    K_left: np.ndarray
    K_right: np.ndarray
    R_LR: np.ndarray
    t_LR_unit: np.ndarray
    T_R_from_L: np.ndarray


@dataclass(frozen=True)
class TriangulationResult:
    points_left_3d: np.ndarray
    valid_mask: np.ndarray
    reprojection_error_px: np.ndarray
    depth_left: np.ndarray
    depth_right: np.ndarray


@dataclass(frozen=True)
class PnPResult:
    success: bool
    failure_reason: str | None
    R_current_from_previous: np.ndarray | None
    t_current_from_previous: np.ndarray | None
    inlier_indices: np.ndarray
    inlier_reprojection_errors_px: np.ndarray


@dataclass
class FrameDiagnostic:
    previous_frame_idx: int
    frame_idx: int
    stereo_matches: int
    temporal_matches: int
    shared_tracks: int
    valid_3d_points: int
    pnp_success: bool
    failure_reason: str | None
    pnp_inliers: int
    pnp_inlier_ratio: float
    pnp_reprojection_error_mean_px: float
    pnp_reprojection_error_median_px: float
    pnp_reprojection_error_p95_px: float
    stereo_reprojection_error_median_px: float
    stereo_reprojection_error_p95_px: float
    triangulated_depth_median_baselines: float
    triangulated_depth_p95_baselines: float
    incremental_translation_baselines: float
    incremental_rotation_deg: float


@dataclass(frozen=True)
class DebugFrame:
    previous_frame_idx: int
    frame_idx: int
    previous_left_path: Path
    previous_right_path: Path
    current_left_path: Path
    stereo_left_points: np.ndarray
    stereo_right_points: np.ndarray
    temporal_previous_points: np.ndarray
    temporal_current_points: np.ndarray
    pnp_previous_points: np.ndarray
    pnp_current_points: np.ndarray


@dataclass(frozen=True)
class SequenceVOResult:
    rows: list[PoseRow]
    diagnostics: list[FrameDiagnostic]
    debug_frames: list[DebugFrame]
    summary: dict[str, Any]


def session_id_from_sequence_name(sequence_name: str) -> str:
    match = re.match(r"^session_(\d+)_", sequence_name)
    if match is None:
        raise ValueError(f"Cannot determine session ID from sequence name: {sequence_name}")
    return match.group(1).zfill(3)


def load_session_stereo_calibration(
    calibration_root: Path,
    session_id: str,
) -> SessionStereoCalibration:
    path = calibration_root / f"session_{session_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing session calibration: {path}")
    with np.load(path, allow_pickle=False) as archive:
        calibration = SessionStereoCalibration(
            session_id=str(archive["session_id"].item()),
            K_left=np.asarray(archive["K_left"], dtype=np.float64),
            K_right=np.asarray(archive["K_right"], dtype=np.float64),
            R_LR=np.asarray(archive["R_LR"], dtype=np.float64),
            t_LR_unit=np.asarray(archive["t_LR_unit"], dtype=np.float64).reshape(3),
            T_R_from_L=np.asarray(archive["T_R_from_L"], dtype=np.float64),
        )
    if calibration.session_id != session_id:
        raise RuntimeError(
            f"Calibration session mismatch: requested {session_id}, file stores "
            f"{calibration.session_id}"
        )
    return calibration


def _normalise_points(points_px: np.ndarray, K: np.ndarray) -> np.ndarray:
    return cv2.undistortPoints(
        np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2),
        K,
        None,
    ).reshape(-1, 2)


def _project_points(K: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    homogeneous = (K @ points_3d.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def triangulate_stereo_tracks(
    points_left_px: np.ndarray,
    points_right_px: np.ndarray,
    calibration: SessionStereoCalibration,
    config: StereoVOConfig,
) -> TriangulationResult:
    count = int(points_left_px.shape[0])
    if count == 0:
        return TriangulationResult(
            points_left_3d=np.empty((0, 3), dtype=np.float64),
            valid_mask=np.empty(0, dtype=bool),
            reprojection_error_px=np.empty(0, dtype=np.float64),
            depth_left=np.empty(0, dtype=np.float64),
            depth_right=np.empty(0, dtype=np.float64),
        )

    left_norm = _normalise_points(points_left_px, calibration.K_left)
    right_norm = _normalise_points(points_right_px, calibration.K_right)
    P_left = np.hstack([np.eye(3), np.zeros((3, 1))])
    P_right = np.hstack(
        [calibration.R_LR, calibration.t_LR_unit.reshape(3, 1)]
    )
    homogeneous = cv2.triangulatePoints(P_left, P_right, left_norm.T, right_norm.T)
    finite_w = np.abs(homogeneous[3]) > np.finfo(np.float64).eps
    points_left_3d = np.full((count, 3), np.nan, dtype=np.float64)
    points_left_3d[finite_w] = (
        homogeneous[:3, finite_w] / homogeneous[3, finite_w]
    ).T
    points_right_3d = (
        calibration.R_LR @ points_left_3d.T
    ).T + calibration.t_LR_unit
    depth_left = points_left_3d[:, 2]
    depth_right = points_right_3d[:, 2]

    finite = (
        finite_w
        & np.isfinite(points_left_3d).all(axis=1)
        & np.isfinite(points_right_3d).all(axis=1)
    )
    left_reprojection_error = np.full(count, np.inf, dtype=np.float64)
    right_reprojection_error = np.full(count, np.inf, dtype=np.float64)
    reprojection_error = np.full(count, np.inf, dtype=np.float64)
    if np.any(finite):
        projected_left = _project_points(calibration.K_left, points_left_3d[finite])
        projected_right = _project_points(calibration.K_right, points_right_3d[finite])
        left_reprojection_error[finite] = np.linalg.norm(
            projected_left - points_left_px[finite], axis=1
        )
        right_reprojection_error[finite] = np.linalg.norm(
            projected_right - points_right_px[finite], axis=1
        )
        reprojection_error[finite] = 0.5 * (
            left_reprojection_error[finite] + right_reprojection_error[finite]
        )

    valid = (
        finite
        & (depth_left > config.min_depth)
        & (depth_right > config.min_depth)
        & (depth_left < config.max_depth)
        & (depth_right < config.max_depth)
        & (left_reprojection_error <= config.max_stereo_reprojection_error_px)
        & (right_reprojection_error <= config.max_stereo_reprojection_error_px)
    )
    return TriangulationResult(
        points_left_3d=points_left_3d,
        valid_mask=valid,
        reprojection_error_px=reprojection_error,
        depth_left=depth_left,
        depth_right=depth_right,
    )


def estimate_motion_pnp(
    points_previous_3d: np.ndarray,
    points_current_px: np.ndarray,
    K_left: np.ndarray,
    config: StereoVOConfig,
) -> PnPResult:
    count = int(points_previous_3d.shape[0])
    empty_indices = np.empty(0, dtype=np.int64)
    empty_errors = np.empty(0, dtype=np.float64)
    if count < config.pnp_min_points:
        return PnPResult(
            False,
            f"only {count} valid 3D-2D correspondences",
            None,
            None,
            empty_indices,
            empty_errors,
        )

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.asarray(points_previous_3d, dtype=np.float64),
            np.asarray(points_current_px, dtype=np.float64),
            K_left,
            None,
            iterationsCount=config.pnp_iterations,
            reprojectionError=config.pnp_reprojection_error_px,
            confidence=config.pnp_confidence,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error as error:
        return PnPResult(False, f"solvePnPRansac error: {error}", None, None, empty_indices, empty_errors)

    if not success or inliers is None:
        return PnPResult(False, "solvePnPRansac failed", None, None, empty_indices, empty_errors)
    inlier_indices = inliers.reshape(-1).astype(np.int64)
    if inlier_indices.size < config.pnp_min_inliers:
        return PnPResult(
            False,
            f"only {inlier_indices.size} PnP inliers",
            None,
            None,
            inlier_indices,
            empty_errors,
        )

    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            points_previous_3d[inlier_indices],
            points_current_px[inlier_indices],
            K_left,
            None,
            rvec,
            tvec,
        )
    R_current_from_previous, _ = cv2.Rodrigues(rvec)
    t_current_from_previous = np.asarray(tvec, dtype=np.float64).reshape(3)
    projected, _ = cv2.projectPoints(
        points_previous_3d[inlier_indices],
        rvec,
        tvec,
        K_left,
        None,
    )
    errors = np.linalg.norm(
        projected.reshape(-1, 2) - points_current_px[inlier_indices], axis=1
    )
    return PnPResult(
        True,
        None,
        np.asarray(R_current_from_previous, dtype=np.float64),
        t_current_from_previous,
        inlier_indices,
        errors,
    )


def _pose_row(frame_idx: int, T_initial_from_camera: np.ndarray) -> PoseRow:
    return PoseRow(
        frame_idx=frame_idx,
        t=np.asarray(T_initial_from_camera[:3, 3], dtype=np.float64),
        q_xyzw=Rotation.from_matrix(T_initial_from_camera[:3, :3]).as_quat(),
    )


def _safe_distribution(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(values)),
        float(np.median(values)),
        float(np.percentile(values, 95)),
    )


def run_sequence_stereo_vo(
    sequence_dir: Path,
    calibration_root: Path,
    matcher: SharedFeatureMatcher,
    config: StereoVOConfig = StereoVOConfig(),
    max_frames: int | None = None,
    debug_frame_count: int = 3,
) -> SequenceVOResult:
    sequence = load_sequence(sequence_dir)
    frame_count = len(sequence.frame_ids)
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    if frame_count == 0:
        raise RuntimeError(f"Sequence has no frames: {sequence_dir}")

    session_id = session_id_from_sequence_name(sequence.sequence_name)
    calibration = load_session_stereo_calibration(calibration_root, session_id)
    if not (
        np.allclose(sequence.k2_l, calibration.K_left, rtol=0.0, atol=1e-6)
        and np.allclose(sequence.k2_r, calibration.K_right, rtol=0.0, atol=1e-6)
    ):
        raise RuntimeError(
            f"Sequence intrinsics do not match session calibration for {sequence.sequence_name}"
        )

    T_initial_from_previous = np.eye(4, dtype=np.float64)
    rows = [_pose_row(sequence.frame_ids[0], T_initial_from_previous)]
    diagnostics: list[FrameDiagnostic] = []
    debug_frames: list[DebugFrame] = []
    previous_left_features = matcher.extract(sequence.e2_l_images[0])

    for index in range(1, frame_count):
        previous_frame_idx = sequence.frame_ids[index - 1]
        frame_idx = sequence.frame_ids[index]
        previous_right_features = matcher.extract(sequence.e2_r_images[index - 1])
        current_left_features = matcher.extract(sequence.e2_l_images[index])
        stereo_matches = matcher.match_indices(previous_left_features, previous_right_features)
        temporal_matches = matcher.match_indices(previous_left_features, current_left_features)

        previous_left_keypoints = matcher.keypoints(previous_left_features)
        previous_right_keypoints = matcher.keypoints(previous_right_features)
        current_left_keypoints = matcher.keypoints(current_left_features)
        shared_left, shared_right, shared_current = intersect_left_keypoint_tracks(
            stereo_matches, temporal_matches
        )
        shared_previous_px = previous_left_keypoints[shared_left]
        shared_right_px = previous_right_keypoints[shared_right]
        shared_current_px = current_left_keypoints[shared_current]

        triangulation = triangulate_stereo_tracks(
            shared_previous_px,
            shared_right_px,
            calibration,
            config,
        )
        valid_indices = np.flatnonzero(triangulation.valid_mask)
        valid_points_3d = triangulation.points_left_3d[valid_indices]
        valid_current_px = shared_current_px[valid_indices]

        failure_reason: str | None = None
        if shared_left.size < config.min_shared_tracks:
            pnp = PnPResult(
                False,
                f"only {shared_left.size} shared stereo-temporal tracks",
                None,
                None,
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )
        else:
            pnp = estimate_motion_pnp(
                valid_points_3d,
                valid_current_px,
                calibration.K_left,
                config,
            )

        incremental_translation = float("nan")
        incremental_rotation = float("nan")
        if pnp.success:
            assert pnp.R_current_from_previous is not None
            assert pnp.t_current_from_previous is not None
            T_current_from_previous = np.eye(4, dtype=np.float64)
            T_current_from_previous[:3, :3] = pnp.R_current_from_previous
            T_current_from_previous[:3, 3] = pnp.t_current_from_previous

            # solvePnP returns X_current = R * X_previous + t, i.e.
            # T_current_from_previous. The challenge trajectory is a
            # camera-to-initial-frame pose, so invert this increment and
            # right-compose it: T_0_from_current = T_0_from_previous @
            # T_previous_from_current.
            T_previous_from_current = np.linalg.inv(T_current_from_previous)
            T_initial_from_current = T_initial_from_previous @ T_previous_from_current
            incremental_translation = float(np.linalg.norm(pnp.t_current_from_previous))
            incremental_rotation = float(
                np.degrees(Rotation.from_matrix(pnp.R_current_from_previous).magnitude())
            )
        else:
            # Identity incremental motion: retain the previous valid pose but
            # record the frame as a PnP failure in diagnostics.
            T_initial_from_current = T_initial_from_previous.copy()
            failure_reason = pnp.failure_reason

        rows.append(_pose_row(frame_idx, T_initial_from_current))
        pnp_mean, pnp_median, pnp_p95 = _safe_distribution(
            pnp.inlier_reprojection_errors_px
        )
        stereo_valid_errors = triangulation.reprojection_error_px[
            triangulation.valid_mask
        ]
        _, stereo_median, stereo_p95 = _safe_distribution(stereo_valid_errors)
        valid_depths = triangulation.depth_left[triangulation.valid_mask]
        _, depth_median, depth_p95 = _safe_distribution(valid_depths)
        diagnostics.append(
            FrameDiagnostic(
                previous_frame_idx=previous_frame_idx,
                frame_idx=frame_idx,
                stereo_matches=int(stereo_matches.shape[0]),
                temporal_matches=int(temporal_matches.shape[0]),
                shared_tracks=int(shared_left.size),
                valid_3d_points=int(valid_indices.size),
                pnp_success=pnp.success,
                failure_reason=failure_reason,
                pnp_inliers=int(pnp.inlier_indices.size),
                pnp_inlier_ratio=float(
                    pnp.inlier_indices.size / max(valid_indices.size, 1)
                ),
                pnp_reprojection_error_mean_px=pnp_mean,
                pnp_reprojection_error_median_px=pnp_median,
                pnp_reprojection_error_p95_px=pnp_p95,
                stereo_reprojection_error_median_px=stereo_median,
                stereo_reprojection_error_p95_px=stereo_p95,
                triangulated_depth_median_baselines=depth_median,
                triangulated_depth_p95_baselines=depth_p95,
                incremental_translation_baselines=incremental_translation,
                incremental_rotation_deg=incremental_rotation,
            )
        )

        if len(debug_frames) < debug_frame_count:
            stereo_left_px = previous_left_keypoints[stereo_matches[:, 0]]
            stereo_right_px = previous_right_keypoints[stereo_matches[:, 1]]
            temporal_previous_px = previous_left_keypoints[temporal_matches[:, 0]]
            temporal_current_px = current_left_keypoints[temporal_matches[:, 1]]
            if pnp.success:
                shared_inlier_indices = valid_indices[pnp.inlier_indices]
                pnp_previous_px = shared_previous_px[shared_inlier_indices]
                pnp_current_px = shared_current_px[shared_inlier_indices]
            else:
                pnp_previous_px = np.empty((0, 2), dtype=np.float64)
                pnp_current_px = np.empty((0, 2), dtype=np.float64)
            debug_frames.append(
                DebugFrame(
                    previous_frame_idx=previous_frame_idx,
                    frame_idx=frame_idx,
                    previous_left_path=sequence.e2_l_images[index - 1],
                    previous_right_path=sequence.e2_r_images[index - 1],
                    current_left_path=sequence.e2_l_images[index],
                    stereo_left_points=stereo_left_px,
                    stereo_right_points=stereo_right_px,
                    temporal_previous_points=temporal_previous_px,
                    temporal_current_points=temporal_current_px,
                    pnp_previous_points=pnp_previous_px,
                    pnp_current_points=pnp_current_px,
                )
            )

        T_initial_from_previous = T_initial_from_current
        previous_left_features = current_left_features

    successes = [diagnostic for diagnostic in diagnostics if diagnostic.pnp_success]
    pnp_inliers = [diagnostic.pnp_inliers for diagnostic in successes]
    summary: dict[str, Any] = {
        "sequence": sequence.sequence_name,
        "session_id": session_id,
        "num_frames": frame_count,
        "num_increments": len(diagnostics),
        "num_pnp_successes": len(successes),
        "pnp_success_rate": float(len(successes) / max(len(diagnostics), 1)),
        "median_pnp_inliers": float(np.median(pnp_inliers)) if pnp_inliers else 0.0,
        "failure_policy": "identity incremental motion / previous accumulated pose",
        "translation_unit": "arbitrary session stereo-baseline units",
    }
    return SequenceVOResult(rows, diagnostics, debug_frames, summary)
