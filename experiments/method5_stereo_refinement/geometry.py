"""Small, deterministic geometry core for session stereo refinement."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from imcpe.methods.cross_stereo_pnp.config import Method2Config
from imcpe.methods.cross_stereo_pnp.cross_pnp import estimate_cross_camera_pnp
from imcpe.methods.cross_stereo_pnp.triangulation import (
    E1StereoCalibration,
    triangulate_e1_stereo,
)

from .load_observations import SessionObservations


@dataclass(frozen=True)
class PreparedFrame:
    source_index: int
    sequence: str
    frame_id: int
    temporal_fraction: float
    x_e1l: np.ndarray
    x_e1r: np.ndarray
    x_e2l: np.ndarray
    K1_L: np.ndarray
    K1_R: np.ndarray
    K2_L: np.ndarray
    initial_pose: np.ndarray


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def corrected_stereo(
    parameters: np.ndarray, R_original: np.ndarray, t_original: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(parameters, dtype=np.float64).reshape(6)
    rotation = Rotation.from_rotvec(values[:3]).as_matrix() @ R_original
    norm = float(np.linalg.norm(t_original))
    direction = t_original / max(norm, np.finfo(np.float64).eps)
    direction = Rotation.from_rotvec(values[3:]).apply(direction)
    direction /= max(float(np.linalg.norm(direction)), np.finfo(np.float64).eps)
    return rotation, norm * direction


def correction_magnitudes_deg(parameters: np.ndarray) -> tuple[float, float]:
    values = np.asarray(parameters, dtype=np.float64).reshape(6)
    return (
        float(np.degrees(np.linalg.norm(values[:3]))),
        float(np.degrees(np.linalg.norm(values[3:]))),
    )


def rotation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(first @ second.T).magnitude()))


def direction_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(3)
    b = np.asarray(second, dtype=np.float64).reshape(3)
    cosine = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def calibration_from(
    observations: SessionObservations, R_LR: np.ndarray, t_LR: np.ndarray
) -> E1StereoCalibration:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R_LR
    transform[:3, 3] = t_LR
    return E1StereoCalibration(
        session_id=observations.session_id,
        K_left=observations.K1_L[0],
        K_right=observations.K1_R[0],
        R_LR=R_LR,
        t_LR_unit=t_LR,
        T_R_from_L=transform,
        suggested_min_depth=observations.suggested_min_depth,
        suggested_max_depth=observations.suggested_max_depth,
    )


def prepare_frames(
    observations: SessionObservations,
    method2_config: Method2Config = Method2Config(),
) -> tuple[list[PreparedFrame], list[dict[str, object]]]:
    """Freeze original-calibration PnP inliers before optimization.

    RANSAC is deliberately run once here, never inside the nonlinear objective.
    This makes the objective deterministic and prevents the candidate calibration
    from changing which correspondences it is judged on.
    """

    calibration = calibration_from(
        observations, observations.original_R_LR, observations.original_t_LR
    )
    frames: list[PreparedFrame] = []
    diagnostics: list[dict[str, object]] = []
    for index in range(observations.num_frames):
        point_slice = observations.frame_slice(index)
        x_left = observations.x_e1l[point_slice]
        x_right = observations.x_e1r[point_slice]
        x_e2 = observations.x_e2l[point_slice]
        triangulation = triangulate_e1_stereo(
            x_left, x_right, calibration, method2_config
        )
        valid = np.flatnonzero(triangulation.valid_mask)
        pnp = estimate_cross_camera_pnp(
            triangulation.points_e1_left_3d[valid],
            x_e2[valid],
            observations.K2_L[index],
            method2_config,
        )
        accepted = valid[pnp.inlier_indices] if pnp.success else np.empty(0, np.int64)
        diagnostics.append({
            "sequence": str(observations.sequences[index]),
            "frame_id": int(observations.frame_ids[index]),
            "cached_tracks": int(len(x_left)),
            "original_valid_triangulated": int(len(valid)),
            "original_pnp_success": bool(pnp.success),
            "fixed_pnp_inliers": int(len(accepted)),
        })
        if len(accepted) < method2_config.pnp_min_inliers:
            continue
        frames.append(
            PreparedFrame(
                source_index=index,
                sequence=str(observations.sequences[index]),
                frame_id=int(observations.frame_ids[index]),
                temporal_fraction=float(observations.temporal_fraction[index]),
                x_e1l=x_left[accepted],
                x_e1r=x_right[accepted],
                x_e2l=x_e2[accepted],
                K1_L=observations.K1_L[index],
                K1_R=observations.K1_R[index],
                K2_L=observations.K2_L[index],
                initial_pose=observations.T_e2_from_e1_initial[index],
            )
        )
    return frames, diagnostics


def triangulate_raw(
    x_left: np.ndarray,
    x_right: np.ndarray,
    K_left: np.ndarray,
    K_right: np.ndarray,
    R_LR: np.ndarray,
    t_LR: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left = cv2.undistortPoints(x_left.reshape(-1, 1, 2), K_left, None).reshape(-1, 2)
    right = cv2.undistortPoints(x_right.reshape(-1, 1, 2), K_right, None).reshape(-1, 2)
    P_left = np.hstack((np.eye(3), np.zeros((3, 1))))
    P_right = np.hstack((R_LR, t_LR.reshape(3, 1)))
    homogeneous = cv2.triangulatePoints(P_left, P_right, left.T, right.T)
    valid_w = np.abs(homogeneous[3]) > np.finfo(np.float64).eps
    points = np.full((len(x_left), 3), np.nan, dtype=np.float64)
    points[valid_w] = (homogeneous[:3, valid_w] / homogeneous[3, valid_w]).T
    right_points = (R_LR @ points.T).T + t_LR.reshape(1, 3)
    valid = (
        valid_w
        & np.isfinite(points).all(axis=1)
        & np.isfinite(right_points).all(axis=1)
        & (points[:, 2] > 0.0)
        & (right_points[:, 2] > 0.0)
    )
    return points, valid


def project(K: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = (K @ np.asarray(points, dtype=np.float64).T).T
    denominator = homogeneous[:, 2:3]
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.abs(denominator[:, 0]) > np.finfo(np.float64).eps
    result[valid] = homogeneous[valid, :2] / denominator[valid]
    return result


def refine_pose(
    points_e1: np.ndarray,
    points_e2_px: np.ndarray,
    K_e2: np.ndarray,
    initial_pose: np.ndarray,
) -> np.ndarray:
    rotation = np.asarray(initial_pose[:3, :3], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = np.asarray(initial_pose[:3, 3], dtype=np.float64).reshape(3, 1)
    valid = np.isfinite(points_e1).all(axis=1)
    if np.count_nonzero(valid) < 6:
        return initial_pose.copy()
    try:
        if hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(
                points_e1[valid], points_e2_px[valid], K_e2, None, rvec, tvec
            )
        else:
            success, rvec, tvec = cv2.solvePnP(
                points_e1[valid],
                points_e2_px[valid],
                K_e2,
                None,
                rvec,
                tvec,
                True,
                cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                return initial_pose.copy()
    except cv2.error:
        return initial_pose.copy()
    refined = np.eye(4, dtype=np.float64)
    refined[:3, :3] = cv2.Rodrigues(rvec)[0]
    refined[:3, 3] = np.asarray(tvec).reshape(3)
    return refined


def frame_residuals(
    frame: PreparedFrame,
    R_LR: np.ndarray,
    t_LR: np.ndarray,
    invalid_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points, valid = triangulate_raw(
        frame.x_e1l, frame.x_e1r, frame.K1_L, frame.K1_R, R_LR, t_LR
    )
    pose = refine_pose(points, frame.x_e2l, frame.K2_L, frame.initial_pose)
    right_points = (R_LR @ points.T).T + t_LR.reshape(1, 3)
    e2_points = (pose[:3, :3] @ points.T).T + pose[:3, 3]
    valid &= np.isfinite(right_points).all(axis=1) & np.isfinite(e2_points).all(axis=1)
    valid &= (right_points[:, 2] > 0.0) & (e2_points[:, 2] > 0.0)
    residuals = []
    for prediction, target in (
        (project(frame.K1_L, points), frame.x_e1l),
        (project(frame.K1_R, right_points), frame.x_e1r),
        (project(frame.K2_L, e2_points), frame.x_e2l),
    ):
        residual = prediction - target
        residual[~valid] = invalid_penalty
        residual[~np.isfinite(residual)] = invalid_penalty
        residuals.append(residual)
    return residuals[0], residuals[1], residuals[2], valid

