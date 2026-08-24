from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from imcpe.io_pose import PoseRow


def homogeneous(extrinsic_3x4: np.ndarray) -> np.ndarray:
    matrix = np.asarray(extrinsic_3x4, dtype=np.float64)
    if matrix.shape != (3, 4):
        raise ValueError(f"expected 3x4 extrinsic, got {matrix.shape}")
    result = np.eye(4, dtype=np.float64)
    result[:3, :4] = matrix
    return result


def invert_se3(T_a_from_b: np.ndarray) -> np.ndarray:
    T = np.asarray(T_a_from_b, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = R.T
    result[:3, 3] = -R.T @ t
    return result


def camera_center_world(T_cam_from_world: np.ndarray) -> np.ndarray:
    T = np.asarray(T_cam_from_world, dtype=np.float64)
    return -T[:3, :3].T @ T[:3, 3]


def relative_camera_transform(
    T_b_from_world: np.ndarray, T_a_from_world: np.ndarray
) -> np.ndarray:
    """Return ``T_B_from_A = T_B_from_W @ inverse(T_A_from_W)``."""

    return np.asarray(T_b_from_world) @ invert_se3(T_a_from_world)


def scaled_camera_extrinsics(
    transforms_cam_from_world: np.ndarray, scale: float
) -> np.ndarray:
    result = np.asarray(transforms_cam_from_world, dtype=np.float64).copy()
    result[..., :3, 3] *= float(scale)
    return result


def challenge_relative_from_cross(
    T_cross_frame0: np.ndarray, T_cross_current: np.ndarray
) -> np.ndarray:
    """Convert raw cross-camera poses to challenge ``T_E2(0)_from_E2(t)``.

    Both raw poses are ``T_E2(t)_from_E1`` and E1 is the fixed reference, so
    ``T_rel(t) = T_cross(0) @ inverse(T_cross(t))``.  This is deliberately not
    ``inverse(T_cross(0)) @ T_cross(t)``.
    """

    return np.asarray(T_cross_frame0) @ invert_se3(T_cross_current)


def transform_to_pose_row(frame_id: int, T_initial_from_current: np.ndarray) -> PoseRow:
    T = np.asarray(T_initial_from_current, dtype=np.float64)
    return PoseRow(
        frame_idx=int(frame_id),
        t=T[:3, 3].copy(),
        q_xyzw=Rotation.from_matrix(T[:3, :3]).as_quat(),
    )


def rotation_angle_degrees(R: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(R).magnitude()))


def rotation_quality(R: np.ndarray) -> tuple[float, float]:
    matrix = np.asarray(R, dtype=np.float64)
    orthogonality = float(np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(matrix))
    return orthogonality, determinant
