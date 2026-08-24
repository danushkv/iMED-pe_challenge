from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    relative = np.asarray(R_a, dtype=np.float64).T @ np.asarray(R_b, dtype=np.float64)
    return float(np.degrees(Rotation.from_matrix(relative).magnitude()))


def fuse_rotation_so3(
    R_predicted: np.ndarray,
    R_anchor: np.ndarray,
    alpha: float,
    max_correction_deg: float,
) -> tuple[np.ndarray, float, float]:
    """Move a predicted rotation toward an anchor on SO(3).

    Returns `(R_corrected, disagreement_deg, applied_correction_deg)`.
    The correction itself, after multiplication by alpha, is clipped.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    R_predicted = np.asarray(R_predicted, dtype=np.float64)
    R_anchor = np.asarray(R_anchor, dtype=np.float64)
    error = Rotation.from_matrix(R_predicted.T @ R_anchor)
    error_rotvec = error.as_rotvec()
    disagreement_rad = float(np.linalg.norm(error_rotvec))
    correction_rotvec = alpha * error_rotvec
    correction_rad = float(np.linalg.norm(correction_rotvec))
    maximum_rad = float(np.radians(max_correction_deg))
    if correction_rad > maximum_rad:
        correction_rotvec *= maximum_rad / correction_rad
        correction_rad = maximum_rad
    corrected = R_predicted @ Rotation.from_rotvec(correction_rotvec).as_matrix()
    return corrected, float(np.degrees(disagreement_rad)), float(np.degrees(correction_rad))


def so3_errors(R: np.ndarray) -> tuple[float, float]:
    R = np.asarray(R, dtype=np.float64)
    determinant_error = abs(float(np.linalg.det(R)) - 1.0)
    orthogonality_error = float(np.max(np.abs(R.T @ R - np.eye(3))))
    return determinant_error, orthogonality_error

