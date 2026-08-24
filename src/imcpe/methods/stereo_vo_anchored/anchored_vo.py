from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from imcpe.io_pose import PoseRow

from .anchor_estimator import CrossCameraAnchorCache
from .config import AnchorThresholds, FusionConfig
from .rotation_fusion import fuse_rotation_so3, rotation_angle_deg, so3_errors


@dataclass(frozen=True)
class FusionFrameDiagnostic:
    list_index: int
    previous_frame_idx: int
    frame_idx: int
    anchor_success: bool
    anchor_confidence_valid: bool
    anchor_scheduled: bool
    anchor_applied: bool
    anchor_rejection_reason: str | None
    angular_disagreement_deg: float
    applied_correction_deg: float
    local_translation_increment: float
    local_rotation_increment_deg: float
    anchor_matches: int
    anchor_essential_inliers: int
    anchor_inlier_ratio: float
    anchor_cheirality_fraction: float
    anchor_median_sampson_error_normalized: float


@dataclass(frozen=True)
class AnchoredTrajectoryResult:
    rows: list[PoseRow]
    diagnostics: list[FusionFrameDiagnostic]
    summary: dict[str, Any]


def _pose_to_matrix(row: PoseRow) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(row.q_xyzw).as_matrix()
    matrix[:3, 3] = row.t
    return matrix


def _matrix_to_pose(frame_idx: int, matrix: np.ndarray) -> PoseRow:
    return PoseRow(
        frame_idx=frame_idx,
        t=np.asarray(matrix[:3, 3], dtype=np.float64),
        q_xyzw=Rotation.from_matrix(matrix[:3, :3]).as_quat(),
    )


def _confidence_rejection_reason(
    cache: CrossCameraAnchorCache,
    index: int,
    thresholds: AnchorThresholds,
) -> str | None:
    if not cache.success[index]:
        return "anchor estimation failed"
    reasons: list[str] = []
    if cache.num_matches[index] < thresholds.min_matches:
        reasons.append("matches below training-derived threshold")
    if cache.num_essential_inliers[index] < thresholds.min_essential_inliers:
        reasons.append("essential inliers below training-derived threshold")
    if cache.essential_inlier_ratio[index] < thresholds.min_inlier_ratio:
        reasons.append("inlier ratio below training-derived threshold")
    if cache.cheirality_fraction[index] < thresholds.min_cheirality_fraction:
        reasons.append("cheirality below training-derived threshold")
    if not np.isfinite(cache.median_sampson_error_normalized[index]):
        reasons.append("non-finite Sampson error")
    elif (
        cache.median_sampson_error_normalized[index]
        > thresholds.max_median_sampson_error_normalized
    ):
        reasons.append("Sampson error above training-derived threshold")
    if not np.isfinite(cache.R_anchor_rel[index]).all():
        reasons.append("non-finite relative anchor rotation")
    return "; ".join(reasons) if reasons else None


def fuse_anchored_trajectory(
    method1_rows: list[PoseRow],
    anchors: CrossCameraAnchorCache,
    thresholds: AnchorThresholds,
    config: FusionConfig,
) -> AnchoredTrajectoryResult:
    """Re-integrate Method 1 local increments with gated rotation anchors.

    No ground-truth pose is accepted by this function. Local increments are
    recovered exactly from Method 1's saved `T_0_from_t` trajectory.
    """

    if not method1_rows:
        raise ValueError("Method 1 trajectory is empty")
    method1_ids = np.asarray([row.frame_idx for row in method1_rows], dtype=np.int64)
    if not np.array_equal(method1_ids, anchors.frame_ids):
        raise RuntimeError("Method 1 and anchor frame IDs differ")
    if not np.allclose(anchors.R_anchor_rel[0], np.eye(3), atol=1e-9, rtol=0.0):
        raise RuntimeError("R_anchor_rel(0) is not identity")

    method1_matrices = np.asarray([_pose_to_matrix(row) for row in method1_rows])
    if not np.allclose(method1_matrices[0], np.eye(4), atol=1e-8, rtol=0.0):
        raise RuntimeError("Method 1 frame-zero pose is not identity")

    confidence_valid = anchors.confidence_valid(thresholds)
    corrected_previous = np.eye(4, dtype=np.float64)
    corrected_matrices = [corrected_previous.copy()]
    output_rows = [_matrix_to_pose(method1_rows[0].frame_idx, corrected_previous)]
    diagnostics: list[FusionFrameDiagnostic] = []
    corrections_applied = 0
    maximum_det_error = 0.0
    maximum_orthogonality_error = 0.0

    for index in range(1, len(method1_rows)):
        method1_previous = method1_matrices[index - 1]
        method1_current = method1_matrices[index]
        local_previous_from_current = np.linalg.inv(method1_previous) @ method1_current
        predicted_current = corrected_previous @ local_previous_from_current
        predicted_rotation = predicted_current[:3, :3]

        rejection_reason = _confidence_rejection_reason(anchors, index, thresholds)
        scheduled = index % config.anchor_interval == 0
        disagreement = float("nan")
        correction_angle = 0.0
        applied = False
        if anchors.success[index] and np.isfinite(anchors.R_anchor_rel[index]).all():
            disagreement = rotation_angle_deg(
                predicted_rotation, anchors.R_anchor_rel[index]
            )
        if scheduled and confidence_valid[index]:
            if (
                config.max_anchor_disagreement_deg is not None
                and disagreement > config.max_anchor_disagreement_deg
            ):
                rejection_reason = "angular disagreement exceeds configured maximum"
            elif config.alpha > 0.0:
                corrected_rotation, disagreement, correction_angle = fuse_rotation_so3(
                    predicted_rotation,
                    anchors.R_anchor_rel[index],
                    config.alpha,
                    config.max_correction_deg,
                )
                predicted_current[:3, :3] = corrected_rotation
                applied = correction_angle > 0.0
                corrections_applied += int(applied)
                rejection_reason = None
            else:
                rejection_reason = None

        determinant_error, orthogonality_error = so3_errors(predicted_current[:3, :3])
        maximum_det_error = max(maximum_det_error, determinant_error)
        maximum_orthogonality_error = max(
            maximum_orthogonality_error, orthogonality_error
        )
        if determinant_error > 1e-8 or orthogonality_error > 1e-8:
            raise RuntimeError(f"Invalid corrected SO(3) rotation at frame {method1_ids[index]}")

        local_rotation_deg = float(
            np.degrees(
                Rotation.from_matrix(local_previous_from_current[:3, :3]).magnitude()
            )
        )
        diagnostics.append(
            FusionFrameDiagnostic(
                list_index=index,
                previous_frame_idx=int(method1_ids[index - 1]),
                frame_idx=int(method1_ids[index]),
                anchor_success=bool(anchors.success[index]),
                anchor_confidence_valid=bool(confidence_valid[index]),
                anchor_scheduled=scheduled,
                anchor_applied=applied,
                anchor_rejection_reason=rejection_reason,
                angular_disagreement_deg=disagreement,
                applied_correction_deg=correction_angle,
                local_translation_increment=float(
                    np.linalg.norm(local_previous_from_current[:3, 3])
                ),
                local_rotation_increment_deg=local_rotation_deg,
                anchor_matches=int(anchors.num_matches[index]),
                anchor_essential_inliers=int(anchors.num_essential_inliers[index]),
                anchor_inlier_ratio=float(anchors.essential_inlier_ratio[index]),
                anchor_cheirality_fraction=float(anchors.cheirality_fraction[index]),
                anchor_median_sampson_error_normalized=float(
                    anchors.median_sampson_error_normalized[index]
                ),
            )
        )
        corrected_previous = predicted_current
        corrected_matrices.append(corrected_previous.copy())
        output_rows.append(_matrix_to_pose(int(method1_ids[index]), corrected_previous))

    corrected_array = np.asarray(corrected_matrices)
    alpha_zero_max_matrix_error = (
        float(np.max(np.abs(corrected_array - method1_matrices)))
        if config.alpha == 0.0
        else float("nan")
    )
    if config.alpha == 0.0 and alpha_zero_max_matrix_error > 1e-8:
        raise RuntimeError(
            "alpha=0 failed to reproduce Method 1: max matrix error "
            f"{alpha_zero_max_matrix_error:.3e}"
        )

    summary: dict[str, Any] = {
        "sequence": anchors.sequence_name,
        "num_frames": len(method1_rows),
        "num_anchor_estimation_successes": int(np.count_nonzero(anchors.success)),
        "num_confidence_valid_anchors": int(np.count_nonzero(confidence_valid)),
        "num_scheduled_anchor_frames": int(
            sum(index % config.anchor_interval == 0 for index in range(1, len(method1_rows)))
        ),
        "num_corrections_applied": corrections_applied,
        "alpha_zero_max_matrix_error": alpha_zero_max_matrix_error,
        "maximum_rotation_determinant_error": maximum_det_error,
        "maximum_rotation_orthogonality_error": maximum_orthogonality_error,
        "fusion_config": asdict(config),
        "anchor_thresholds": asdict(thresholds),
    }
    return AnchoredTrajectoryResult(output_rows, diagnostics, summary)

