from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from imcpe.io_pose import PoseRow
from imcpe.methods.cross_stereo_pnp.data import (
    load_method2_inputs,
    session_id_from_sequence_name,
)

from .config import ReverseMethod2Config
from .matching import Method2BFeatureMatcher, intersect_e2_left_tracks
from .triangulation import (
    E2StereoCalibration,
    ReverseTriangulationResult,
    load_e2_stereo_calibration,
    triangulate_e2_stereo,
)


@dataclass(frozen=True)
class ReversePnPResult:
    success: bool
    failure_reason: str | None
    T_e1_from_e2: np.ndarray | None
    T_e2_from_e1: np.ndarray | None
    inlier_indices: np.ndarray
    inlier_reprojection_errors_px: np.ndarray
    refinement_attempted: bool
    refinement_succeeded: bool
    rotation_determinant: float
    rotation_orthogonality_error: float
    inverse_consistency_error: float


@dataclass
class Method2BFrameDiagnostic:
    frame_idx: int
    e2_stereo_matches: int
    e2_to_e1_matches: int
    shared_e2_left_tracks: int
    valid_triangulated_3d: int
    pnp_success: bool
    failure_reason: str | None
    pnp_inliers: int
    pnp_inlier_ratio: float
    pnp_reprojection_error_mean_px: float
    pnp_reprojection_error_median_px: float
    pnp_reprojection_error_p95_px: float
    stereo_reprojection_error_median_px: float
    stereo_reprojection_error_p95_px: float
    depth_median_baselines: float
    depth_p95_baselines: float
    triangulation_angle_median_deg: float
    triangulation_angle_p05_deg: float
    pnp_translation_magnitude_baselines: float
    common_translation_magnitude_baselines: float
    raw_rotation_determinant: float
    raw_rotation_orthogonality_error: float
    inverse_consistency_error: float
    refinement_attempted: bool
    refinement_succeeded: bool
    relative_rotation_change_deg: float
    previous_pose_fallback_used: bool


@dataclass(frozen=True)
class Method2BDebugFrame:
    frame_idx: int
    e2_left_path: Path
    e2_right_path: Path
    e1_left_path: Path
    stereo_e2_left_points: np.ndarray
    stereo_e2_right_points: np.ndarray
    cross_e2_left_points: np.ndarray
    cross_e1_left_points: np.ndarray
    surviving_e2_left_points: np.ndarray
    surviving_e1_left_points: np.ndarray
    surviving_inlier_mask: np.ndarray


@dataclass(frozen=True)
class Method2BSequenceResult:
    rows: list[PoseRow]
    diagnostics: list[Method2BFrameDiagnostic]
    debug_frames: list[Method2BDebugFrame]
    summary: dict[str, Any]
    raw_T_e1_from_e2: np.ndarray
    raw_T_e2_from_e1: np.ndarray
    relative_T_initial_from_current: np.ndarray
    raw_success: np.ndarray


def _empty_pnp(reason: str, inliers: np.ndarray | None = None) -> ReversePnPResult:
    return ReversePnPResult(
        success=False,
        failure_reason=reason,
        T_e1_from_e2=None,
        T_e2_from_e1=None,
        inlier_indices=(
            np.empty(0, dtype=np.int64)
            if inliers is None
            else np.asarray(inliers, dtype=np.int64)
        ),
        inlier_reprojection_errors_px=np.empty(0, dtype=np.float64),
        refinement_attempted=False,
        refinement_succeeded=False,
        rotation_determinant=float("nan"),
        rotation_orthogonality_error=float("nan"),
        inverse_consistency_error=float("nan"),
    )


def estimate_reverse_pnp(
    points_e2_3d: np.ndarray,
    points_e1_left_px: np.ndarray,
    K_e1_left: np.ndarray,
    config: ReverseMethod2Config,
) -> ReversePnPResult:
    """Estimate `T_E1_from_E2`, then explicitly invert to `T_E2_from_E1`."""

    count = int(np.asarray(points_e2_3d).shape[0])
    if count < config.pnp_min_points:
        return _empty_pnp(f"only {count} valid 3D-2D correspondences")
    object_points = np.asarray(points_e2_3d, dtype=np.float64)
    image_points = np.asarray(points_e1_left_px, dtype=np.float64)
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            np.asarray(K_e1_left, dtype=np.float64),
            None,
            iterationsCount=config.pnp_iterations,
            reprojectionError=config.pnp_reprojection_error_px,
            confidence=config.pnp_confidence,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error as error:
        return _empty_pnp(f"solvePnPRansac error: {error}")
    if not success or inliers is None:
        return _empty_pnp("solvePnPRansac failed")
    inlier_indices = inliers.reshape(-1).astype(np.int64)
    if inlier_indices.size < config.pnp_min_inliers:
        return _empty_pnp(
            f"only {inlier_indices.size} PnP inliers", inlier_indices
        )

    refinement_attempted = bool(config.refine_lm and hasattr(cv2, "solvePnPRefineLM"))
    refinement_succeeded = False
    if refinement_attempted:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[inlier_indices],
                image_points[inlier_indices],
                np.asarray(K_e1_left, dtype=np.float64),
                None,
                rvec,
                tvec,
            )
            refinement_succeeded = True
        except cv2.error:
            refinement_succeeded = False

    rotation, _ = cv2.Rodrigues(rvec)
    transform_e1_from_e2 = np.eye(4, dtype=np.float64)
    transform_e1_from_e2[:3, :3] = rotation
    transform_e1_from_e2[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    transform_e2_from_e1 = np.linalg.inv(transform_e1_from_e2)
    inverse_error = float(
        np.max(
            np.abs(
                transform_e1_from_e2 @ transform_e2_from_e1
                - np.eye(4, dtype=np.float64)
            )
        )
    )
    if inverse_error > 1e-8:
        return _empty_pnp(f"transform inverse check failed: {inverse_error:.3e}")

    projected, _ = cv2.projectPoints(
        object_points[inlier_indices],
        rvec,
        tvec,
        np.asarray(K_e1_left, dtype=np.float64),
        None,
    )
    errors = np.linalg.norm(
        projected.reshape(-1, 2) - image_points[inlier_indices], axis=1
    )
    determinant = float(np.linalg.det(rotation))
    orthogonality = float(
        np.max(np.abs(rotation.T @ rotation - np.eye(3, dtype=np.float64)))
    )
    return ReversePnPResult(
        success=True,
        failure_reason=None,
        T_e1_from_e2=transform_e1_from_e2,
        T_e2_from_e1=transform_e2_from_e1,
        inlier_indices=inlier_indices,
        inlier_reprojection_errors_px=errors,
        refinement_attempted=refinement_attempted,
        refinement_succeeded=refinement_succeeded,
        rotation_determinant=determinant,
        rotation_orthogonality_error=orthogonality,
        inverse_consistency_error=inverse_error,
    )


def _pose_row(frame_idx: int, transform: np.ndarray) -> PoseRow:
    return PoseRow(
        frame_idx=frame_idx,
        t=np.asarray(transform[:3, 3], dtype=np.float64),
        q_xyzw=Rotation.from_matrix(transform[:3, :3]).as_quat(),
    )


def _distribution(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(values)),
        float(np.median(values)),
        float(np.percentile(values, 95)),
    )


def _p05(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, 5)) if values.size else float("nan")


def _rotation_change_degrees(previous: np.ndarray, current: np.ndarray) -> float:
    relative = previous[:3, :3].T @ current[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _maximum_translation_jump_ratio(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    ratios: list[float] = []
    for previous, current in zip(values[:-1], values[1:], strict=True):
        if not np.isfinite(previous) or not np.isfinite(current):
            continue
        lower = max(min(previous, current), 1e-6)
        ratios.append(max(previous, current) / lower)
    return float(max(ratios)) if ratios else float("nan")


def _validate_intrinsics(
    calibration: E2StereoCalibration,
    K_left: np.ndarray,
    K_right: np.ndarray,
    sequence_name: str,
) -> None:
    if not (
        np.allclose(calibration.K_left, K_left, rtol=0.0, atol=1e-6)
        and np.allclose(calibration.K_right, K_right, rtol=0.0, atol=1e-6)
    ):
        raise RuntimeError(
            f"E2 intrinsics do not match session calibration for {sequence_name}"
        )


def run_sequence_method2b(
    *,
    sequence_dir: Path,
    calibration_root: Path,
    matcher: Method2BFeatureMatcher,
    config: ReverseMethod2Config = ReverseMethod2Config(),
    max_frames: int | None = None,
    debug_frame_count: int = 3,
) -> Method2BSequenceResult:
    """Run pure independent reverse PnP without opening ground truth."""

    inputs = load_method2_inputs(sequence_dir)
    frame_count = len(inputs.frame_ids)
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    if frame_count <= 0:
        raise RuntimeError(f"No frames selected in {sequence_dir}")
    session_id = session_id_from_sequence_name(inputs.sequence_name)
    calibration = load_e2_stereo_calibration(calibration_root, session_id)
    _validate_intrinsics(calibration, inputs.k2_l, inputs.k2_r, inputs.sequence_name)
    if calibration.valid is False:
        print(
            f"WARNING: session_{session_id} E2 calibration is invalid: "
            + "; ".join(calibration.validity_failures),
            flush=True,
        )

    raw_e1_from_e2 = np.full((frame_count, 4, 4), np.nan, dtype=np.float64)
    raw_e2_from_e1 = np.full((frame_count, 4, 4), np.nan, dtype=np.float64)
    raw_success = np.zeros(frame_count, dtype=bool)
    diagnostics: list[Method2BFrameDiagnostic] = []
    debug_frames: list[Method2BDebugFrame] = []

    for index in range(frame_count):
        frame_id = inputs.frame_ids[index]
        # E2-L is extracted exactly once and shared by both match operations.
        e2_left_features = matcher.extract(inputs.e2_l_images[index])
        e2_right_features = matcher.extract(inputs.e2_r_images[index])
        e1_left_features = matcher.extract(inputs.e1_l_images[index])
        stereo_matches = matcher.match_indices(e2_left_features, e2_right_features)
        cross_matches = matcher.match_indices(e2_left_features, e1_left_features)
        e2_left_keypoints = matcher.keypoints(e2_left_features)
        e2_right_keypoints = matcher.keypoints(e2_right_features)
        e1_left_keypoints = matcher.keypoints(e1_left_features)
        shared_left, shared_right, shared_e1 = intersect_e2_left_tracks(
            stereo_matches, cross_matches
        )
        shared_e2_left_px = e2_left_keypoints[shared_left]
        shared_e2_right_px = e2_right_keypoints[shared_right]
        shared_e1_left_px = e1_left_keypoints[shared_e1]
        triangulation = triangulate_e2_stereo(
            shared_e2_left_px,
            shared_e2_right_px,
            calibration,
            config,
        )
        valid_indices = np.flatnonzero(triangulation.valid_mask)
        points_3d = triangulation.points_e2_left_3d[valid_indices]
        points_e1 = shared_e1_left_px[valid_indices]
        if shared_left.size < config.min_shared_tracks:
            pnp = _empty_pnp(
                f"only {shared_left.size} shared E2-left feature tracks"
            )
        else:
            pnp = estimate_reverse_pnp(points_3d, points_e1, inputs.k1_l, config)
        if pnp.success:
            assert pnp.T_e1_from_e2 is not None and pnp.T_e2_from_e1 is not None
            raw_e1_from_e2[index] = pnp.T_e1_from_e2
            raw_e2_from_e1[index] = pnp.T_e2_from_e1
            raw_success[index] = True

        pnp_mean, pnp_median, pnp_p95 = _distribution(
            pnp.inlier_reprojection_errors_px
        )
        valid_stereo_errors = triangulation.reprojection_error_px[
            triangulation.valid_mask
        ]
        _, stereo_median, stereo_p95 = _distribution(valid_stereo_errors)
        valid_depths = triangulation.depth_left[triangulation.valid_mask]
        _, depth_median, depth_p95 = _distribution(valid_depths)
        valid_angles = triangulation.triangulation_angle_deg[
            triangulation.valid_mask
        ]
        diagnostics.append(
            Method2BFrameDiagnostic(
                frame_idx=frame_id,
                e2_stereo_matches=int(stereo_matches.shape[0]),
                e2_to_e1_matches=int(cross_matches.shape[0]),
                shared_e2_left_tracks=int(shared_left.size),
                valid_triangulated_3d=int(valid_indices.size),
                pnp_success=pnp.success,
                failure_reason=pnp.failure_reason,
                pnp_inliers=int(pnp.inlier_indices.size),
                pnp_inlier_ratio=float(
                    pnp.inlier_indices.size / max(valid_indices.size, 1)
                ),
                pnp_reprojection_error_mean_px=pnp_mean,
                pnp_reprojection_error_median_px=pnp_median,
                pnp_reprojection_error_p95_px=pnp_p95,
                stereo_reprojection_error_median_px=stereo_median,
                stereo_reprojection_error_p95_px=stereo_p95,
                depth_median_baselines=depth_median,
                depth_p95_baselines=depth_p95,
                triangulation_angle_median_deg=(
                    float(np.median(valid_angles))
                    if valid_angles.size
                    else float("nan")
                ),
                triangulation_angle_p05_deg=_p05(valid_angles),
                pnp_translation_magnitude_baselines=(
                    float(np.linalg.norm(pnp.T_e1_from_e2[:3, 3]))
                    if pnp.T_e1_from_e2 is not None
                    else float("nan")
                ),
                common_translation_magnitude_baselines=(
                    float(np.linalg.norm(pnp.T_e2_from_e1[:3, 3]))
                    if pnp.T_e2_from_e1 is not None
                    else float("nan")
                ),
                raw_rotation_determinant=pnp.rotation_determinant,
                raw_rotation_orthogonality_error=pnp.rotation_orthogonality_error,
                inverse_consistency_error=pnp.inverse_consistency_error,
                refinement_attempted=pnp.refinement_attempted,
                refinement_succeeded=pnp.refinement_succeeded,
                relative_rotation_change_deg=float("nan"),
                previous_pose_fallback_used=False,
            )
        )

        if len(debug_frames) < debug_frame_count:
            inlier_mask = np.zeros(valid_indices.size, dtype=bool)
            if pnp.success:
                inlier_mask[pnp.inlier_indices] = True
            debug_frames.append(
                Method2BDebugFrame(
                    frame_idx=frame_id,
                    e2_left_path=inputs.e2_l_images[index],
                    e2_right_path=inputs.e2_r_images[index],
                    e1_left_path=inputs.e1_l_images[index],
                    stereo_e2_left_points=e2_left_keypoints[stereo_matches[:, 0]],
                    stereo_e2_right_points=e2_right_keypoints[stereo_matches[:, 1]],
                    cross_e2_left_points=e2_left_keypoints[cross_matches[:, 0]],
                    cross_e1_left_points=e1_left_keypoints[cross_matches[:, 1]],
                    surviving_e2_left_points=shared_e2_left_px[valid_indices],
                    surviving_e1_left_points=shared_e1_left_px[valid_indices],
                    surviving_inlier_mask=inlier_mask,
                )
            )

    if not raw_success[0]:
        raise RuntimeError(
            "Method 2B frame-zero PnP failed; relative normalization is undefined"
        )

    reference = raw_e2_from_e1[0]
    relative_transforms = np.full_like(raw_e2_from_e1, np.nan)
    rows: list[PoseRow] = []
    previous_valid_relative = np.eye(4, dtype=np.float64)
    previous_output: np.ndarray | None = None
    for index, frame_id in enumerate(inputs.frame_ids[:frame_count]):
        if raw_success[index]:
            # C(t)=T_E2(t)_from_E1. The challenge stores T_E2(0)_from_E2(t).
            relative = reference @ np.linalg.inv(raw_e2_from_e1[index])
            previous_valid_relative = relative
        else:
            relative = previous_valid_relative.copy()
            diagnostics[index].previous_pose_fallback_used = True
        relative_transforms[index] = relative
        rows.append(_pose_row(frame_id, relative))
        diagnostics[index].relative_rotation_change_deg = (
            0.0
            if previous_output is None
            else _rotation_change_degrees(previous_output, relative)
        )
        previous_output = relative

    frame_zero_error = float(
        np.max(np.abs(relative_transforms[0] - np.eye(4, dtype=np.float64)))
    )
    if frame_zero_error > 1e-9:
        raise RuntimeError(
            f"Method 2B frame-zero relative pose is not identity: {frame_zero_error}"
        )
    rotations = relative_transforms[:, :3, :3]
    determinant_error = float(np.max(np.abs(np.linalg.det(rotations) - 1.0)))
    orthogonality_error = float(
        np.max(
            np.abs(
                np.einsum("nji,njk->nik", rotations, rotations)
                - np.eye(3, dtype=np.float64)
            )
        )
    )
    if determinant_error > 1e-6 or orthogonality_error > 1e-6:
        raise RuntimeError(
            "Method 2B generated invalid SO(3) rotations: "
            f"det={determinant_error}, orth={orthogonality_error}"
        )

    successful = [item for item in diagnostics if item.pnp_success]
    shared = np.asarray(
        [item.shared_e2_left_tracks for item in diagnostics], dtype=np.float64
    )
    inliers = np.asarray([item.pnp_inliers for item in successful], dtype=np.float64)
    ratios = np.asarray(
        [item.pnp_inlier_ratio for item in successful], dtype=np.float64
    )
    reprojection = np.asarray(
        [item.pnp_reprojection_error_median_px for item in successful],
        dtype=np.float64,
    )
    translations = np.asarray(
        [item.common_translation_magnitude_baselines for item in diagnostics],
        dtype=np.float64,
    )
    success_rate = float(np.mean(raw_success))
    median_shared = float(np.median(shared)) if shared.size else 0.0
    median_reprojection = (
        float(np.nanmedian(reprojection)) if reprojection.size else float("nan")
    )
    maximum_jump = _maximum_translation_jump_ratio(translations)
    stop_reasons: list[str] = []
    if success_rate < config.stop_min_success_rate:
        stop_reasons.append(
            f"PnP success rate {success_rate:.3f} < {config.stop_min_success_rate:.3f}"
        )
    if median_shared < config.stop_min_median_shared_tracks:
        stop_reasons.append(
            f"median shared tracks {median_shared:.1f} < "
            f"{config.stop_min_median_shared_tracks:.1f}"
        )
    if (
        not np.isfinite(median_reprojection)
        or median_reprojection > config.stop_max_median_reprojection_error_px
    ):
        stop_reasons.append(
            f"median PnP reprojection {median_reprojection:.3f}px exceeds threshold"
        )
    if np.isfinite(maximum_jump) and maximum_jump > config.stop_max_translation_jump_ratio:
        stop_reasons.append(
            f"maximum translation jump ratio {maximum_jump:.2f} exceeds "
            f"{config.stop_max_translation_jump_ratio:.2f}"
        )
    summary: dict[str, Any] = {
        "sequence": inputs.sequence_name,
        "session_id": session_id,
        "num_frames": frame_count,
        "num_pnp_successes": int(np.count_nonzero(raw_success)),
        "pnp_success_rate": success_rate,
        "median_shared_3d_2d_tracks": median_shared,
        "median_pnp_inliers": float(np.median(inliers)) if inliers.size else 0.0,
        "median_pnp_inlier_ratio": float(np.median(ratios)) if ratios.size else 0.0,
        "median_pnp_reprojection_error_px": median_reprojection,
        "median_common_translation_magnitude_baselines": float(
            np.nanmedian(translations)
        ),
        "maximum_common_translation_jump_ratio": maximum_jump,
        "num_previous_pose_fallbacks": int(
            sum(item.previous_pose_fallback_used for item in diagnostics)
        ),
        "calibration_path": str(calibration.source_path),
        "calibration_valid": calibration.valid,
        "calibration_validity_failures": list(calibration.validity_failures),
        "maximum_inverse_consistency_error": float(
            np.nanmax([item.inverse_consistency_error for item in successful])
            if successful
            else np.nan
        ),
        "maximum_relative_rotation_determinant_error": determinant_error,
        "maximum_relative_rotation_orthogonality_error": orthogonality_error,
        "frame_zero_identity_error": frame_zero_error,
        "translation_unit": "arbitrary fixed E2 stereo-baseline units",
        "failure_policy": "previous valid Method-2B relative pose",
        "geometry_stop_recommended": bool(stop_reasons),
        "geometry_stop_reasons": stop_reasons,
    }
    return Method2BSequenceResult(
        rows=rows,
        diagnostics=diagnostics,
        debug_frames=debug_frames,
        summary=summary,
        raw_T_e1_from_e2=raw_e1_from_e2,
        raw_T_e2_from_e1=raw_e2_from_e1,
        relative_T_initial_from_current=relative_transforms,
        raw_success=raw_success,
    )
