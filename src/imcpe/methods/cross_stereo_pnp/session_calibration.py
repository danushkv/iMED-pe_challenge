"""Session-level, scale-free Endoscope-1 stereo self-calibration.

This is an isolated E1 adaptation of the validated Method 1 calibration
strategy. It reads only K.txt and synchronized images, never pose.txt.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import SessionCalibrationConfig
from .data import image_path_map, read_intrinsics_only
from .feature_matching import Method2FeatureMatcher


@dataclass(frozen=True)
class CalibrationFrame:
    split: str
    sequence_name: str
    frame_id: int
    left_path: Path
    right_path: Path


@dataclass
class PairEstimate:
    frame: CalibrationFrame
    num_matches: int
    num_essential_inliers: int = 0
    essential_inlier_ratio: float = 0.0
    num_cheirality_inliers: int = 0
    cheirality_fraction: float = 0.0
    success: bool = False
    failure_reason: str | None = None
    E: np.ndarray | None = None
    R: np.ndarray | None = None
    t_unit: np.ndarray | None = None
    left_px: np.ndarray | None = None
    right_px: np.ndarray | None = None
    left_norm: np.ndarray | None = None
    right_norm: np.ndarray | None = None
    essential_mask: np.ndarray | None = None
    cheirality_mask: np.ndarray | None = None


@dataclass(frozen=True)
class EssentialEstimate:
    E: np.ndarray
    R: np.ndarray
    t_unit: np.ndarray
    essential_mask: np.ndarray
    cheirality_mask: np.ndarray


def _normalize_session_id(value: str | int) -> str:
    text = str(value)
    if text.startswith("session_"):
        text = text[len("session_"):]
    if not text.isdigit():
        raise ValueError(f"Expected numeric session ID, got {value!r}")
    return text.zfill(3)


def discover_session_ids(
    data_root: Path, splits: tuple[str, ...] = ("train", "test")
) -> list[str]:
    identifiers: set[str] = set()
    for split in splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            continue
        for sequence_dir in split_dir.iterdir():
            if not sequence_dir.is_dir() or not sequence_dir.name.startswith("session_"):
                continue
            parts = sequence_dir.name.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                identifiers.add(parts[1].zfill(3))
    return sorted(identifiers)


def _discover_frames(
    data_root: Path,
    session_id: str,
    splits: tuple[str, ...],
) -> tuple[list[list[CalibrationFrame]], np.ndarray, np.ndarray]:
    groups: list[list[CalibrationFrame]] = []
    K_left: np.ndarray | None = None
    K_right: np.ndarray | None = None
    prefix = f"session_{session_id}_"
    for split in splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            continue
        sequence_dirs = sorted(
            path
            for path in split_dir.iterdir()
            if path.is_dir() and path.name.startswith(prefix)
        )
        for sequence_dir in sequence_dirs:
            intrinsics = read_intrinsics_only(sequence_dir / "K.txt")
            current_left = intrinsics["K1_L"]
            current_right = intrinsics["K1_R"]
            if K_left is None:
                K_left = current_left.copy()
                K_right = current_right.copy()
            elif not (
                np.allclose(K_left, current_left, rtol=0.0, atol=1e-9)
                and np.allclose(K_right, current_right, rtol=0.0, atol=1e-9)
            ):
                raise RuntimeError(
                    f"Endoscope1 intrinsics vary within session {session_id}: "
                    f"{sequence_dir}"
                )
            left = image_path_map(sequence_dir / "endoscope1" / "L")
            right = image_path_map(sequence_dir / "endoscope1" / "R")
            if not left or set(left) != set(right):
                raise RuntimeError(f"E1 stereo frame mismatch in {sequence_dir}")
            groups.append(
                [
                    CalibrationFrame(
                        split=split,
                        sequence_name=sequence_dir.name,
                        frame_id=frame_id,
                        left_path=left[frame_id],
                        right_path=right[frame_id],
                    )
                    for frame_id in sorted(left)
                ]
            )
    if not groups or K_left is None or K_right is None:
        raise RuntimeError(f"No frames found for session {session_id}")
    return groups, K_left, K_right


def _sample_across_groups(
    groups: list[list[CalibrationFrame]], num_samples: int
) -> list[CalibrationFrame]:
    total = sum(len(group) for group in groups)
    target = min(num_samples, total)
    if target <= 0:
        raise ValueError("num_samples must be positive")
    groups = [group for group in groups if group]
    allocations = [0] * len(groups)
    remaining = target
    if target >= len(groups):
        allocations = [1] * len(groups)
        remaining -= len(groups)
    while remaining > 0:
        candidates = [
            (len(group) / (allocations[index] + 1), index)
            for index, group in enumerate(groups)
            if allocations[index] < len(group)
        ]
        if not candidates:
            break
        _, index = max(candidates)
        allocations[index] += 1
        remaining -= 1
    sampled: list[CalibrationFrame] = []
    for group, count in zip(groups, allocations, strict=True):
        if count:
            indices = np.unique(np.linspace(0, len(group) - 1, count, dtype=int))
            sampled.extend(group[int(index)] for index in indices)
    return sampled


def _normalize_points(points_px: np.ndarray, K: np.ndarray) -> np.ndarray:
    return cv2.undistortPoints(
        np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2), K, None
    ).reshape(-1, 2)


def _essential_candidates(E: np.ndarray) -> list[np.ndarray]:
    E = np.asarray(E, dtype=np.float64)
    if E.shape == (3, 3):
        return [E]
    if E.ndim == 2 and E.shape[1] == 3 and E.shape[0] % 3 == 0:
        return [E[index:index + 3] for index in range(0, E.shape[0], 3)]
    raise RuntimeError(f"Unexpected essential-matrix shape {E.shape}")


def _estimate_essential(
    left_normalized: np.ndarray,
    right_normalized: np.ndarray,
    config: SessionCalibrationConfig,
) -> EssentialEstimate:
    E_all, mask = cv2.findEssentialMat(
        left_normalized,
        right_normalized,
        cameraMatrix=np.eye(3, dtype=np.float64),
        method=cv2.RANSAC,
        prob=config.essential_ransac_probability,
        threshold=config.essential_ransac_threshold_normalized,
    )
    if E_all is None or mask is None:
        raise RuntimeError("findEssentialMat failed")
    essential_mask = mask.reshape(-1).astype(bool)
    best: EssentialEstimate | None = None
    best_count = -1
    for E in _essential_candidates(E_all):
        recover_mask = essential_mask.astype(np.uint8).reshape(-1, 1)
        count, R, t, pose_mask = cv2.recoverPose(
            E,
            left_normalized,
            right_normalized,
            np.eye(3, dtype=np.float64),
            mask=recover_mask,
        )
        direction = np.asarray(t, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
        if int(count) > best_count and norm > 0.0:
            best = EssentialEstimate(
                E=np.asarray(E, dtype=np.float64),
                R=np.asarray(R, dtype=np.float64),
                t_unit=direction / norm,
                essential_mask=essential_mask,
                cheirality_mask=pose_mask.reshape(-1).astype(bool),
            )
            best_count = int(count)
    if best is None:
        raise RuntimeError("recoverPose failed for all essential candidates")
    return best


def _rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    relative = R_a @ R_b.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _direction_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    cosine = np.dot(a, b) / max(
        np.linalg.norm(a) * np.linalg.norm(b), np.finfo(np.float64).eps
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _estimate_pair(
    frame: CalibrationFrame,
    matcher: Method2FeatureMatcher,
    K_left: np.ndarray,
    K_right: np.ndarray,
    config: SessionCalibrationConfig,
) -> PairEstimate:
    left_features = matcher.extract(frame.left_path)
    right_features = matcher.extract(frame.right_path)
    matches = matcher.match_indices(left_features, right_features)
    left_keypoints = matcher.keypoints(left_features)
    right_keypoints = matcher.keypoints(right_features)
    left_px = left_keypoints[matches[:, 0]]
    right_px = right_keypoints[matches[:, 1]]
    estimate = PairEstimate(
        frame=frame,
        num_matches=int(matches.shape[0]),
        left_px=np.asarray(left_px, dtype=np.float64),
        right_px=np.asarray(right_px, dtype=np.float64),
    )
    if estimate.num_matches < config.min_matches:
        estimate.failure_reason = f"only {estimate.num_matches} matches"
        return estimate
    estimate.left_norm = _normalize_points(estimate.left_px, K_left)
    estimate.right_norm = _normalize_points(estimate.right_px, K_right)
    try:
        essential = _estimate_essential(
            estimate.left_norm, estimate.right_norm, config
        )
    except (cv2.error, RuntimeError) as error:
        estimate.failure_reason = str(error)
        return estimate
    estimate.E = essential.E
    estimate.R = essential.R
    estimate.t_unit = essential.t_unit
    estimate.essential_mask = essential.essential_mask
    estimate.cheirality_mask = essential.cheirality_mask
    estimate.num_essential_inliers = int(np.count_nonzero(essential.essential_mask))
    estimate.essential_inlier_ratio = (
        estimate.num_essential_inliers / estimate.num_matches
    )
    estimate.num_cheirality_inliers = int(np.count_nonzero(essential.cheirality_mask))
    estimate.cheirality_fraction = estimate.num_cheirality_inliers / max(
        estimate.num_essential_inliers, 1
    )
    failures: list[str] = []
    if estimate.num_essential_inliers < config.min_essential_inliers:
        failures.append("insufficient essential inliers")
    if estimate.essential_inlier_ratio < config.min_inlier_ratio:
        failures.append("low essential inlier ratio")
    if estimate.cheirality_fraction < config.min_cheirality_fraction:
        failures.append("low cheirality fraction")
    if failures:
        estimate.failure_reason = "; ".join(failures)
        return estimate
    estimate.success = True
    return estimate


def _select_stable(
    estimates: list[PairEstimate], config: SessionCalibrationConfig
) -> tuple[list[PairEstimate], PairEstimate]:
    valid = [estimate for estimate in estimates if estimate.success]
    if len(valid) < config.min_stable_pairs:
        raise RuntimeError(
            f"Only {len(valid)} individually valid stereo pairs; "
            f"need {config.min_stable_pairs}"
        )
    costs: list[float] = []
    for candidate in valid:
        assert candidate.R is not None and candidate.t_unit is not None
        distances = []
        for other in valid:
            assert other.R is not None and other.t_unit is not None
            distances.append(
                _rotation_angle_deg(candidate.R, other.R)
                / config.max_rotation_deviation_deg
                + _direction_angle_deg(candidate.t_unit, other.t_unit)
                / config.max_translation_deviation_deg
            )
        costs.append(float(np.median(distances)))
    medoid = valid[int(np.argmin(costs))]
    assert medoid.R is not None and medoid.t_unit is not None
    stable = [
        estimate
        for estimate in valid
        if estimate.R is not None
        and estimate.t_unit is not None
        and _rotation_angle_deg(estimate.R, medoid.R)
        <= config.max_rotation_deviation_deg
        and _direction_angle_deg(estimate.t_unit, medoid.t_unit)
        <= config.max_translation_deviation_deg
    ]
    if len(stable) < config.min_stable_pairs:
        raise RuntimeError(f"Only {len(stable)} geometrically stable stereo pairs")
    return stable, medoid


def _pool_stable(
    stable: list[PairEstimate], maximum_per_pair: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_px_all: list[np.ndarray] = []
    right_px_all: list[np.ndarray] = []
    left_norm_all: list[np.ndarray] = []
    right_norm_all: list[np.ndarray] = []
    for estimate in stable:
        assert estimate.cheirality_mask is not None
        assert estimate.left_px is not None and estimate.right_px is not None
        assert estimate.left_norm is not None and estimate.right_norm is not None
        indices = np.flatnonzero(estimate.cheirality_mask)
        if indices.size > maximum_per_pair:
            indices = indices[
                np.linspace(0, indices.size - 1, maximum_per_pair, dtype=int)
            ]
        left_px_all.append(estimate.left_px[indices])
        right_px_all.append(estimate.right_px[indices])
        left_norm_all.append(estimate.left_norm[indices])
        right_norm_all.append(estimate.right_norm[indices])
    return (
        np.concatenate(left_px_all),
        np.concatenate(right_px_all),
        np.concatenate(left_norm_all),
        np.concatenate(right_norm_all),
    )


def _sampson_errors(
    E: np.ndarray, left_normalized: np.ndarray, right_normalized: np.ndarray
) -> np.ndarray:
    ones = np.ones((left_normalized.shape[0], 1), dtype=np.float64)
    left_h = np.concatenate([left_normalized, ones], axis=1)
    right_h = np.concatenate([right_normalized, ones], axis=1)
    E_left = (E @ left_h.T).T
    Et_right = (E.T @ right_h.T).T
    numerator = np.abs(np.sum(right_h * E_left, axis=1))
    denominator = np.sqrt(
        E_left[:, 0] ** 2
        + E_left[:, 1] ** 2
        + Et_right[:, 0] ** 2
        + Et_right[:, 1] ** 2
    )
    return numerator / np.maximum(denominator, np.finfo(np.float64).eps)


def _triangulation_statistics(
    left_px: np.ndarray,
    right_px: np.ndarray,
    left_norm: np.ndarray,
    right_norm: np.ndarray,
    K_left: np.ndarray,
    K_right: np.ndarray,
    essential: EssentialEstimate,
) -> dict[str, float | int]:
    indices = np.flatnonzero(essential.essential_mask)
    left_n = left_norm[indices]
    right_n = right_norm[indices]
    P_left = np.hstack([np.eye(3), np.zeros((3, 1))])
    P_right = np.hstack([essential.R, essential.t_unit.reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(P_left, P_right, left_n.T, right_n.T)
    finite_w = np.abs(homogeneous[3]) > np.finfo(np.float64).eps
    points_left = np.full((len(indices), 3), np.nan, dtype=np.float64)
    points_left[finite_w] = (
        homogeneous[:3, finite_w] / homogeneous[3, finite_w]
    ).T
    points_right = (essential.R @ points_left.T).T + essential.t_unit
    valid = (
        finite_w
        & np.isfinite(points_left).all(axis=1)
        & np.isfinite(points_right).all(axis=1)
        & (points_left[:, 2] > 0.0)
        & (points_right[:, 2] > 0.0)
    )
    if not np.any(valid):
        raise RuntimeError("Final E1 stereo calibration triangulates no positive points")
    left_valid = points_left[valid]
    right_valid = points_right[valid]
    left_observed = left_px[indices][valid]
    right_observed = right_px[indices][valid]
    projected_left_h = (K_left @ left_valid.T).T
    projected_right_h = (K_right @ right_valid.T).T
    projected_left = projected_left_h[:, :2] / projected_left_h[:, 2:3]
    projected_right = projected_right_h[:, :2] / projected_right_h[:, 2:3]
    reprojection = 0.5 * (
        np.linalg.norm(projected_left - left_observed, axis=1)
        + np.linalg.norm(projected_right - right_observed, axis=1)
    )
    depths = left_valid[:, 2]
    p01 = float(np.percentile(depths, 1))
    p99 = float(np.percentile(depths, 99))
    suggested_min = max(0.01, 0.5 * p01)
    suggested_max = max(suggested_min + 1.0, 2.0 * p99)
    return {
        "num_triangulated": int(len(depths)),
        "cheirality_fraction": float(np.mean(valid)),
        "reprojection_error_median_px": float(np.median(reprojection)),
        "reprojection_error_p95_px": float(np.percentile(reprojection, 95)),
        "depth_median_baselines": float(np.median(depths)),
        "depth_mad_baselines": float(
            np.median(np.abs(depths - np.median(depths)))
        ),
        "depth_p01_baselines": p01,
        "depth_p99_baselines": p99,
        "suggested_min_depth": suggested_min,
        "suggested_max_depth": suggested_max,
    }


def _variation(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_median_deg": float(np.median(array)),
        f"{prefix}_p90_deg": float(np.percentile(array, 90)),
        f"{prefix}_max_deg": float(np.max(array)),
    }


def calibrate_e1_session(
    *,
    data_root: Path,
    session_id: str | int,
    output_dir: Path,
    matcher: Method2FeatureMatcher,
    config: SessionCalibrationConfig = SessionCalibrationConfig(),
    splits: tuple[str, ...] = ("train", "test"),
) -> tuple[Path, dict[str, Any]]:
    """Estimate one fixed `T_E1R_from_E1L` without reading pose.txt."""

    session = _normalize_session_id(session_id)
    groups, K_left, K_right = _discover_frames(data_root, session, splits)
    sampled = _sample_across_groups(groups, config.num_samples)
    total_frames = sum(len(group) for group in groups)
    estimates: list[PairEstimate] = []
    print(
        f"session_{session}: {total_frames} synchronized E1 stereo frames; "
        f"sampling {len(sampled)}",
        flush=True,
    )
    for index, frame in enumerate(sampled, start=1):
        estimate = _estimate_pair(frame, matcher, K_left, K_right, config)
        estimates.append(estimate)
        status = "valid" if estimate.success else f"rejected: {estimate.failure_reason}"
        print(
            f"[{index:02d}/{len(sampled):02d}] {frame.split}/{frame.sequence_name} "
            f"frame={frame.frame_id:06d} matches={estimate.num_matches} "
            f"E_inliers={estimate.num_essential_inliers} "
            f"ratio={estimate.essential_inlier_ratio:.3f} "
            f"cheirality={estimate.cheirality_fraction:.3f} {status}",
            flush=True,
        )

    stable, medoid = _select_stable(estimates, config)
    left_px, right_px, left_norm, right_norm = _pool_stable(
        stable, config.max_pooled_points_per_pair
    )
    final = _estimate_essential(left_norm, right_norm, config)
    sampson = _sampson_errors(final.E, left_norm, right_norm)
    inlier_sampson = sampson[final.essential_mask]
    focal_mean = float(
        np.mean([K_left[0, 0], K_left[1, 1], K_right[0, 0], K_right[1, 1]])
    )
    rotation_variation = [
        _rotation_angle_deg(estimate.R, final.R)
        for estimate in stable
        if estimate.R is not None
    ]
    translation_variation = [
        _direction_angle_deg(estimate.t_unit, final.t_unit)
        for estimate in stable
        if estimate.t_unit is not None
    ]
    individually_valid = [estimate for estimate in estimates if estimate.success]
    all_valid_rotation_variation = [
        _rotation_angle_deg(estimate.R, final.R)
        for estimate in individually_valid
        if estimate.R is not None
    ]
    all_valid_translation_variation = [
        _direction_angle_deg(estimate.t_unit, final.t_unit)
        for estimate in individually_valid
        if estimate.t_unit is not None
    ]
    rotation_determinant = float(np.linalg.det(final.R))
    rotation_orthogonality_error = float(
        np.linalg.norm(final.R.T @ final.R - np.eye(3), ord="fro")
    )
    translation_direction_norm = float(np.linalg.norm(final.t_unit))
    if (
        not np.isfinite(rotation_determinant)
        or abs(rotation_determinant - 1.0) > 1e-4
        or rotation_orthogonality_error > 1e-4
        or not np.isfinite(translation_direction_norm)
        or abs(translation_direction_norm - 1.0) > 1e-6
    ):
        raise RuntimeError(
            "Estimated E1 stereo calibration failed its SO(3)/unit-translation checks"
        )
    triangulation = _triangulation_statistics(
        left_px,
        right_px,
        left_norm,
        right_norm,
        K_left,
        K_right,
        final,
    )
    stable_fraction = len(stable) / max(len(individually_valid), 1)
    diagnostics: dict[str, Any] = {
        "session_id": session,
        "camera_pair": "endoscope1/L -> endoscope1/R",
        "num_sequences": len(groups),
        "num_frames": total_frames,
        "num_sampled": len(sampled),
        "num_individually_valid": len(individually_valid),
        "num_stable_pairs": len(stable),
        "stable_fraction_of_individually_valid": stable_fraction,
        "num_pooled_correspondences": int(left_norm.shape[0]),
        "num_pooled_essential_inliers": int(np.count_nonzero(final.essential_mask)),
        "pooled_essential_inlier_ratio": float(np.mean(final.essential_mask)),
        "pooled_inlier_epipolar_sampson_median_normalized": float(
            np.median(inlier_sampson)
        ),
        "pooled_inlier_epipolar_sampson_p95_normalized": float(
            np.percentile(inlier_sampson, 95)
        ),
        "pooled_inlier_epipolar_sampson_median_approx_px": float(
            np.median(inlier_sampson) * focal_mean
        ),
        "pooled_inlier_epipolar_sampson_p95_approx_px": float(
            np.percentile(inlier_sampson, 95) * focal_mean
        ),
        **_variation(rotation_variation, "rotation_variation"),
        **_variation(translation_variation, "translation_direction_variation"),
        **_variation(all_valid_rotation_variation, "all_valid_rotation_variation"),
        **_variation(
            all_valid_translation_variation,
            "all_valid_translation_direction_variation",
        ),
        "rotation_determinant": rotation_determinant,
        "rotation_orthogonality_error": rotation_orthogonality_error,
        "translation_direction_norm": translation_direction_norm,
        **triangulation,
        "medoid": {
            "split": medoid.frame.split,
            "sequence": medoid.frame.sequence_name,
            "frame_id": medoid.frame.frame_id,
        },
        "config": asdict(config),
        "sampled_pairs": [
            {
                "split": estimate.frame.split,
                "sequence": estimate.frame.sequence_name,
                "frame_id": estimate.frame.frame_id,
                "num_matches": estimate.num_matches,
                "num_essential_inliers": estimate.num_essential_inliers,
                "essential_inlier_ratio": estimate.essential_inlier_ratio,
                "cheirality_fraction": estimate.cheirality_fraction,
                "individually_valid": estimate.success,
                "failure_reason": estimate.failure_reason,
                "rotation_deviation_from_final_deg": (
                    _rotation_angle_deg(estimate.R, final.R)
                    if estimate.R is not None
                    else None
                ),
                "translation_deviation_from_final_deg": (
                    _direction_angle_deg(estimate.t_unit, final.t_unit)
                    if estimate.t_unit is not None
                    else None
                ),
            }
            for estimate in estimates
        ],
    }
    validity_failures: list[str] = []
    if stable_fraction < config.min_stable_fraction:
        validity_failures.append("stable fraction below configured minimum")
    if diagnostics["pooled_essential_inlier_ratio"] < config.min_inlier_ratio:
        validity_failures.append("pooled essential inlier ratio below minimum")
    if diagnostics["rotation_variation_p90_deg"] > config.max_rotation_deviation_deg:
        validity_failures.append("rotation variation p90 above maximum")
    if (
        diagnostics["translation_direction_variation_p90_deg"]
        > config.max_translation_deviation_deg
    ):
        validity_failures.append("translation-direction variation p90 above maximum")
    if diagnostics["cheirality_fraction"] < config.min_cheirality_fraction:
        validity_failures.append("pooled cheirality below minimum")
    if (
        diagnostics["pooled_inlier_epipolar_sampson_p95_approx_px"]
        > config.max_epipolar_p95_px
    ):
        validity_failures.append("pooled epipolar p95 above maximum")
    if diagnostics["reprojection_error_median_px"] > config.max_reprojection_median_px:
        validity_failures.append("median stereo reprojection error above maximum")
    diagnostics["fixed_transform_appears_valid"] = not validity_failures
    diagnostics["validity_failures"] = validity_failures

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = final.R
    transform[:3, 3] = final.t_unit
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"session_{session}.npz"
    diagnostics_path = output_dir / f"session_{session}_diagnostics.json"
    np.savez_compressed(
        npz_path,
        session_id=np.array(session),
        K_left=K_left,
        K_right=K_right,
        E=final.E,
        R_LR=final.R,
        t_LR_unit=final.t_unit,
        T_R_from_L=transform,
        suggested_min_depth=np.array(diagnostics["suggested_min_depth"]),
        suggested_max_depth=np.array(diagnostics["suggested_max_depth"]),
        diagnostic_statistics_json=np.array(json.dumps(diagnostics, sort_keys=True)),
    )
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(f"Saved E1 stereo calibration: {npz_path}", flush=True)
    print(f"Saved calibration diagnostics: {diagnostics_path}", flush=True)
    return npz_path, diagnostics
