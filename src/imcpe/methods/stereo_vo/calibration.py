from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from imcpe.data_io import load_sequence
from imcpe.matcher import ALikeLightGlueMatcher


@dataclass(frozen=True)
class CalibrationConfig:
    """Thresholds for session-level, target-free stereo calibration."""

    num_samples: int = 30
    essential_ransac_threshold: float = 1e-3
    essential_ransac_probability: float = 0.999
    min_matches: int = 50
    min_essential_inliers: int = 30
    min_inlier_ratio: float = 0.25
    min_cheirality_fraction: float = 0.50
    max_rotation_deviation_deg: float = 5.0
    max_translation_deviation_deg: float = 20.0
    min_stable_pairs: int = 3
    min_stable_fraction: float = 0.50
    max_pooled_points_per_pair: int = 400
    max_epipolar_p95_px: float = 2.0
    max_reprojection_median_px: float = 2.0


@dataclass(frozen=True)
class StereoFrame:
    split: str
    sequence_name: str
    frame_id: int
    left_path: Path
    right_path: Path


@dataclass
class PairEstimate:
    frame: StereoFrame
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
    points_left_px: np.ndarray | None = None
    points_right_px: np.ndarray | None = None
    points_left_norm: np.ndarray | None = None
    points_right_norm: np.ndarray | None = None
    cheirality_mask: np.ndarray | None = None


@dataclass(frozen=True)
class EssentialEstimate:
    E: np.ndarray
    R: np.ndarray
    t_unit: np.ndarray
    essential_mask: np.ndarray
    cheirality_mask: np.ndarray


def _normalise_session_id(session_id: str | int) -> str:
    text = str(session_id)
    if text.startswith("session_"):
        text = text[len("session_"):]
    if not text.isdigit():
        raise ValueError(f"Expected a numeric session ID, got {session_id!r}")
    return text.zfill(3)


def _normalise_points(points_px: np.ndarray, K: np.ndarray) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
    # The dataset supplies no distortion coefficients. Passing None makes this
    # purely an intrinsic normalisation; it does not correct lens distortion.
    return cv2.undistortPoints(points, K, None).reshape(-1, 2)


def _essential_candidates(E: np.ndarray) -> list[np.ndarray]:
    E = np.asarray(E, dtype=np.float64)
    if E.shape == (3, 3):
        return [E]
    if E.ndim == 2 and E.shape[1] == 3 and E.shape[0] % 3 == 0:
        return [E[i:i + 3] for i in range(0, E.shape[0], 3)]
    raise RuntimeError(f"Unexpected essential-matrix shape: {E.shape}")


def _estimate_essential(
    points_left_norm: np.ndarray,
    points_right_norm: np.ndarray,
    config: CalibrationConfig,
) -> EssentialEstimate:
    E_all, mask = cv2.findEssentialMat(
        points_left_norm,
        points_right_norm,
        cameraMatrix=np.eye(3, dtype=np.float64),
        method=cv2.RANSAC,
        prob=config.essential_ransac_probability,
        threshold=config.essential_ransac_threshold,
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
            points_left_norm,
            points_right_norm,
            np.eye(3, dtype=np.float64),
            mask=recover_mask,
        )
        pose_mask_bool = pose_mask.reshape(-1).astype(bool)
        if int(count) > best_count:
            t_unit = np.asarray(t, dtype=np.float64).reshape(3)
            norm = float(np.linalg.norm(t_unit))
            if norm == 0.0:
                continue
            best = EssentialEstimate(
                E=np.asarray(E, dtype=np.float64),
                R=np.asarray(R, dtype=np.float64),
                t_unit=t_unit / norm,
                essential_mask=essential_mask,
                cheirality_mask=pose_mask_bool,
            )
            best_count = int(count)
    if best is None:
        raise RuntimeError("recoverPose failed for every essential-matrix candidate")
    return best


def _rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    relative = R_a @ R_b.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _direction_angle_deg(t_a: np.ndarray, t_b: np.ndarray) -> float:
    a = np.asarray(t_a, dtype=np.float64).reshape(3)
    b = np.asarray(t_b, dtype=np.float64).reshape(3)
    cosine = float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _discover_session_frames(
    data_root: Path,
    session_id: str,
    splits: tuple[str, ...],
) -> tuple[list[list[StereoFrame]], np.ndarray, np.ndarray]:
    prefix = f"session_{session_id}_"
    groups: list[list[StereoFrame]] = []
    K_left: np.ndarray | None = None
    K_right: np.ndarray | None = None

    for split in splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            continue
        for sequence_dir in sorted(p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)):
            sequence = load_sequence(sequence_dir)
            if K_left is None:
                K_left = sequence.k2_l.copy()
                K_right = sequence.k2_r.copy()
            elif not (
                np.allclose(K_left, sequence.k2_l, rtol=0.0, atol=1e-9)
                and np.allclose(K_right, sequence.k2_r, rtol=0.0, atol=1e-9)
            ):
                raise RuntimeError(
                    f"Endoscope2 intrinsics differ within session {session_id}: {sequence_dir}"
                )

            groups.append([
                StereoFrame(
                    split=split,
                    sequence_name=sequence.sequence_name,
                    frame_id=frame_id,
                    left_path=left_path,
                    right_path=right_path,
                )
                for frame_id, left_path, right_path in zip(
                    sequence.frame_ids,
                    sequence.e2_l_images,
                    sequence.e2_r_images,
                    strict=True,
                )
            ])

    if not groups or K_left is None or K_right is None:
        raise RuntimeError(f"No sequences found for session {session_id} under {data_root}")
    return groups, K_left, K_right


def _sample_across_sequences(groups: list[list[StereoFrame]], num_samples: int) -> list[StereoFrame]:
    total = sum(len(group) for group in groups)
    target = min(num_samples, total)
    if target <= 0:
        raise ValueError("num_samples must be positive")

    nonempty = [group for group in groups if group]
    allocations = [0] * len(nonempty)
    if target >= len(nonempty):
        allocations = [1] * len(nonempty)
        remaining = target - len(nonempty)
    else:
        remaining = target

    while remaining > 0:
        candidates = [
            (len(group) / (allocations[i] + 1), i)
            for i, group in enumerate(nonempty)
            if allocations[i] < len(group)
        ]
        if not candidates:
            break
        _, selected = max(candidates)
        allocations[selected] += 1
        remaining -= 1

    sampled: list[StereoFrame] = []
    for group, count in zip(nonempty, allocations, strict=True):
        if count == 0:
            continue
        indices = np.linspace(0, len(group) - 1, num=count, dtype=int)
        sampled.extend(group[int(index)] for index in np.unique(indices))
    return sampled


def _estimate_pair(
    frame: StereoFrame,
    matcher: ALikeLightGlueMatcher,
    K_left: np.ndarray,
    K_right: np.ndarray,
    config: CalibrationConfig,
) -> PairEstimate:
    points_left_px, points_right_px = matcher.match(frame.left_path, frame.right_path)
    estimate = PairEstimate(frame=frame, num_matches=int(points_left_px.shape[0]))
    estimate.points_left_px = np.asarray(points_left_px, dtype=np.float64)
    estimate.points_right_px = np.asarray(points_right_px, dtype=np.float64)

    if estimate.num_matches < config.min_matches:
        estimate.failure_reason = f"only {estimate.num_matches} matches"
        return estimate

    estimate.points_left_norm = _normalise_points(estimate.points_left_px, K_left)
    estimate.points_right_norm = _normalise_points(estimate.points_right_px, K_right)
    try:
        essential = _estimate_essential(
            estimate.points_left_norm,
            estimate.points_right_norm,
            config,
        )
    except (cv2.error, RuntimeError) as error:
        estimate.failure_reason = str(error)
        return estimate

    estimate.E = essential.E
    estimate.R = essential.R
    estimate.t_unit = essential.t_unit
    estimate.cheirality_mask = essential.cheirality_mask
    estimate.num_essential_inliers = int(np.count_nonzero(essential.essential_mask))
    estimate.essential_inlier_ratio = estimate.num_essential_inliers / estimate.num_matches
    estimate.num_cheirality_inliers = int(np.count_nonzero(essential.cheirality_mask))
    estimate.cheirality_fraction = (
        estimate.num_cheirality_inliers / estimate.num_essential_inliers
        if estimate.num_essential_inliers else 0.0
    )

    failures: list[str] = []
    if estimate.num_essential_inliers < config.min_essential_inliers:
        failures.append(f"only {estimate.num_essential_inliers} essential inliers")
    if estimate.essential_inlier_ratio < config.min_inlier_ratio:
        failures.append(f"inlier ratio {estimate.essential_inlier_ratio:.3f}")
    if estimate.cheirality_fraction < config.min_cheirality_fraction:
        failures.append(f"cheirality fraction {estimate.cheirality_fraction:.3f}")
    if failures:
        estimate.failure_reason = "; ".join(failures)
        return estimate

    estimate.success = True
    return estimate


def _select_stable_pairs(
    estimates: list[PairEstimate],
    config: CalibrationConfig,
) -> tuple[list[PairEstimate], PairEstimate]:
    candidates = [estimate for estimate in estimates if estimate.success]
    if len(candidates) < config.min_stable_pairs:
        raise RuntimeError(
            f"Only {len(candidates)} individually valid pairs; need {config.min_stable_pairs}"
        )

    costs: list[float] = []
    for candidate in candidates:
        assert candidate.R is not None and candidate.t_unit is not None
        distances = []
        for other in candidates:
            assert other.R is not None and other.t_unit is not None
            rotation = _rotation_angle_deg(candidate.R, other.R)
            translation = _direction_angle_deg(candidate.t_unit, other.t_unit)
            distances.append(
                rotation / config.max_rotation_deviation_deg
                + translation / config.max_translation_deviation_deg
            )
        costs.append(float(np.median(distances)))

    medoid = candidates[int(np.argmin(costs))]
    assert medoid.R is not None and medoid.t_unit is not None
    stable = [
        candidate
        for candidate in candidates
        if candidate.R is not None
        and candidate.t_unit is not None
        and _rotation_angle_deg(candidate.R, medoid.R) <= config.max_rotation_deviation_deg
        and _direction_angle_deg(candidate.t_unit, medoid.t_unit)
        <= config.max_translation_deviation_deg
    ]
    if len(stable) < config.min_stable_pairs:
        raise RuntimeError(
            f"Only {len(stable)} geometrically stable pairs; need {config.min_stable_pairs}"
        )
    return stable, medoid


def _pool_stable_correspondences(
    stable: list[PairEstimate],
    max_points_per_pair: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_px: list[np.ndarray] = []
    right_px: list[np.ndarray] = []
    left_norm: list[np.ndarray] = []
    right_norm: list[np.ndarray] = []

    for estimate in stable:
        assert estimate.cheirality_mask is not None
        assert estimate.points_left_px is not None and estimate.points_right_px is not None
        assert estimate.points_left_norm is not None and estimate.points_right_norm is not None
        indices = np.flatnonzero(estimate.cheirality_mask)
        if indices.size > max_points_per_pair:
            keep = np.linspace(0, indices.size - 1, max_points_per_pair, dtype=int)
            indices = indices[keep]
        left_px.append(estimate.points_left_px[indices])
        right_px.append(estimate.points_right_px[indices])
        left_norm.append(estimate.points_left_norm[indices])
        right_norm.append(estimate.points_right_norm[indices])

    return (
        np.concatenate(left_px),
        np.concatenate(right_px),
        np.concatenate(left_norm),
        np.concatenate(right_norm),
    )


def _sampson_errors(
    E: np.ndarray,
    points_left_norm: np.ndarray,
    points_right_norm: np.ndarray,
) -> np.ndarray:
    ones = np.ones((points_left_norm.shape[0], 1), dtype=np.float64)
    left_h = np.concatenate([points_left_norm, ones], axis=1)
    right_h = np.concatenate([points_right_norm, ones], axis=1)
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


def _triangulation_diagnostics(
    points_left_px: np.ndarray,
    points_right_px: np.ndarray,
    points_left_norm: np.ndarray,
    points_right_norm: np.ndarray,
    K_left: np.ndarray,
    K_right: np.ndarray,
    essential: EssentialEstimate,
) -> dict[str, float | int]:
    essential_indices = np.flatnonzero(essential.essential_mask)
    left_n = points_left_norm[essential_indices]
    right_n = points_right_norm[essential_indices]
    left_px = points_left_px[essential_indices]
    right_px = points_right_px[essential_indices]

    P_left = np.hstack([np.eye(3), np.zeros((3, 1))])
    P_right = np.hstack([essential.R, essential.t_unit.reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(P_left, P_right, left_n.T, right_n.T)
    finite_w = np.abs(homogeneous[3]) > np.finfo(np.float64).eps
    points_left_3d = np.full((homogeneous.shape[1], 3), np.nan, dtype=np.float64)
    points_left_3d[finite_w] = (homogeneous[:3, finite_w] / homogeneous[3, finite_w]).T
    points_right_3d = (essential.R @ points_left_3d.T).T + essential.t_unit
    positive = (
        finite_w
        & np.isfinite(points_left_3d).all(axis=1)
        & np.isfinite(points_right_3d).all(axis=1)
        & (points_left_3d[:, 2] > 0.0)
        & (points_right_3d[:, 2] > 0.0)
    )

    valid_left = points_left_3d[positive]
    valid_right = points_right_3d[positive]
    observed_left = left_px[positive]
    observed_right = right_px[positive]
    if valid_left.shape[0] == 0:
        return {
            "num_triangulated": 0,
            "cheirality_fraction": 0.0,
            "reprojection_error_median_px": math.nan,
            "reprojection_error_p95_px": math.nan,
        }

    projected_left_h = (K_left @ valid_left.T).T
    projected_right_h = (K_right @ valid_right.T).T
    projected_left = projected_left_h[:, :2] / projected_left_h[:, 2:3]
    projected_right = projected_right_h[:, :2] / projected_right_h[:, 2:3]
    left_error = np.linalg.norm(projected_left - observed_left, axis=1)
    right_error = np.linalg.norm(projected_right - observed_right, axis=1)
    reprojection_error = 0.5 * (left_error + right_error)
    return {
        "num_triangulated": int(valid_left.shape[0]),
        "cheirality_fraction": float(valid_left.shape[0] / max(len(essential_indices), 1)),
        "reprojection_error_median_px": float(np.median(reprojection_error)),
        "reprojection_error_p95_px": float(np.percentile(reprojection_error, 95)),
    }


def _variation_statistics(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_median_deg": float(np.median(array)),
        f"{prefix}_p90_deg": float(np.percentile(array, 90)),
        f"{prefix}_max_deg": float(np.max(array)),
    }


def _pair_record(
    estimate: PairEstimate,
    stable_ids: set[int],
    medoid: PairEstimate,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "split": estimate.frame.split,
        "sequence": estimate.frame.sequence_name,
        "frame_id": estimate.frame.frame_id,
        "num_matches": estimate.num_matches,
        "num_essential_inliers": estimate.num_essential_inliers,
        "essential_inlier_ratio": estimate.essential_inlier_ratio,
        "num_cheirality_inliers": estimate.num_cheirality_inliers,
        "cheirality_fraction": estimate.cheirality_fraction,
        "individually_valid": estimate.success,
        "selected_stable": id(estimate) in stable_ids,
        "failure_reason": estimate.failure_reason,
    }
    if estimate.R is not None and estimate.t_unit is not None:
        assert medoid.R is not None and medoid.t_unit is not None
        record["rotation_deviation_from_medoid_deg"] = _rotation_angle_deg(
            estimate.R, medoid.R
        )
        record["translation_deviation_from_medoid_deg"] = _direction_angle_deg(
            estimate.t_unit, medoid.t_unit
        )
    else:
        record["rotation_deviation_from_medoid_deg"] = None
        record["translation_deviation_from_medoid_deg"] = None
    return record


def calibrate_session(
    data_root: Path,
    session_id: str | int,
    output_dir: Path,
    matcher: ALikeLightGlueMatcher,
    config: CalibrationConfig = CalibrationConfig(),
    splits: tuple[str, ...] = ("train", "test"),
) -> tuple[Path, dict[str, Any]]:
    """Estimate one scale-free Endoscope2 transform for a complete session.

    The returned transform obeys ``X_R = R_LR @ X_L + t_LR_unit``. Its
    translation norm is fixed to one because the dataset does not supply a
    physical stereo baseline. This routine never writes into ``data_root``.
    """

    session = _normalise_session_id(session_id)
    groups, K_left, K_right = _discover_session_frames(data_root, session, splits)
    sampled = _sample_across_sequences(groups, config.num_samples)
    total_frames = sum(len(group) for group in groups)

    estimates: list[PairEstimate] = []
    print(
        f"session_{session}: {total_frames} synchronized frames across "
        f"{len(groups)} sequences; sampling {len(sampled)}",
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

    stable, medoid = _select_stable_pairs(estimates, config)
    pooled_left_px, pooled_right_px, pooled_left_norm, pooled_right_norm = (
        _pool_stable_correspondences(stable, config.max_pooled_points_per_pair)
    )
    final = _estimate_essential(pooled_left_norm, pooled_right_norm, config)

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
    assert medoid.R is not None and medoid.t_unit is not None
    all_rotation_variation = [
        _rotation_angle_deg(estimate.R, medoid.R)
        for estimate in individually_valid
        if estimate.R is not None
    ]
    all_translation_variation = [
        _direction_angle_deg(estimate.t_unit, medoid.t_unit)
        for estimate in individually_valid
        if estimate.t_unit is not None
    ]
    sampson = _sampson_errors(final.E, pooled_left_norm, pooled_right_norm)
    sampson_inliers = sampson[final.essential_mask]
    mean_focal_px = float(
        np.mean([K_left[0, 0], K_left[1, 1], K_right[0, 0], K_right[1, 1]])
    )
    triangulation = _triangulation_diagnostics(
        pooled_left_px,
        pooled_right_px,
        pooled_left_norm,
        pooled_right_norm,
        K_left,
        K_right,
        final,
    )

    diagnostics: dict[str, Any] = {
        "session_id": session,
        "num_sequences": len(groups),
        "num_frames": total_frames,
        "num_sampled": len(sampled),
        "num_individually_valid": sum(estimate.success for estimate in estimates),
        "num_stable_pairs": len(stable),
        "num_pooled_correspondences": int(pooled_left_norm.shape[0]),
        "num_pooled_essential_inliers": int(np.count_nonzero(final.essential_mask)),
        "pooled_essential_inlier_ratio": float(np.mean(final.essential_mask)),
        "pooled_epipolar_sampson_median_normalized": float(np.median(sampson)),
        "pooled_epipolar_sampson_p95_normalized": float(np.percentile(sampson, 95)),
        "pooled_inlier_epipolar_sampson_median_normalized": float(np.median(sampson_inliers)),
        "pooled_inlier_epipolar_sampson_p95_normalized": float(np.percentile(sampson_inliers, 95)),
        "pooled_inlier_epipolar_sampson_median_approx_px": float(
            np.median(sampson_inliers) * mean_focal_px
        ),
        "pooled_inlier_epipolar_sampson_p95_approx_px": float(
            np.percentile(sampson_inliers, 95) * mean_focal_px
        ),
        **_variation_statistics(all_rotation_variation, "all_valid_rotation_variation"),
        **_variation_statistics(
            all_translation_variation, "all_valid_translation_direction_variation"
        ),
        **_variation_statistics(rotation_variation, "rotation_variation"),
        **_variation_statistics(translation_variation, "translation_direction_variation"),
        **triangulation,
        "medoid": {
            "split": medoid.frame.split,
            "sequence": medoid.frame.sequence_name,
            "frame_id": medoid.frame.frame_id,
        },
        "config": asdict(config),
    }
    validity_failures: list[str] = []
    stable_fraction = len(stable) / max(len(individually_valid), 1)
    diagnostics["stable_fraction_of_individually_valid"] = stable_fraction
    if stable_fraction < config.min_stable_fraction:
        validity_failures.append(
            f"stable fraction {stable_fraction:.3f} < {config.min_stable_fraction:.3f}"
        )
    if diagnostics["pooled_essential_inlier_ratio"] < config.min_inlier_ratio:
        validity_failures.append(
            "pooled essential inlier ratio below configured minimum"
        )
    if diagnostics["rotation_variation_p90_deg"] > config.max_rotation_deviation_deg:
        validity_failures.append("stable-pair rotation p90 exceeds configured maximum")
    if (
        diagnostics["translation_direction_variation_p90_deg"]
        > config.max_translation_deviation_deg
    ):
        validity_failures.append(
            "stable-pair translation-direction p90 exceeds configured maximum"
        )
    if diagnostics["cheirality_fraction"] < config.min_cheirality_fraction:
        validity_failures.append("pooled cheirality fraction below configured minimum")
    if (
        diagnostics["pooled_inlier_epipolar_sampson_p95_approx_px"]
        > config.max_epipolar_p95_px
    ):
        validity_failures.append("pooled inlier epipolar p95 exceeds configured maximum")
    if diagnostics["reprojection_error_median_px"] > config.max_reprojection_median_px:
        validity_failures.append("median stereo reprojection error exceeds configured maximum")
    diagnostics["fixed_transform_appears_valid"] = not validity_failures
    diagnostics["validity_failures"] = validity_failures
    stable_ids = {id(estimate) for estimate in stable}
    diagnostics["sampled_pairs"] = [
        _pair_record(estimate, stable_ids, medoid) for estimate in estimates
    ]

    T_right_from_left = np.eye(4, dtype=np.float64)
    T_right_from_left[:3, :3] = final.R
    T_right_from_left[:3, 3] = final.t_unit

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"session_{session}.npz"
    json_path = output_dir / f"session_{session}_diagnostics.json"
    np.savez_compressed(
        npz_path,
        K_left=K_left,
        K_right=K_right,
        E=final.E,
        R_LR=final.R,
        t_LR_unit=final.t_unit,
        T_R_from_L=T_right_from_left,
        session_id=np.array(session),
        diagnostic_statistics_json=np.array(json.dumps(diagnostics, sort_keys=True)),
    )
    json_path.write_text(json.dumps(diagnostics, indent=2) + "\n")

    print(f"Saved calibration: {npz_path}", flush=True)
    print(f"Saved diagnostics: {json_path}", flush=True)
    return npz_path, diagnostics
