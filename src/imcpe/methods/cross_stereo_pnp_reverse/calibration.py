"""Inference-only, session-level Endoscope-2 stereo self-calibration."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from imcpe.methods.cross_stereo_pnp.data import image_path_map, read_intrinsics_only
from imcpe.methods.cross_stereo_pnp.session_calibration import (
    CalibrationFrame,
    _direction_angle_deg,
    _estimate_essential,
    _estimate_pair,
    _pool_stable,
    _rotation_angle_deg,
    _sample_across_groups,
    _sampson_errors,
    _select_stable,
    _triangulation_statistics,
    _variation,
)

from .config import ReverseCalibrationConfig
from .matching import Method2BFeatureMatcher


def _normalise_session_id(value: str | int) -> str:
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
            parts = sequence_dir.name.split("_")
            if sequence_dir.is_dir() and len(parts) >= 2 and parts[1].isdigit():
                identifiers.add(parts[1].zfill(3))
    return sorted(identifiers)


def _discover_e2_frames(
    data_root: Path,
    session_id: str,
    splits: tuple[str, ...],
) -> tuple[list[list[CalibrationFrame]], np.ndarray, np.ndarray]:
    """Read only K.txt and E2 images; never instantiate the GT-aware loader."""

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
            current_left = intrinsics["K2_L"]
            current_right = intrinsics["K2_R"]
            if K_left is None:
                K_left = current_left.copy()
                K_right = current_right.copy()
            elif not (
                np.allclose(K_left, current_left, rtol=0.0, atol=1e-9)
                and np.allclose(K_right, current_right, rtol=0.0, atol=1e-9)
            ):
                raise RuntimeError(
                    f"Endoscope2 intrinsics vary within session {session_id}: "
                    f"{sequence_dir}"
                )
            left = image_path_map(sequence_dir / "endoscope2" / "L")
            right = image_path_map(sequence_dir / "endoscope2" / "R")
            if not left or set(left) != set(right):
                raise RuntimeError(f"E2 stereo frame mismatch in {sequence_dir}")
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
        raise RuntimeError(f"No E2 stereo frames found for session {session_id}")
    return groups, K_left, K_right


def calibrate_e2_session(
    *,
    data_root: Path,
    session_id: str | int,
    output_dir: Path,
    matcher: Method2BFeatureMatcher,
    config: ReverseCalibrationConfig = ReverseCalibrationConfig(),
    splits: tuple[str, ...] = ("train", "test"),
) -> tuple[Path, dict[str, Any]]:
    """Estimate one fixed scale-free `T_E2R_from_E2L` without reading GT."""

    session = _normalise_session_id(session_id)
    groups, K_left, K_right = _discover_e2_frames(data_root, session, splits)
    sampled = _sample_across_groups(groups, config.num_samples)
    total_frames = sum(len(group) for group in groups)
    estimates = []
    print(
        f"session_{session}: {total_frames} synchronized E2 stereo frames; "
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
    individually_valid = [estimate for estimate in estimates if estimate.success]
    stable_rotation = [
        _rotation_angle_deg(estimate.R, final.R)
        for estimate in stable
        if estimate.R is not None
    ]
    stable_translation = [
        _direction_angle_deg(estimate.t_unit, final.t_unit)
        for estimate in stable
        if estimate.t_unit is not None
    ]
    all_rotation = [
        _rotation_angle_deg(estimate.R, final.R)
        for estimate in individually_valid
        if estimate.R is not None
    ]
    all_translation = [
        _direction_angle_deg(estimate.t_unit, final.t_unit)
        for estimate in individually_valid
        if estimate.t_unit is not None
    ]
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
    rotation_determinant = float(np.linalg.det(final.R))
    rotation_orthogonality_error = float(
        np.linalg.norm(final.R.T @ final.R - np.eye(3), ord="fro")
    )
    translation_norm = float(np.linalg.norm(final.t_unit))
    diagnostics: dict[str, Any] = {
        "session_id": session,
        "camera_pair": "endoscope2/L -> endoscope2/R",
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
        **_variation(stable_rotation, "rotation_variation"),
        **_variation(stable_translation, "translation_direction_variation"),
        **_variation(all_rotation, "all_valid_rotation_variation"),
        **_variation(
            all_translation, "all_valid_translation_direction_variation"
        ),
        "rotation_p50_deg": float(np.median(all_rotation)),
        "rotation_p90_deg": float(np.percentile(all_rotation, 90)),
        "translation_direction_p50_deg": float(np.median(all_translation)),
        "translation_direction_p90_deg": float(np.percentile(all_translation, 90)),
        "median_epipolar_error_px": float(np.median(inlier_sampson) * focal_mean),
        "rotation_determinant": rotation_determinant,
        "rotation_orthogonality_error": rotation_orthogonality_error,
        "translation_direction_norm": translation_norm,
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
    if (
        abs(rotation_determinant - 1.0) > 1e-4
        or rotation_orthogonality_error > 1e-4
        or abs(translation_norm - 1.0) > 1e-6
    ):
        validity_failures.append("SO(3) or unit-translation check failed")
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
    print(f"Saved E2 stereo calibration: {npz_path}", flush=True)
    print(f"Saved calibration diagnostics: {diagnostics_path}", flush=True)
    if not diagnostics["fixed_transform_appears_valid"]:
        print(
            "WARNING: E2 stereo calibration is invalid: "
            + "; ".join(validity_failures),
            flush=True,
        )
    return npz_path, diagnostics
