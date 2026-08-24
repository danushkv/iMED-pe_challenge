from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class AnchorConfidence:
    weight_a: np.ndarray
    weight_b: np.ndarray
    valid_a: np.ndarray
    valid_b: np.ndarray
    position_disagreement: np.ndarray
    position_disagreement_normalized: np.ndarray
    rotation_disagreement_deg: np.ndarray
    robust_z: np.ndarray
    trajectory_step_scale: float
    a_inlier_ratio: np.ndarray
    b_inlier_ratio: np.ndarray
    a_reprojection_px: np.ndarray
    b_reprojection_px: np.ndarray


def _frame_quantity(
    frame_ids: np.ndarray,
    frames: dict[int, dict[str, Any]],
    key: str,
    default: float,
) -> np.ndarray:
    return np.asarray(
        [float(frames.get(int(frame_id), {}).get(key, default)) for frame_id in frame_ids],
        dtype=np.float64,
    )


def _valid_flags(
    frame_ids: np.ndarray, frames: dict[int, dict[str, Any]]
) -> np.ndarray:
    flags = np.asarray(
        [bool(frames.get(int(frame_id), {}).get("pnp_success", frame_id == frame_ids[0])) for frame_id in frame_ids]
    )
    flags[0] = True
    return flags


def _quality_v2(
    valid: np.ndarray,
    ratios: np.ndarray,
    reprojection: np.ndarray,
    inliers: np.ndarray,
    shared: np.ndarray,
    calibration_valid: bool,
) -> np.ndarray:
    mask = valid & np.isfinite(ratios) & np.isfinite(reprojection)
    if not np.any(mask):
        return valid.astype(np.float64)
    median_ratio = max(float(np.median(ratios[mask])), 1e-6)
    median_reprojection = max(float(np.median(reprojection[mask])), 1e-6)
    median_inliers = max(float(np.median(inliers[mask])), 1.0)
    median_shared = max(float(np.median(shared[mask])), 1.0)
    components = np.stack(
        [
            np.clip(ratios / median_ratio, 0.25, 2.0),
            np.clip(median_reprojection / np.maximum(reprojection, 1e-6), 0.25, 2.0),
            np.sqrt(np.clip(inliers / median_inliers, 0.25, 4.0)),
            np.sqrt(np.clip(shared / median_shared, 0.25, 4.0)),
        ],
        axis=1,
    )
    quality = np.exp(np.mean(np.log(components), axis=1))
    quality *= 1.0 if calibration_valid else 0.5
    quality[~valid] = 0.0
    return np.clip(quality, 0.0, 2.0)


def build_anchor_confidence(
    *,
    frame_ids: np.ndarray,
    centers_a_aligned: np.ndarray,
    centers_b: np.ndarray,
    rotations_a: np.ndarray,
    rotations_b: np.ndarray,
    frames_a: dict[int, dict[str, Any]],
    frames_b: dict[int, dict[str, Any]],
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
    version: str,
) -> AnchorConfidence:
    displacement = np.linalg.norm(np.diff(centers_b, axis=0), axis=1)
    nonzero = displacement[displacement > 1e-9]
    step_scale = float(np.median(nonzero)) if nonzero.size else 1.0
    position = np.linalg.norm(centers_a_aligned - centers_b, axis=1)
    position_normalized = position / max(step_scale, 1e-9)
    rotation = np.asarray(
        [
            np.degrees(Rotation.from_matrix(Ra.T @ Rb).magnitude())
            for Ra, Rb in zip(rotations_a, rotations_b)
        ]
    )
    median = float(np.median(position))
    mad = float(np.median(np.abs(position - median)))
    sigma = max(1.4826 * mad, 1e-9)
    robust_z = np.maximum((position - median) / sigma, 0.0)
    agreement_weight = 1.0 / (1.0 + robust_z**2)
    valid_a = _valid_flags(frame_ids, frames_a)
    valid_b = _valid_flags(frame_ids, frames_b)
    a_ratio = _frame_quantity(frame_ids, frames_a, "pnp_inlier_ratio", 1.0)
    b_ratio = _frame_quantity(frame_ids, frames_b, "pnp_inlier_ratio", 1.0)
    a_reprojection = _frame_quantity(
        frame_ids, frames_a, "pnp_reprojection_error_median_px", 0.0
    )
    b_reprojection = _frame_quantity(
        frame_ids, frames_b, "pnp_reprojection_error_median_px", 0.0
    )
    if version == "v0":
        weight_a = valid_a.astype(np.float64)
        weight_b = valid_b.astype(np.float64)
    elif version == "v1":
        weight_a = valid_a.astype(np.float64) * agreement_weight
        weight_b = valid_b.astype(np.float64) * agreement_weight
    elif version == "v2":
        a_inliers = _frame_quantity(frame_ids, frames_a, "pnp_inliers", 1.0)
        b_inliers = _frame_quantity(frame_ids, frames_b, "pnp_inliers", 1.0)
        a_shared = _frame_quantity(frame_ids, frames_a, "shared_e1_left_tracks", 1.0)
        b_shared = _frame_quantity(frame_ids, frames_b, "shared_e2_left_tracks", 1.0)
        quality_a = _quality_v2(
            valid_a, a_ratio, a_reprojection, a_inliers, a_shared,
            bool(summary_a.get("calibration_valid", True)),
        )
        quality_b = _quality_v2(
            valid_b, b_ratio, b_reprojection, b_inliers, b_shared,
            bool(summary_b.get("calibration_valid", True)),
        )
        weight_a = quality_a * agreement_weight
        weight_b = quality_b * agreement_weight
    else:
        raise ValueError(f"Unknown confidence version {version}")
    weight_a[0] = 1.0
    weight_b[0] = 1.0
    return AnchorConfidence(
        weight_a=weight_a,
        weight_b=weight_b,
        valid_a=valid_a,
        valid_b=valid_b,
        position_disagreement=position,
        position_disagreement_normalized=position_normalized,
        rotation_disagreement_deg=rotation,
        robust_z=robust_z,
        trajectory_step_scale=step_scale,
        a_inlier_ratio=a_ratio,
        b_inlier_ratio=b_ratio,
        a_reprojection_px=a_reprojection,
        b_reprojection_px=b_reprojection,
    )
