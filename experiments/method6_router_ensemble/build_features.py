"""Inference-only Method-6 feature construction from saved diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.method4c_consensus.optimize import align_vo_displacements, vo_weights
from experiments.method4c_consensus.robust_alignment import (
    align_method1_to_reference,
    align_observers_to_reference,
)

from .config import EXPERT_NAMES, FusionConfig
from .load_experts import ObserverBundle


@dataclass(frozen=True)
class InferenceFeatures:
    frame_ids: np.ndarray
    feature_names: tuple[str, ...]
    feature_groups: tuple[str, ...]
    values: np.ndarray
    expert_centers: np.ndarray
    expert_valid: np.ndarray
    expert_rotations: np.ndarray
    rotation_source: np.ndarray
    delta_vo: np.ndarray
    vo_weights: np.ndarray
    trajectory_step_scale: float


def _number(mapping: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (bool, int, float)) and np.isfinite(float(value)):
            return float(value)
    return float("nan")


def _boolean(mapping: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            return float(value)
    return float("nan")


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(first @ second.T).magnitude()))


def _calibration_features(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "calib_rotation_p90_deg": _number(
            payload,
            "rotation_variation_p90_deg",
            "all_valid_rotation_variation_p90_deg",
        ),
        "calib_translation_direction_p90_deg": _number(
            payload,
            "translation_direction_variation_p90_deg",
            "all_valid_translation_direction_variation_p90_deg",
        ),
        "calib_reprojection_median_px": _number(
            payload, "reprojection_error_median_px"
        ),
        "calib_valid": _boolean(
            payload, "fixed_transform_appears_valid", "calibration_valid"
        ),
    }


def _expert_frame_features(
    name: str,
    diagnostic: dict[str, Any],
    *,
    translation_magnitude: float,
    translation_jump: float,
    rotation_jump_deg: float,
) -> dict[str, float]:
    prefix = f"{name}_"
    if name == "2b":
        shared_keys = ("shared_e2_left_tracks", "shared_3d_2d_tracks")
        translation_keys = (
            "common_translation_magnitude_baselines",
            "pnp_translation_magnitude_baselines",
        )
        cross_keys = ("e2_to_e1_matches",)
    else:
        shared_keys = ("shared_e1_left_tracks", "shared_3d_2d_tracks")
        translation_keys = ("raw_translation_magnitude_baselines",)
        cross_keys = ("cross_camera_matches",)
    return {
        prefix + "pnp_inliers": _number(diagnostic, "pnp_inliers"),
        prefix + "pnp_inlier_ratio": _number(diagnostic, "pnp_inlier_ratio"),
        prefix + "pnp_reprojection_px": _number(
            diagnostic,
            "pnp_reprojection_error_median_px",
            "pnp_reprojection_error_mean_px",
        ),
        prefix + "shared_tracks": _number(diagnostic, *shared_keys),
        prefix + "triangulated_points": _number(
            diagnostic, "valid_triangulated_3d"
        ),
        prefix + "diagnostic_translation_magnitude": _number(
            diagnostic, *translation_keys
        ),
        prefix + "trajectory_translation_magnitude": float(translation_magnitude),
        prefix + "translation_jump": float(translation_jump),
        prefix + "rotation_jump_deg": float(rotation_jump_deg),
        prefix + "pnp_success": _boolean(diagnostic, "pnp_success"),
        prefix + "cross_matches": _number(diagnostic, *cross_keys),
        prefix + "magsac_inliers": _number(
            diagnostic, "magsac_inliers", "cross_essential_inliers"
        ),
        prefix + "magsac_inlier_ratio": _number(
            diagnostic, "magsac_inlier_ratio", "cross_essential_inlier_ratio"
        ),
    }


def build_inference_features(
    *,
    bundle: ObserverBundle,
    e1_calibration: dict[str, Any],
    e2_calibration: dict[str, Any],
    fusion: FusionConfig,
) -> InferenceFeatures:
    aligned = align_observers_to_reference(
        bundle,
        iterations=fusion.alignment_iterations,
        huber_mad=fusion.alignment_huber_mad,
    )
    if aligned.names != EXPERT_NAMES:
        raise RuntimeError(
            f"Expected deterministic expert order {EXPERT_NAMES}, got {aligned.names}"
        )
    vo_similarity = align_method1_to_reference(
        bundle,
        aligned.reference_centers,
        iterations=fusion.alignment_iterations,
        huber_mad=fusion.alignment_huber_mad,
    )
    _, delta_vo = align_vo_displacements(bundle, vo_similarity)
    edge_weights = vo_weights(bundle)
    num_frames = len(bundle.frame_ids)

    expert_rotations = np.stack(
        [bundle.observers[name].rotations_initial_from_camera for name in EXPERT_NAMES]
    )
    method1_rotations = bundle.method1.rotations_initial_from_camera
    calibration = {
        "2a": _calibration_features(e1_calibration),
        "2b": _calibration_features(e2_calibration),
        "loftr": _calibration_features(e1_calibration),
    }

    rows: list[dict[str, float]] = []
    groups_by_name: dict[str, str] = {}
    important: set[str] = set()
    for frame_index, frame_id in enumerate(bundle.frame_ids):
        row: dict[str, float] = {}
        for expert_index, name in enumerate(EXPERT_NAMES):
            center = aligned.centers[expert_index, frame_index]
            translation_magnitude = float(np.linalg.norm(center))
            if frame_index == 0:
                translation_jump = 0.0
                rotation_jump = 0.0
            else:
                translation_jump = float(
                    np.linalg.norm(
                        center - aligned.centers[expert_index, frame_index - 1]
                    )
                )
                rotation_jump = _rotation_distance_deg(
                    expert_rotations[expert_index, frame_index - 1],
                    expert_rotations[expert_index, frame_index],
                )
            diagnostic = bundle.observers[name].frames.get(int(frame_id), {})
            values = _expert_frame_features(
                name,
                diagnostic,
                translation_magnitude=translation_magnitude,
                translation_jump=translation_jump,
                rotation_jump_deg=rotation_jump,
            )
            row.update(values)
            for key in values:
                groups_by_name[key] = "r1"
            important.update(
                key
                for key in values
                if any(
                    token in key
                    for token in (
                        "pnp_inliers",
                        "pnp_inlier_ratio",
                        "pnp_reprojection",
                        "shared_tracks",
                        "triangulated_points",
                        "translation_jump",
                        "rotation_jump",
                    )
                )
            )
            for suffix, value in calibration[name].items():
                key = f"{name}_{suffix}"
                row[key] = value
                groups_by_name[key] = "r1"

        pairs = ((0, 1), (0, 2), (1, 2))
        for first, second in pairs:
            left, right = EXPERT_NAMES[first], EXPERT_NAMES[second]
            key = f"disagreement_position_{left}_{right}"
            distance = float(
                np.linalg.norm(
                    aligned.centers[first, frame_index]
                    - aligned.centers[second, frame_index]
                )
            )
            row[key] = distance
            row[key + "_normalized"] = distance / max(
                aligned.trajectory_step_scale, 1.0e-12
            )
            row[f"disagreement_rotation_deg_{left}_{right}"] = (
                _rotation_distance_deg(
                    expert_rotations[first, frame_index],
                    expert_rotations[second, frame_index],
                )
            )
            groups_by_name[key] = "r2"
            groups_by_name[key + "_normalized"] = "r2"
            groups_by_name[f"disagreement_rotation_deg_{left}_{right}"] = "r2"

        for expert_index, name in enumerate(EXPERT_NAMES):
            if frame_index == 0:
                translation_innovation = 0.0
                rotation_innovation = 0.0
            else:
                expert_delta = (
                    aligned.centers[expert_index, frame_index]
                    - aligned.centers[expert_index, frame_index - 1]
                )
                translation_innovation = float(
                    np.linalg.norm(expert_delta - delta_vo[frame_index - 1])
                )
                expert_relative = (
                    expert_rotations[expert_index, frame_index - 1].T
                    @ expert_rotations[expert_index, frame_index]
                )
                vo_relative = (
                    method1_rotations[frame_index - 1].T
                    @ method1_rotations[frame_index]
                )
                rotation_innovation = _rotation_distance_deg(
                    expert_relative, vo_relative
                )
            row[f"vo_translation_innovation_{name}"] = translation_innovation
            row[f"vo_rotation_innovation_deg_{name}"] = rotation_innovation
            groups_by_name[f"vo_translation_innovation_{name}"] = "r2"
            groups_by_name[f"vo_rotation_innovation_deg_{name}"] = "r2"

        vo_diagnostic = bundle.method1.frames.get(int(frame_id), {})
        for key, value in {
            "vo_pnp_inliers": _number(vo_diagnostic, "pnp_inliers"),
            "vo_pnp_inlier_ratio": _number(vo_diagnostic, "pnp_inlier_ratio"),
            "vo_pnp_reprojection_px": _number(
                vo_diagnostic,
                "pnp_reprojection_error_median_px",
                "pnp_reprojection_error_mean_px",
            ),
            "vo_pnp_success": _boolean(vo_diagnostic, "pnp_success"),
        }.items():
            row[key] = value
            groups_by_name[key] = "r2"
        rows.append(row)

    # Previous value and first difference are inference-only temporal history.
    for key in sorted(important):
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
        previous = np.concatenate(([values[0]], values[:-1]))
        difference = values - previous
        previous_name = key + "__previous"
        difference_name = key + "__difference"
        for index, row in enumerate(rows):
            row[previous_name] = float(previous[index])
            row[difference_name] = float(difference[index])
        groups_by_name[previous_name] = groups_by_name[key]
        groups_by_name[difference_name] = groups_by_name[key]

    base_names = tuple(sorted(groups_by_name))
    base = np.asarray(
        [[row.get(name, np.nan) for name in base_names] for row in rows],
        dtype=np.float64,
    )
    missing = (~np.isfinite(base)).astype(np.float64)
    missing_names = tuple(name + "__missing" for name in base_names)
    feature_names = base_names + missing_names
    feature_groups = tuple(groups_by_name[name] for name in base_names) * 2
    values = np.concatenate((base, missing), axis=1)
    return InferenceFeatures(
        frame_ids=bundle.frame_ids.copy(),
        feature_names=feature_names,
        feature_groups=feature_groups,
        values=values,
        expert_centers=aligned.centers.copy(),
        expert_valid=aligned.valid.copy(),
        expert_rotations=expert_rotations,
        rotation_source=bundle.rotation_source.rotations_initial_from_camera.copy(),
        delta_vo=delta_vo,
        vo_weights=edge_weights,
        trajectory_step_scale=aligned.trajectory_step_scale,
    )

