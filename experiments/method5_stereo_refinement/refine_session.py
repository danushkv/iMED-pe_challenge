"""Six-parameter session optimizer and geometry-only validity checks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .config import StereoRefinementConfig
from .geometry import (
    PreparedFrame,
    corrected_stereo,
    correction_magnitudes_deg,
    direction_angle_deg,
    frame_residuals,
    rotation_angle_deg,
)


@dataclass(frozen=True)
class FitResult:
    parameters: np.ndarray
    R_LR: np.ndarray
    t_LR: np.ndarray
    optimizer_success: bool
    optimizer_message: str
    nfev: int
    initial_cost: float
    final_cost: float
    runtime_seconds: float
    reprojection: dict[str, Any]


def _huber_vectors(residual: np.ndarray, delta: float) -> np.ndarray:
    values = np.asarray(residual, dtype=np.float64).reshape(-1, 2)
    norm = np.linalg.norm(values, axis=1)
    weight = np.ones_like(norm)
    large = norm > delta
    weight[large] = delta / np.maximum(norm[large], 1e-12)
    return (np.sqrt(weight)[:, None] * values).reshape(-1)


def _all_residuals(
    parameters: np.ndarray,
    frames: list[PreparedFrame],
    R_original: np.ndarray,
    t_original: np.ndarray,
    config: StereoRefinementConfig,
) -> np.ndarray:
    R_LR, t_LR = corrected_stereo(parameters, R_original, t_original)
    blocks: list[np.ndarray] = []
    total_points = sum(len(frame.x_e1l) for frame in frames)
    scale = np.sqrt(max(total_points, 1))
    for frame in frames:
        left, right, e2, _ = frame_residuals(
            frame, R_LR, t_LR, config.invalid_projection_penalty_px
        )
        blocks.extend(
            _huber_vectors(value, config.huber_delta_px) / scale
            for value in (left, right, e2)
        )
    blocks.append(
        np.sqrt(config.rotation_prior_weight)
        * np.asarray(parameters[:3], dtype=np.float64)
    )
    blocks.append(
        np.sqrt(config.translation_prior_weight)
        * np.asarray(parameters[3:], dtype=np.float64)
    )
    return np.concatenate(blocks)


def reprojection_statistics(
    frames: list[PreparedFrame],
    R_LR: np.ndarray,
    t_LR: np.ndarray,
    config: StereoRefinementConfig,
) -> dict[str, Any]:
    values: dict[str, list[np.ndarray]] = {"e1_left": [], "e1_right": [], "e2_left": []}
    valid_counts: list[int] = []
    total_counts: list[int] = []
    per_frame: list[dict[str, Any]] = []
    for frame in frames:
        left, right, e2, valid = frame_residuals(
            frame, R_LR, t_LR, config.invalid_projection_penalty_px
        )
        errors = [np.linalg.norm(item, axis=1) for item in (left, right, e2)]
        for name, error in zip(values, errors, strict=True):
            values[name].append(error)
        valid_counts.append(int(np.count_nonzero(valid)))
        total_counts.append(int(len(valid)))
        per_frame.append({
            "sequence": frame.sequence,
            "frame_id": frame.frame_id,
            "tracks": len(valid),
            "valid_geometry": int(np.count_nonzero(valid)),
            "e1_left_median_px": float(np.median(errors[0])),
            "e1_right_median_px": float(np.median(errors[1])),
            "e2_left_median_px": float(np.median(errors[2])),
        })
    summary: dict[str, Any] = {}
    all_views = []
    for name, chunks in values.items():
        merged = np.concatenate(chunks) if chunks else np.empty(0)
        finite = merged[np.isfinite(merged)]
        all_views.append(finite)
        summary[name] = {
            "median_px": float(np.median(finite)) if len(finite) else float("nan"),
            "p90_px": float(np.percentile(finite, 90)) if len(finite) else float("nan"),
        }
    merged_all = np.concatenate(all_views) if all_views else np.empty(0)
    summary["all_views"] = {
        "median_px": float(np.median(merged_all)) if len(merged_all) else float("nan"),
        "p90_px": float(np.percentile(merged_all, 90)) if len(merged_all) else float("nan"),
    }
    summary["valid_geometry_fraction"] = float(
        sum(valid_counts) / max(sum(total_counts), 1)
    )
    summary["per_frame"] = per_frame
    return summary


def fit_correction(
    frames: list[PreparedFrame],
    R_original: np.ndarray,
    t_original: np.ndarray,
    config: StereoRefinementConfig,
) -> FitResult:
    if len(frames) < config.minimum_subset_frames:
        raise RuntimeError(
            f"Need at least {config.minimum_subset_frames} frames; got {len(frames)}"
        )
    initial = np.zeros(6, dtype=np.float64)
    lower = -np.radians(
        [config.max_rotation_correction_deg] * 3
        + [config.max_translation_direction_correction_deg] * 3
    )
    upper = -lower
    initial_residual = _all_residuals(initial, frames, R_original, t_original, config)
    started = time.perf_counter()
    result = least_squares(
        _all_residuals,
        initial,
        args=(frames, R_original, t_original, config),
        bounds=(lower, upper),
        loss="linear",  # Vector Huber weights are already applied per observation.
        max_nfev=config.optimizer_max_nfev,
        x_scale="jac",
    )
    runtime = time.perf_counter() - started
    R_LR, t_LR = corrected_stereo(result.x, R_original, t_original)
    return FitResult(
        parameters=result.x,
        R_LR=R_LR,
        t_LR=t_LR,
        optimizer_success=bool(result.success and np.isfinite(result.x).all()),
        optimizer_message=str(result.message),
        nfev=int(result.nfev),
        initial_cost=float(0.5 * np.dot(initial_residual, initial_residual)),
        final_cost=float(result.cost),
        runtime_seconds=runtime,
        reprojection=reprojection_statistics(frames, R_LR, t_LR, config),
    )


def _fit_dict(fit: FitResult) -> dict[str, Any]:
    rotation_deg, translation_deg = correction_magnitudes_deg(fit.parameters)
    return {
        "parameters_rad": fit.parameters.tolist(),
        "delta_rotation_deg": rotation_deg,
        "delta_translation_direction_deg": translation_deg,
        "R_LR": fit.R_LR.tolist(),
        "t_LR": fit.t_LR.tolist(),
        "optimizer_success": fit.optimizer_success,
        "optimizer_message": fit.optimizer_message,
        "nfev": fit.nfev,
        "initial_cost": fit.initial_cost,
        "final_cost": fit.final_cost,
        "runtime_seconds": fit.runtime_seconds,
        "reprojection": fit.reprojection,
    }


def refine_and_validate(
    frames: list[PreparedFrame],
    R_original: np.ndarray,
    t_original: np.ndarray,
    config: StereoRefinementConfig,
) -> tuple[FitResult, bool, dict[str, Any]]:
    original_reprojection = reprojection_statistics(
        frames, R_original, t_original, config
    )
    full = fit_correction(frames, R_original, t_original, config)

    first = [frame for frame in frames if frame.temporal_fraction < 0.5]
    second = [frame for frame in frames if frame.temporal_fraction >= 0.5]
    split_fits: list[FitResult] = []
    if min(len(first), len(second)) >= config.minimum_subset_frames:
        split_fits = [
            fit_correction(first, R_original, t_original, config),
            fit_correction(second, R_original, t_original, config),
        ]

    sequence_fits: dict[str, FitResult] = {}
    for sequence in sorted({frame.sequence for frame in frames}):
        selected = [frame for frame in frames if frame.sequence == sequence]
        if len(selected) >= config.minimum_subset_frames:
            sequence_fits[sequence] = fit_correction(
                selected, R_original, t_original, config
            )

    if len(split_fits) == 2:
        split_rotation = rotation_angle_deg(split_fits[0].R_LR, split_fits[1].R_LR)
        split_translation = direction_angle_deg(split_fits[0].t_LR, split_fits[1].t_LR)
    else:
        split_rotation = split_translation = float("inf")

    sequence_rotation = [
        rotation_angle_deg(item.R_LR, full.R_LR) for item in sequence_fits.values()
    ]
    sequence_translation = [
        direction_angle_deg(item.t_LR, full.t_LR) for item in sequence_fits.values()
    ]
    sequence_rotation_p90 = (
        float(np.percentile(sequence_rotation, 90)) if len(sequence_rotation) >= 2 else 0.0
    )
    sequence_translation_p90 = (
        float(np.percentile(sequence_translation, 90))
        if len(sequence_translation) >= 2
        else 0.0
    )

    before_median = original_reprojection["all_views"]["median_px"]
    after_median = full.reprojection["all_views"]["median_px"]
    relative_improvement = (before_median - after_median) / max(before_median, 1e-12)
    per_view_p90_regression = {}
    for name in ("e1_left", "e1_right", "e2_left"):
        before = original_reprojection[name]["p90_px"]
        after = full.reprojection[name]["p90_px"]
        per_view_p90_regression[name] = (after - before) / max(before, 1e-12)

    rotation_deg, translation_deg = correction_magnitudes_deg(full.parameters)
    failures: list[str] = []
    if not full.optimizer_success:
        failures.append("full optimizer did not converge")
    if len(split_fits) != 2 or not all(item.optimizer_success for item in split_fits):
        failures.append("split-half fits unavailable or unconverged")
    if split_rotation > config.split_rotation_stability_deg:
        failures.append("split-half rotation correction unstable")
    if split_translation > config.split_translation_stability_deg:
        failures.append("split-half translation-direction correction unstable")
    if sequence_rotation_p90 > config.sequence_rotation_stability_p90_deg:
        failures.append("cross-sequence rotation correction unstable")
    if sequence_translation_p90 > config.sequence_translation_stability_p90_deg:
        failures.append("cross-sequence translation-direction correction unstable")
    if relative_improvement < config.minimum_relative_median_improvement:
        failures.append("median multi-view reprojection did not improve sufficiently")
    if any(
        value > config.maximum_per_view_p90_regression
        for value in per_view_p90_regression.values()
    ):
        failures.append("at least one view's p90 reprojection regressed")
    margin = 1.0 - config.bound_margin_fraction
    if rotation_deg >= margin * config.max_rotation_correction_deg:
        failures.append("rotation correction approached safety bound")
    if translation_deg >= margin * config.max_translation_direction_correction_deg:
        failures.append("translation correction approached safety bound")

    diagnostics = {
        "full": _fit_dict(full),
        "original_reprojection": original_reprojection,
        "relative_median_reprojection_improvement": relative_improvement,
        "per_view_p90_regression": per_view_p90_regression,
        "split_half": {
            "first_num_frames": len(first),
            "second_num_frames": len(second),
            "rotation_disagreement_deg": split_rotation,
            "translation_direction_disagreement_deg": split_translation,
            "fits": [_fit_dict(item) for item in split_fits],
        },
        "cross_sequence": {
            "num_fits": len(sequence_fits),
            "rotation_disagreement_p90_deg": sequence_rotation_p90,
            "translation_direction_disagreement_p90_deg": sequence_translation_p90,
            "fits": {name: _fit_dict(item) for name, item in sequence_fits.items()},
        },
        "correction_valid": not failures,
        "validity_failures": failures,
        "validity_uses_ground_truth": False,
    }
    return full, not failures, diagnostics

