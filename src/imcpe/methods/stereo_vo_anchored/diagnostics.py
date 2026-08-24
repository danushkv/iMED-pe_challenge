"""Diagnostic plots for Method 1.5.

Ground truth is read only here, after inference, and is never passed to anchor
generation or trajectory fusion.  The similarity alignment in this module is
used for visualization only; reported metrics come from the repository's
official evaluation path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .anchor_estimator import load_anchor_cache
from .config import AnchorThresholds


def _frame_id(value: str) -> str:
    """Return a stable identifier shared by dataset and prediction pose files."""
    return Path(value).stem


def _rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = (float(np.trace(rotation)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _anchor_array(anchor: dict[str, Any], *candidates: str) -> np.ndarray:
    for candidate in candidates:
        if candidate in anchor:
            return np.asarray(anchor[candidate])
    raise KeyError(f"Anchor cache contains none of: {', '.join(candidates)}")


def read_pose_file(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Read challenge pose rows as frame identifier plus a row-major 4x4 pose."""
    names: list[str] = []
    poses: list[np.ndarray] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 17:
                raise ValueError(f"Malformed pose row {line_number} in {path}")
            values = np.asarray([float(value) for value in fields[-16:]], dtype=np.float64)
            names.append(_frame_id(fields[0]))
            poses.append(values.reshape(4, 4))
    if not poses:
        raise ValueError(f"No poses found in {path}")
    return names, np.stack(poses)


def _select_common(
    reference_names: list[str],
    reference_poses: np.ndarray,
    prediction_names: list[str],
    prediction_poses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    prediction_by_name = {
        name: pose for name, pose in zip(prediction_names, prediction_poses, strict=True)
    }
    reference: list[np.ndarray] = []
    prediction: list[np.ndarray] = []
    common_names: list[str] = []
    for name, pose in zip(reference_names, reference_poses, strict=True):
        if name in prediction_by_name:
            common_names.append(name)
            reference.append(pose)
            prediction.append(prediction_by_name[name])
    if len(common_names) < 3:
        raise ValueError("At least three common poses are required for alignment")
    return np.stack(reference), np.stack(prediction), common_names


def _similarity_align_points(
    prediction: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Horn/Umeyama Sim(3) alignment for plotting translations only."""
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    prediction_mean = prediction.mean(axis=0)
    reference_mean = reference.mean(axis=0)
    prediction_centered = prediction - prediction_mean
    reference_centered = reference - reference_mean

    covariance = reference_centered.T @ prediction_centered / prediction.shape[0]
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u_matrix @ vt_matrix) < 0:
        sign[-1] = -1.0
    rotation = u_matrix @ np.diag(sign) @ vt_matrix
    variance = np.mean(np.sum(prediction_centered * prediction_centered, axis=1))
    if variance <= np.finfo(np.float64).eps:
        raise ValueError("Degenerate predicted trajectory cannot be similarity-aligned")
    scale = float(np.sum(singular_values * sign) / variance)
    translation = reference_mean - scale * (rotation @ prediction_mean)
    aligned = (scale * (rotation @ prediction.T)).T + translation
    return aligned, scale, rotation, translation


def _aligned_translation_series(
    ground_truth_path: Path, prediction_path: Path
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    gt_names, gt_poses = read_pose_file(ground_truth_path)
    pred_names, pred_poses = read_pose_file(prediction_path)
    gt_common, pred_common, names = _select_common(
        gt_names, gt_poses, pred_names, pred_poses
    )
    ground_truth = gt_common[:, :3, 3]
    prediction = pred_common[:, :3, 3]
    aligned, _, _, _ = _similarity_align_points(prediction, ground_truth)
    return ground_truth, aligned, names


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _numeric_series(
    rows: Iterable[dict[str, Any]], candidates: tuple[str, ...], length: int
) -> np.ndarray:
    result = np.full(length, np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        if index >= length:
            break
        for candidate in candidates:
            value = row.get(candidate)
            if value is not None:
                try:
                    result[index] = float(value)
                except (TypeError, ValueError):
                    pass
                break
    return result


def _event_series(rows: Iterable[dict[str, Any]], length: int) -> np.ndarray:
    result = np.zeros(length, dtype=bool)
    candidates = ("anchor_applied", "correction_applied", "applied")
    for index, row in enumerate(rows):
        if index >= length:
            break
        result[index] = any(bool(row.get(candidate, False)) for candidate in candidates)
    return result


def _increment_series(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation = np.zeros(len(poses), dtype=np.float64)
    rotation = np.zeros(len(poses), dtype=np.float64)
    for index in range(1, len(poses)):
        local_increment = np.linalg.inv(poses[index - 1]) @ poses[index]
        translation[index] = np.linalg.norm(local_increment[:3, 3])
        rotation[index] = _rotation_angle_degrees(local_increment[:3, :3])
    return translation, rotation


def _save_trajectory_plot(
    output_path: Path,
    ground_truth: np.ndarray,
    baseline: np.ndarray,
    method_1: np.ndarray,
    method_1_5: np.ndarray,
    title: str,
) -> None:
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    for points, label, color in (
        (ground_truth, "Ground truth", "black"),
        (baseline, "Baseline", "tab:orange"),
        (method_1, "Method 1", "tab:blue"),
        (method_1_5, "Method 1.5", "tab:green"),
    ):
        axis.plot(points[:, 0], points[:, 1], points[:, 2], label=label, color=color)
        axis.scatter(*points[0], color=color, marker="o", s=25)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def create_sequence_diagnostics(
    *,
    sequence: str,
    ground_truth_pose: str | Path,
    baseline_pose: str | Path,
    method_1_pose: str | Path,
    method_1_5_pose: str | Path,
    anchor_cache: str | Path,
    method_1_diagnostics: str | Path,
    fusion_diagnostics: str | Path,
    output_dir: str | Path,
    thresholds_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create all requested post-inference plots for one sequence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt, baseline, gt_names = _aligned_translation_series(
        Path(ground_truth_pose), Path(baseline_pose)
    )
    gt_m1, method_1, names_m1 = _aligned_translation_series(
        Path(ground_truth_pose), Path(method_1_pose)
    )
    gt_m15, method_1_5, names_m15 = _aligned_translation_series(
        Path(ground_truth_pose), Path(method_1_5_pose)
    )
    if not (gt_names == names_m1 == names_m15):
        raise ValueError("Pose files do not contain the same ordered frame identifiers")
    if not (np.allclose(gt, gt_m1) and np.allclose(gt, gt_m15)):
        raise ValueError("Ground-truth subsets differ across predictions")

    _save_trajectory_plot(
        output_dir / "trajectory.png", gt, baseline, method_1, method_1_5, sequence
    )

    frame_axis = np.arange(len(gt))
    method_1_error = np.linalg.norm(method_1 - gt, axis=1)
    method_1_5_error = np.linalg.norm(method_1_5 - gt, axis=1)
    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.plot(frame_axis, method_1_error, label="Method 1", color="tab:blue")
    axis.plot(frame_axis, method_1_5_error, label="Method 1.5", color="tab:green")
    axis.set_xlabel("Frame index")
    axis.set_ylabel("Aligned point error")
    axis.set_title(f"{sequence}: aligned point error versus frame")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "error_vs_frame.png", dpi=180)
    plt.close(figure)

    _, method_1_poses = read_pose_file(method_1_pose)
    anchor = load_anchor_cache(anchor_cache)
    anchor_rotation = np.asarray(
        _anchor_array(anchor, "R_anchor_rel", "relative_rotation"), dtype=np.float64
    )
    common_length = min(len(method_1_poses), len(anchor_rotation))
    disagreement = np.asarray(
        [
            _rotation_angle_degrees(
                method_1_poses[index, :3, :3].T @ anchor_rotation[index]
            )
            for index in range(common_length)
        ]
    )
    fusion_json = _load_json(Path(fusion_diagnostics))
    fusion_rows = fusion_json.get("frames", [])
    anchor_events = _event_series(fusion_rows, common_length)
    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.plot(np.arange(common_length), disagreement, color="tab:purple")
    event_indices = np.flatnonzero(anchor_events)
    if len(event_indices):
        axis.scatter(
            event_indices,
            disagreement[event_indices],
            marker="|",
            s=100,
            color="tab:red",
            label="Applied anchor",
        )
        axis.legend()
    axis.set_xlabel("Frame index")
    axis.set_ylabel("Angular disagreement (degrees)")
    axis.set_title(f"{sequence}: Method 1 orientation versus anchor")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "orientation_disagreement.png", dpi=180)
    plt.close(figure)

    translation_increment, rotation_increment = _increment_series(method_1_poses)
    method_1_json = _load_json(Path(method_1_diagnostics))
    pnp_rows = method_1_json.get("frames", [])
    pnp_inliers = _numeric_series(
        pnp_rows, ("pnp_inliers", "num_pnp_inliers", "inliers"), len(method_1_poses)
    )
    anchor_inlier_ratio = np.asarray(
        _anchor_array(anchor, "inlier_ratio", "essential_inlier_ratio"),
        dtype=np.float64,
    )
    diagnostic_length = min(len(method_1_poses), len(anchor_inlier_ratio))
    figure, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(translation_increment, color="tab:blue")
    axes[0].set_ylabel("VO |translation|")
    axes[1].plot(rotation_increment, color="tab:orange")
    axes[1].set_ylabel("VO rotation (deg)")
    axes[2].plot(pnp_inliers, color="tab:green")
    axes[2].set_ylabel("PnP inliers")
    axes[3].plot(
        np.arange(diagnostic_length),
        anchor_inlier_ratio[:diagnostic_length],
        color="tab:purple",
    )
    axes[3].set_ylabel("Anchor inlier ratio")
    axes[3].set_xlabel("Frame index")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(f"{sequence}: increment and anchor diagnostics")
    figure.tight_layout()
    figure.savefig(output_dir / "increment_diagnostics.png", dpi=180)
    plt.close(figure)

    confidence_fields = (
        (("matches", "num_matches"), "Matches"),
        (("essential_inliers", "num_essential_inliers"), "Essential inliers"),
        (("inlier_ratio", "essential_inlier_ratio"), "Inlier ratio"),
        (("cheirality_fraction",), "Cheirality fraction"),
        (("median_sampson_error", "sampson_error"), "Median Sampson error"),
    )
    thresholds = AnchorThresholds.load_json(thresholds_path) if thresholds_path else None
    threshold_values = {
        "matches": getattr(thresholds, "min_matches", None) if thresholds else None,
        "essential_inliers": getattr(thresholds, "min_inliers", None) if thresholds else None,
        "inlier_ratio": getattr(thresholds, "min_inlier_ratio", None) if thresholds else None,
        "cheirality_fraction": getattr(thresholds, "min_cheirality_fraction", None)
        if thresholds
        else None,
        "median_sampson_error": getattr(thresholds, "max_median_sampson_error", None)
        if thresholds
        else None,
    }
    figure, axes = plt.subplots(len(confidence_fields), 1, figsize=(13, 12), sharex=True)
    for axis, (fields, label) in zip(axes, confidence_fields, strict=True):
        field = fields[0]
        values = np.asarray(_anchor_array(anchor, *fields), dtype=np.float64)
        axis.plot(values, linewidth=1.0)
        threshold = threshold_values[field]
        if threshold is not None and np.isfinite(float(threshold)):
            axis.axhline(float(threshold), color="tab:red", linestyle="--", label="gate")
            axis.legend(loc="best")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Frame index")
    figure.suptitle(f"{sequence}: cross-camera anchor confidence")
    figure.tight_layout()
    figure.savefig(output_dir / "anchor_confidence.png", dpi=180)
    plt.close(figure)

    result = {
        "sequence": sequence,
        "frames": len(gt),
        "method_1_mean_aligned_point_error": float(np.mean(method_1_error)),
        "method_1_5_mean_aligned_point_error": float(np.mean(method_1_5_error)),
        "median_orientation_disagreement_degrees": float(np.median(disagreement)),
        "maximum_orientation_disagreement_degrees": float(np.max(disagreement)),
        "anchor_events_applied": int(np.sum(anchor_events)),
        "note": "Point errors use plotting-only Sim(3) alignment; official metrics are reported elsewhere.",
    }
    with (output_dir / "diagnostic_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
