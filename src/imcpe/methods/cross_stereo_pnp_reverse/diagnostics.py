"""Post-inference A-vs-B diagnostics. Ground truth is read only here."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imcpe.alignment import horn_align_sim3
from imcpe.io_pose import PoseRow, read_pose_txt


SESSION_005_WINDOWS = {
    "session_005_scene_7_circular": (74, 80),
    "session_005_scene_7_left_right": (416, 422),
    "session_005_scene_7_zoom_in": (566, 572),
}


def _session_id(sequence: str) -> str:
    parts = sequence.split("_")
    if len(parts) < 2 or not parts[1].isdigit():
        raise ValueError(f"Cannot parse session from {sequence}")
    return parts[1].zfill(3)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _calibration_quality(path: Path) -> dict[str, Any]:
    payload = _read_json(path)

    def first(*keys: str) -> float:
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return float(value)
        return float("nan")

    return {
        "rotation_p50_deg": first(
            "rotation_p50_deg", "all_valid_rotation_variation_median_deg"
        ),
        "rotation_p90_deg": first(
            "rotation_p90_deg", "all_valid_rotation_variation_p90_deg"
        ),
        "translation_direction_p50_deg": first(
            "translation_direction_p50_deg",
            "all_valid_translation_direction_variation_median_deg",
        ),
        "translation_direction_p90_deg": first(
            "translation_direction_p90_deg",
            "all_valid_translation_direction_variation_p90_deg",
        ),
        "median_epipolar_error_px": first(
            "median_epipolar_error_px",
            "pooled_inlier_epipolar_sampson_median_approx_px",
        ),
        "median_stereo_reprojection_error_px": first(
            "reprojection_error_median_px"
        ),
        "num_sampled": int(payload.get("num_sampled", 0)),
        "num_valid": int(payload.get("num_individually_valid", 0)),
        "valid": payload.get("fixed_transform_appears_valid"),
        "validity_failures": payload.get("validity_failures", []),
    }


def _summary(root: Path, split: str, sequence: str) -> dict[str, Any]:
    return _read_json(root / split / sequence / "diagnostics.json").get(
        "summary", {}
    )


def _finite_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"median": float("nan"), "p90": float("nan")}
    return {
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }


def _pearson_summary(x_values: list[float], y_values: list[float]) -> dict[str, float | int]:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    correlation = (
        float(np.corrcoef(x, y)[0, 1])
        if x.size >= 2 and np.std(x) > 0.0 and np.std(y) > 0.0
        else float("nan")
    )
    return {"num_sessions": int(x.size), "pearson_r": correlation}


def _rotation_angle(R: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _load_common_raw(
    method2a_cache: Path, method2b_cache: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(method2a_cache, allow_pickle=False) as archive_a:
        ids_a = np.asarray(archive_a["frame_ids"], dtype=np.int64)
        transforms_a = np.asarray(archive_a["T_E2_from_E1"], dtype=np.float64)
        success_a = np.asarray(archive_a["pnp_success"], dtype=bool)
    with np.load(method2b_cache, allow_pickle=False) as archive_b:
        ids_b = np.asarray(archive_b["frame_ids"], dtype=np.int64)
        transforms_b = np.asarray(archive_b["T_E2_from_E1"], dtype=np.float64)
        success_b = np.asarray(archive_b["pnp_success"], dtype=bool)
    index_a = {int(frame_id): index for index, frame_id in enumerate(ids_a)}
    index_b = {int(frame_id): index for index, frame_id in enumerate(ids_b)}
    common_ids = sorted(set(index_a) & set(index_b))
    selected_ids: list[int] = []
    selected_a: list[np.ndarray] = []
    selected_b: list[np.ndarray] = []
    selected_success_a: list[bool] = []
    selected_success_b: list[bool] = []
    for frame_id in common_ids:
        ia = index_a[frame_id]
        ib = index_b[frame_id]
        selected_ids.append(frame_id)
        selected_a.append(transforms_a[ia])
        selected_b.append(transforms_b[ib])
        selected_success_a.append(bool(success_a[ia]))
        selected_success_b.append(bool(success_b[ib]))
    return (
        np.asarray(selected_ids, dtype=np.int64),
        np.asarray(selected_a, dtype=np.float64),
        np.asarray(selected_b, dtype=np.float64),
        np.asarray(selected_success_a, dtype=bool),
        np.asarray(selected_success_b, dtype=bool),
    )


def _disagreement(
    frame_ids: np.ndarray,
    transforms_a: np.ndarray,
    transforms_b: np.ndarray,
    success_a: np.ndarray,
    success_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = success_a & success_b
    valid &= np.isfinite(transforms_a).all(axis=(1, 2))
    valid &= np.isfinite(transforms_b).all(axis=(1, 2))
    selected_ids = frame_ids[valid]
    rotations: list[float] = []
    directions: list[float] = []
    keep_ids: list[int] = []
    for frame_id, transform_a, transform_b in zip(
        selected_ids, transforms_a[valid], transforms_b[valid], strict=True
    ):
        direction_a = transform_a[:3, 3]
        direction_b = transform_b[:3, 3]
        norm_a = float(np.linalg.norm(direction_a))
        norm_b = float(np.linalg.norm(direction_b))
        if norm_a <= 1e-9 or norm_b <= 1e-9:
            continue
        rotations.append(
            _rotation_angle(transform_a[:3, :3].T @ transform_b[:3, :3])
        )
        cosine = float(
            np.clip(np.dot(direction_a, direction_b) / (norm_a * norm_b), -1.0, 1.0)
        )
        directions.append(float(np.degrees(np.arccos(cosine))))
        keep_ids.append(int(frame_id))
    return (
        np.asarray(keep_ids, dtype=np.int64),
        np.asarray(rotations, dtype=np.float64),
        np.asarray(directions, dtype=np.float64),
    )


def _valid_pose_map(rows: list[PoseRow]) -> dict[int, PoseRow]:
    return {
        row.frame_idx: row
        for row in rows
        if np.isfinite(row.t).all() and np.isfinite(row.q_xyzw).all()
    }


def _aligned_xyz(
    ground_truth: list[PoseRow], prediction: list[PoseRow]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = _valid_pose_map(ground_truth)
    pred = _valid_pose_map(prediction)
    ids = np.asarray(sorted(set(gt) & set(pred)), dtype=np.int64)
    if ids.size < 3:
        raise RuntimeError("At least three common poses are required")
    gt_xyz = np.asarray([gt[int(frame_id)].t for frame_id in ids])
    pred_xyz = np.asarray([pred[int(frame_id)].t for frame_id in ids])
    rotation, translation, _, scale = horn_align_sim3(pred_xyz.T, gt_xyz.T)
    aligned = (scale * rotation @ pred_xyz.T + translation).T
    return ids, gt_xyz, aligned


def _adjacent_jump(
    frame_ids: np.ndarray,
    transforms: np.ndarray,
    success: np.ndarray,
    window: tuple[int, int],
) -> dict[str, float | int]:
    magnitudes = np.linalg.norm(transforms[:, :3, 3], axis=1)
    best = {"ratio": float("nan"), "from_frame": -1, "to_frame": -1}
    maximum = -np.inf
    for index in range(1, len(frame_ids)):
        if not success[index - 1] or not success[index]:
            continue
        previous_id = int(frame_ids[index - 1])
        current_id = int(frame_ids[index])
        if not (
            window[0] <= previous_id <= window[1]
            and window[0] <= current_id <= window[1]
        ):
            continue
        previous = float(magnitudes[index - 1])
        current = float(magnitudes[index])
        ratio = max(previous, current) / max(min(previous, current), 1e-9)
        if ratio > maximum:
            maximum = ratio
            best = {
                "ratio": ratio,
                "from_frame": previous_id,
                "to_frame": current_id,
                "from_magnitude": previous,
                "to_magnitude": current,
            }
    return best


def _jump_classification(ratio_a: float, ratio_b: float) -> str:
    if not np.isfinite(ratio_b):
        return "D: Method 2B has no comparable successful pair"
    if ratio_b <= 1.5:
        return "A: removes the jump"
    if np.isfinite(ratio_a) and 0.8 * ratio_a <= ratio_b <= 1.2 * ratio_a:
        return "C: reproduces a similar jump"
    if np.isfinite(ratio_a) and ratio_b < ratio_a:
        return "B: reduces but does not remove the jump"
    return "D: creates a different/equal-or-larger failure"


def create_method2ab_diagnostics(
    *,
    data_root: Path,
    split: str,
    comparison_json: Path,
    e1_calibration_root: Path,
    e2_calibration_root: Path,
    method2a_root: Path,
    method2b_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Create all released-data A/B diagnostics without changing predictions."""

    output_root.mkdir(parents=True, exist_ok=True)
    comparison = _read_json(comparison_json)
    per_sequence = comparison["per_sequence"]
    sessions = sorted({_session_id(sequence) for sequence in per_sequence})
    session_table: dict[str, Any] = {}
    disagreement_by_session: dict[str, dict[str, list[float]]] = {
        session: {"rotation": [], "translation_direction": []}
        for session in sessions
    }

    for sequence in sorted(per_sequence):
        session = _session_id(sequence)
        ids, transforms_a, transforms_b, success_a, success_b = _load_common_raw(
            method2a_root / split / sequence / "raw_poses.npz",
            method2b_root / split / sequence / "raw_poses.npz",
        )
        disagreement_ids, rotation, translation = _disagreement(
            ids, transforms_a, transforms_b, success_a, success_b
        )
        disagreement_by_session[session]["rotation"].extend(rotation.tolist())
        disagreement_by_session[session]["translation_direction"].extend(
            translation.tolist()
        )
        sequence_output = output_root / split / sequence
        sequence_output.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(disagreement_ids, rotation)
        axes[0].set_ylabel("Rotation disagreement (deg)")
        axes[1].plot(disagreement_ids, translation)
        axes[1].set_ylabel("Translation-direction disagreement (deg)")
        axes[1].set_xlabel("Frame ID")
        for axis in axes:
            axis.grid(alpha=0.25)
        figure.suptitle(sequence)
        figure.tight_layout()
        figure.savefig(sequence_output / "method2a_vs_2b_disagreement.png", dpi=180)
        plt.close(figure)

    for session in sessions:
        sequences = [
            sequence for sequence in per_sequence if _session_id(sequence) == session
        ]
        summaries_a = [_summary(method2a_root, split, sequence) for sequence in sequences]
        summaries_b = [_summary(method2b_root, split, sequence) for sequence in sequences]
        rotation_summary = _finite_summary(
            disagreement_by_session[session]["rotation"]
        )
        translation_summary = _finite_summary(
            disagreement_by_session[session]["translation_direction"]
        )
        session_table[session] = {
            "sequences": sequences,
            "e1_stereo": _calibration_quality(
                e1_calibration_root / f"session_{session}_diagnostics.json"
            ),
            "e2_stereo": _calibration_quality(
                e2_calibration_root / f"session_{session}_diagnostics.json"
            ),
            "method2a": {
                "mean_ate_mm": float(
                    np.mean(
                        [per_sequence[sequence]["method2a_mean_ate_mm"] for sequence in sequences]
                    )
                ),
                "median_shared_tracks": float(
                    np.median([item.get("median_shared_3d_2d_tracks", np.nan) for item in summaries_a])
                ),
                "median_pnp_inliers": float(
                    np.median([item.get("median_pnp_inliers", np.nan) for item in summaries_a])
                ),
                "median_reprojection_error_px": float(
                    np.median([item.get("median_pnp_reprojection_error_px", np.nan) for item in summaries_a])
                ),
            },
            "method2b": {
                "mean_ate_mm": float(
                    np.mean(
                        [per_sequence[sequence]["method2b_mean_ate_mm"] for sequence in sequences]
                    )
                ),
                "median_shared_tracks": float(
                    np.median([item.get("median_shared_3d_2d_tracks", np.nan) for item in summaries_b])
                ),
                "median_pnp_inliers": float(
                    np.median([item.get("median_pnp_inliers", np.nan) for item in summaries_b])
                ),
                "median_reprojection_error_px": float(
                    np.median([item.get("median_pnp_reprojection_error_px", np.nan) for item in summaries_b])
                ),
            },
            "a_vs_b_disagreement": {
                "num_frames": len(disagreement_by_session[session]["rotation"]),
                "rotation_median_deg": rotation_summary["median"],
                "rotation_p90_deg": rotation_summary["p90"],
                "translation_direction_median_deg": translation_summary["median"],
                "translation_direction_p90_deg": translation_summary["p90"],
            },
        }

    scatter_specs = [
        ("translation_direction_p90_deg", "mean_ate_mm", "Translation direction p90"),
        ("rotation_p90_deg", "mean_ate_mm", "Rotation p90"),
        ("median_stereo_reprojection_error_px", "mean_ate_mm", "Stereo reprojection"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for column, (quality_key, _, title) in enumerate(scatter_specs):
        for row, (stereo_key, method_key, label) in enumerate(
            (("e1_stereo", "method2a", "E1 / Method 2A"), ("e2_stereo", "method2b", "E2 / Method 2B"))
        ):
            axis = axes[row, column]
            x = [session_table[s][stereo_key][quality_key] for s in sessions]
            y = [session_table[s][method_key]["mean_ate_mm"] for s in sessions]
            axis.scatter(x, y)
            for session, x_value, y_value in zip(sessions, x, y, strict=True):
                axis.annotate(session, (x_value, y_value))
            axis.set_xlabel(f"{title} (deg)" if "p90" in quality_key else f"{title} (px)")
            axis.set_ylabel("Session mean ATE (mm)")
            axis.set_title(label)
            axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_root / "calibration_quality_vs_ate.png", dpi=180)
    plt.close(figure)

    relationships: dict[str, Any] = {}
    for stereo_key, method_key, method_label in (
        ("e1_stereo", "method2a", "method2a"),
        ("e2_stereo", "method2b", "method2b"),
    ):
        for quality_key in (
            "translation_direction_p90_deg",
            "rotation_p90_deg",
            "median_stereo_reprojection_error_px",
        ):
            relationships[f"{stereo_key}_{quality_key}_vs_{method_label}_ate"] = (
                _pearson_summary(
                    [session_table[s][stereo_key][quality_key] for s in sessions],
                    [session_table[s][method_key]["mean_ate_mm"] for s in sessions],
                )
            )

    session_005: dict[str, Any] = {}
    for sequence, window in SESSION_005_WINDOWS.items():
        if sequence not in per_sequence:
            continue
        cache_a = method2a_root / split / sequence / "raw_poses.npz"
        cache_b = method2b_root / split / sequence / "raw_poses.npz"
        ids, transforms_a, transforms_b, success_a, success_b = _load_common_raw(
            cache_a, cache_b
        )
        jump_a = _adjacent_jump(ids, transforms_a, success_a, window)
        jump_b = _adjacent_jump(ids, transforms_b, success_b, window)
        session_005[sequence] = {
            "window": list(window),
            "method2a_jump": jump_a,
            "method2b_jump": jump_b,
            "classification": _jump_classification(
                float(jump_a["ratio"]), float(jump_b["ratio"])
            ),
        }
        gt = read_pose_txt(data_root / split / sequence / "pose.txt")
        rows_a = read_pose_txt(method2a_root / split / sequence / "pose.txt")
        rows_b = read_pose_txt(method2b_root / split / sequence / "pose.txt")
        ids_a, gt_a, aligned_a = _aligned_xyz(gt, rows_a)
        ids_b, _, aligned_b = _aligned_xyz(gt, rows_b)
        gt_map = _valid_pose_map(gt)
        gt_ids = sorted(gt_map)
        gt_xyz = np.asarray([gt_map[frame_id].t for frame_id in gt_ids])
        sequence_output = output_root / split / sequence
        sequence_output.mkdir(parents=True, exist_ok=True)
        figure = plt.figure(figsize=(10, 8))
        axis = figure.add_subplot(111, projection="3d")
        axis.plot(*gt_xyz.T, color="black", linewidth=2, label="Ground truth")
        axis.plot(*aligned_a.T, label="Method 2A")
        axis.plot(*aligned_b.T, label="Method 2B")
        axis.set_title(sequence)
        axis.legend()
        figure.tight_layout()
        figure.savefig(sequence_output / "trajectory_2a_vs_2b.png", dpi=180)
        plt.close(figure)

        magnitude_a = np.linalg.norm(transforms_a[:, :3, 3], axis=1)
        magnitude_b = np.linalg.norm(transforms_b[:, :3, 3], axis=1)
        figure, axis = plt.subplots(figsize=(12, 5))
        axis.plot(ids, magnitude_a, label="Method 2A raw |t|")
        axis.plot(ids, magnitude_b, label="Method 2B raw |t|")
        axis.axvspan(window[0], window[1], color="red", alpha=0.15)
        axis.set_xlabel("Frame ID")
        axis.set_ylabel("Own stereo-baseline units")
        axis.set_title(f"{sequence}: compare direction-specific scales qualitatively")
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(sequence_output / "translation_magnitude_jumps.png", dpi=180)
        plt.close(figure)

    result = {
        "split": split,
        "session_table": session_table,
        "calibration_quality_vs_ate": relationships,
        "session_005_diagnostics": session_005,
        "interpretation_note": (
            "Only four released test sessions are available; calibration-quality "
            "scatter plots are descriptive and must not define thresholds."
        ),
    }
    (output_root / f"method2ab_session_analysis_{split}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
