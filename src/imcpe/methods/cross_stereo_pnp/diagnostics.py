"""Post-inference Method 2 plots. Ground truth is read only in this module."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imcpe.alignment import horn_align_sim3
from imcpe.io_pose import PoseRow, read_pose_txt


def _valid_map(rows: list[PoseRow]) -> dict[int, PoseRow]:
    return {
        row.frame_idx: row
        for row in rows
        if np.isfinite(row.t).all() and np.isfinite(row.q_xyzw).all()
    }


def _aligned(
    ground_truth: list[PoseRow], prediction: list[PoseRow]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = _valid_map(ground_truth)
    pred = _valid_map(prediction)
    frame_ids = np.asarray(sorted(set(gt) & set(pred)), dtype=int)
    if frame_ids.size < 3:
        raise RuntimeError("At least three common poses are required for plotting")
    gt_xyz = np.asarray([gt[int(frame_id)].t for frame_id in frame_ids])
    pred_xyz = np.asarray([pred[int(frame_id)].t for frame_id in frame_ids])
    rotation, translation, _, scale = horn_align_sim3(pred_xyz.T, gt_xyz.T)
    aligned = (scale * rotation @ pred_xyz.T + translation).T
    return frame_ids, gt_xyz, aligned


def create_method2_sequence_plots(
    *,
    sequence_name: str,
    ground_truth_pose: Path,
    baseline_pose: Path,
    method1_pose: Path,
    method1_5_pose: Path,
    method2_pose: Path,
    method2_diagnostics: Path,
    output_dir: Path,
) -> dict[str, float | int | str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = read_pose_txt(ground_truth_pose)
    predictions = {
        "Baseline": read_pose_txt(baseline_pose),
        "Method 1": read_pose_txt(method1_pose),
        "Method 1.5A": read_pose_txt(method1_5_pose),
        "Method 2": read_pose_txt(method2_pose),
    }
    aligned: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        name: _aligned(ground_truth, rows) for name, rows in predictions.items()
    }
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    gt_map = _valid_map(ground_truth)
    gt_ids = sorted(gt_map)
    gt_xyz = np.asarray([gt_map[frame_id].t for frame_id in gt_ids])
    axis.plot(*gt_xyz.T, label="Ground truth", color="black", linewidth=2)
    colors = {
        "Baseline": "tab:orange",
        "Method 1": "tab:blue",
        "Method 1.5A": "tab:green",
        "Method 2": "tab:red",
    }
    for name, (_, _, xyz) in aligned.items():
        axis.plot(*xyz.T, label=name, color=colors[name])
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_zlabel("z (mm)")
    axis.set_title(sequence_name)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "trajectory.png", dpi=180)
    plt.close(figure)

    frame_ids, gt_common, method2_aligned = aligned["Method 2"]
    aligned_error = np.linalg.norm(method2_aligned - gt_common, axis=1)
    payload = json.loads(method2_diagnostics.read_text())
    frames = payload.get("frames", [])
    diagnostic_frame_ids = np.asarray(
        [int(frame["frame_idx"]) for frame in frames], dtype=int
    )

    def series(key: str) -> np.ndarray:
        return np.asarray([float(frame.get(key, np.nan)) for frame in frames])

    figure, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=False)
    axes[0].plot(diagnostic_frame_ids, series("pnp_inlier_ratio"))
    axes[0].set_ylabel("PnP inlier ratio")
    axes[1].plot(
        diagnostic_frame_ids,
        series("pnp_reprojection_error_median_px"),
    )
    axes[1].set_ylabel("Reprojection (px)")
    axes[2].plot(
        diagnostic_frame_ids, series("raw_translation_magnitude_baselines")
    )
    axes[2].set_ylabel("Raw |t| (baselines)")
    axes[3].plot(diagnostic_frame_ids, series("relative_rotation_change_deg"))
    axes[3].set_ylabel("Rotation change (deg)")
    axes[4].plot(frame_ids, aligned_error)
    axes[4].set_ylabel("Aligned point error (mm)")
    axes[4].set_xlabel("Frame ID")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(f"{sequence_name}: Method 2 frame diagnostics")
    figure.tight_layout()
    figure.savefig(output_dir / "frame_diagnostics.png", dpi=180)
    plt.close(figure)
    summary = {
        "sequence": sequence_name,
        "mean_aligned_point_error_mm": float(np.mean(aligned_error)),
        "median_aligned_point_error_mm": float(np.median(aligned_error)),
        "num_aligned_frames": int(frame_ids.size),
    }
    (output_dir / "plot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary

