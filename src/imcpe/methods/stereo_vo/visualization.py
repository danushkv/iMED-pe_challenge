from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from imcpe.alignment import horn_align_sim3
from imcpe.io_pose import PoseRow

from .vo import DebugFrame


def _read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def _draw_matches(
    image0: np.ndarray,
    image1: np.ndarray,
    points0: np.ndarray,
    points1: np.ndarray,
    label: str,
    max_draw: int = 120,
) -> np.ndarray:
    height = max(image0.shape[0], image1.shape[0])
    width0 = image0.shape[1]
    canvas = np.zeros((height, width0 + image1.shape[1], 3), dtype=np.uint8)
    canvas[:image0.shape[0], :width0] = image0
    canvas[:image1.shape[0], width0:] = image1
    count = int(points0.shape[0])
    if count > max_draw:
        indices = np.linspace(0, count - 1, max_draw, dtype=int)
    else:
        indices = np.arange(count)
    for index in indices:
        point0 = tuple(np.rint(points0[index]).astype(int))
        point1_array = np.rint(points1[index]).astype(int)
        point1 = (int(point1_array[0] + width0), int(point1_array[1]))
        cv2.circle(canvas, point0, 2, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, point1, 2, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.line(canvas, point0, point1, (0, 180, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        f"{label}: {count} (showing {len(indices)})",
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def save_debug_match_image(debug: DebugFrame, output_path: Path) -> None:
    previous_left = _read_color(debug.previous_left_path)
    previous_right = _read_color(debug.previous_right_path)
    current_left = _read_color(debug.current_left_path)
    stereo = _draw_matches(
        previous_left,
        previous_right,
        debug.stereo_left_points,
        debug.stereo_right_points,
        "Stereo L(t-1) <-> R(t-1)",
    )
    temporal = _draw_matches(
        previous_left,
        current_left,
        debug.temporal_previous_points,
        debug.temporal_current_points,
        "Temporal L(t-1) <-> L(t)",
    )
    pnp = _draw_matches(
        previous_left,
        current_left,
        debug.pnp_previous_points,
        debug.pnp_current_points,
        "PnP inlier tracks only",
    )
    composite = np.vstack([stereo, temporal, pnp])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), composite):
        raise RuntimeError(f"Could not write debug image: {output_path}")


def _row_map(rows: list[PoseRow]) -> dict[int, PoseRow]:
    return {
        row.frame_idx: row
        for row in rows
        if np.isfinite(row.t).all() and np.isfinite(row.q_xyzw).all()
    }


def _aligned_prediction(
    gt_rows: list[PoseRow],
    prediction_rows: list[PoseRow],
) -> tuple[np.ndarray, np.ndarray]:
    gt = _row_map(gt_rows)
    prediction = _row_map(prediction_rows)
    common = sorted(set(gt) & set(prediction))
    if len(common) < 3:
        return np.empty((0, 3)), np.empty(0, dtype=int)
    gt_xyz = np.array([gt[frame_id].t for frame_id in common], dtype=np.float64)
    pred_xyz = np.array([prediction[frame_id].t for frame_id in common], dtype=np.float64)
    rotation, translation, _, scale = horn_align_sim3(pred_xyz.T, gt_xyz.T)
    aligned = (scale * rotation @ pred_xyz.T + translation).T
    return aligned, np.asarray(common, dtype=int)


def save_trajectory_plot(
    gt_rows: list[PoseRow],
    stereo_vo_rows: list[PoseRow],
    output_path: Path,
    title: str,
    baseline_rows: list[PoseRow] | None = None,
) -> None:
    gt = _row_map(gt_rows)
    gt_ids = sorted(gt)
    gt_xyz = np.array([gt[frame_id].t for frame_id in gt_ids], dtype=np.float64)
    stereo_xyz, _ = _aligned_prediction(gt_rows, stereo_vo_rows)
    baseline_xyz = None
    if baseline_rows is not None:
        baseline_xyz, _ = _aligned_prediction(gt_rows, baseline_rows)

    figure = plt.figure(figsize=(8, 6))
    axes = figure.add_subplot(111, projection="3d")
    axes.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label="Ground truth", linewidth=2)
    if stereo_xyz.size:
        axes.plot(
            stereo_xyz[:, 0],
            stereo_xyz[:, 1],
            stereo_xyz[:, 2],
            label="Stereo VO (Horn aligned)",
            linewidth=2,
        )
    if baseline_xyz is not None and baseline_xyz.size:
        axes.plot(
            baseline_xyz[:, 0],
            baseline_xyz[:, 1],
            baseline_xyz[:, 2],
            label="Baseline (Horn aligned)",
            linewidth=2,
        )
    axes.set_title(title)
    axes.set_xlabel("x (mm)")
    axes.set_ylabel("y (mm)")
    axes.set_zlabel("z (mm)")
    axes.legend()
    axes.grid(True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

