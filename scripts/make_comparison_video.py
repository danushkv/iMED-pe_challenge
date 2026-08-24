#!/usr/bin/env python3
"""Render synchronized endoscope views and Sim(3)-aligned trajectories.

This is an evaluation/communication utility: it reads ``pose.txt`` ground
truth only after predictions already exist. Generated media is gitignored.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.alignment import horn_align_sim3  # noqa: E402
from imcpe.io_pose import PoseRow, read_pose_txt  # noqa: E402


@dataclass(frozen=True)
class Trajectory:
    label: str
    by_frame: dict[int, np.ndarray]


def parse_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=/path/to/pose.txt")
    label, raw_path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("prediction label cannot be empty")
    return label, Path(raw_path)


def valid_rows(rows: list[PoseRow]) -> dict[int, np.ndarray]:
    return {
        row.frame_idx: np.asarray(row.t, dtype=np.float64)
        for row in rows
        if np.isfinite(row.t).all()
    }


def align_prediction(gt: dict[int, np.ndarray], pred_rows: list[PoseRow]) -> dict[int, np.ndarray]:
    pred = valid_rows(pred_rows)
    common = sorted(set(gt) & set(pred))
    if len(common) < 3:
        raise ValueError(f"Only {len(common)} finite GT/prediction correspondences")
    model = np.stack([pred[index] for index in common], axis=1)
    data = np.stack([gt[index] for index in common], axis=1)
    rotation, translation, _, scale = horn_align_sim3(model, data)
    return {
        index: scale * rotation @ point + translation.reshape(3)
        for index, point in pred.items()
    }


def image_map(directory: Path) -> dict[int, Path]:
    return {
        int(path.stem.split("_")[1]): path
        for path in sorted(directory.glob("frame_*.png"))
    }


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def projection_basis(gt: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    points = np.stack(list(gt.values()))
    center = points.mean(axis=0)
    _, _, right = np.linalg.svd(points - center, full_matrices=False)
    return center, right[:2].T


def project(point: np.ndarray, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (np.asarray(point) - center) @ basis


def trajectory_xy(
    values: dict[int, np.ndarray],
    frame_ids: list[int],
    center: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    output = np.full((len(frame_ids), 2), np.nan, dtype=np.float64)
    for position, frame_id in enumerate(frame_ids):
        if frame_id in values:
            output[position] = project(values[frame_id], center, basis)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        type=parse_prediction,
        default=[],
        metavar="LABEL=POSE_TXT",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--still-output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args()

    gt_path = args.sequence_dir / "pose.txt"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Visualization requires evaluation GT: {gt_path}")
    gt = valid_rows(read_pose_txt(gt_path))

    camera_dirs = {
        "E1-L": args.sequence_dir / "endoscope1" / "L",
        "E1-R": args.sequence_dir / "endoscope1" / "R",
        "E2-L": args.sequence_dir / "endoscope2" / "L",
        "E2-R": args.sequence_dir / "endoscope2" / "R",
    }
    images = {label: image_map(path) for label, path in camera_dirs.items()}
    frame_ids = sorted(set.intersection(*(set(mapping) for mapping in images.values())))
    if args.max_frames is not None:
        frame_ids = frame_ids[: args.max_frames]
    if not frame_ids:
        raise RuntimeError("No synchronized frames")

    predictions = [
        Trajectory(label, align_prediction(gt, read_pose_txt(path)))
        for label, path in args.prediction
    ]
    center, basis = projection_basis(gt)
    gt_xy = trajectory_xy(gt, frame_ids, center, basis)
    pred_xy = {
        item.label: trajectory_xy(item.by_frame, frame_ids, center, basis)
        for item in predictions
    }
    all_xy = [gt_xy, *pred_xy.values()]
    finite = np.concatenate([array[np.isfinite(array).all(axis=1)] for array in all_xy])
    lower = finite.min(axis=0)
    upper = finite.max(axis=0)
    padding = np.maximum(0.05 * (upper - lower), 1e-6)

    num_columns = 2 + len(predictions)
    figure = plt.figure(figsize=(4.0 * num_columns, 6.4), dpi=args.dpi)
    grid = figure.add_gridspec(2, num_columns, width_ratios=[1.35] + [1.0] * (num_columns - 1))
    image_axes = [
        figure.add_subplot(grid[row, 0])
        for row in range(2)
    ]
    # Each image axis contains a horizontal stereo pair.
    trajectory_axes = [figure.add_subplot(grid[:, column]) for column in range(1, num_columns)]

    first = load_rgb(images["E1-L"][frame_ids[0]])
    display_height, display_width = first.shape[:2]
    video_width = int(figure.get_figwidth() * args.dpi)
    video_height = int(figure.get_figheight() * args.dpi)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (video_width, video_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {args.output}")

    labels = ["Ground truth", *[item.label for item in predictions]]
    try:
        for frame_number, frame_id in enumerate(frame_ids):
            for axis, left_label, right_label in (
                (image_axes[0], "E1-L", "E1-R"),
                (image_axes[1], "E2-L", "E2-R"),
            ):
                axis.clear()
                left = load_rgb(images[left_label][frame_id])
                right = load_rgb(images[right_label][frame_id])
                if left.shape[:2] != (display_height, display_width):
                    left = cv2.resize(left, (display_width, display_height))
                if right.shape[:2] != (display_height, display_width):
                    right = cv2.resize(right, (display_width, display_height))
                axis.imshow(np.concatenate([left, right], axis=1))
                axis.set_title(f"{left_label} | {right_label}")
                axis.axis("off")

            for column, (axis, label) in enumerate(zip(trajectory_axes, labels)):
                axis.clear()
                upto = frame_number + 1
                axis.plot(gt_xy[:upto, 0], gt_xy[:upto, 1], color="black", lw=2, label="GT")
                if column > 0:
                    values = pred_xy[label]
                    axis.plot(values[:upto, 0], values[:upto, 1], lw=2, label=label)
                axis.scatter(gt_xy[frame_number, 0], gt_xy[frame_number, 1], color="black", s=24)
                axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
                axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
                axis.set_aspect("equal", adjustable="box")
                axis.set_title(label)
                axis.set_xlabel("principal trajectory axis 1 (mm)")
                axis.set_ylabel("principal trajectory axis 2 (mm)")
                axis.grid(alpha=0.25)
                axis.legend(loc="best", fontsize=8)

            figure.suptitle(f"{args.sequence_dir.name} — frame {frame_id}")
            figure.tight_layout()
            figure.canvas.draw()
            rgb = np.asarray(figure.canvas.buffer_rgba())[..., :3]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if bgr.shape[1] != video_width or bgr.shape[0] != video_height:
                bgr = cv2.resize(bgr, (video_width, video_height))
            writer.write(bgr)
        if args.still_output is not None:
            args.still_output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.still_output, bbox_inches="tight")
    finally:
        writer.release()
        plt.close(figure)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

