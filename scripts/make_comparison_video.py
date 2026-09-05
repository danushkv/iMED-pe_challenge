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
from PIL import Image

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


def trajectory_xyz(values: dict[int, np.ndarray], frame_ids: list[int]) -> np.ndarray:
    output = np.full((len(frame_ids), 3), np.nan, dtype=np.float64)
    for position, frame_id in enumerate(frame_ids):
        if frame_id in values:
            output[position] = values[frame_id]
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
    parser.add_argument(
        "--gif-output",
        type=Path,
        default=None,
        help="Optional compact animated GIF suitable for a GitHub README.",
    )
    parser.add_argument(
        "--gif-every",
        type=int,
        default=1,
        help="Keep every Nth rendered frame in the GIF (default: 1, no skipping).",
    )
    parser.add_argument(
        "--gif-width",
        type=int,
        default=720,
        help="Resize the GIF to this width while preserving aspect ratio.",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args()
    if args.gif_every < 1:
        raise ValueError("--gif-every must be at least 1")
    if args.gif_width < 1:
        raise ValueError("--gif-width must be positive")

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
    gt_xyz = trajectory_xyz(gt, frame_ids)
    pred_xyz = {
        item.label: trajectory_xyz(item.by_frame, frame_ids)
        for item in predictions
    }
    all_xyz = [gt_xyz, *pred_xyz.values()]
    finite = np.concatenate([array[np.isfinite(array).all(axis=1)] for array in all_xyz])
    lower = finite.min(axis=0)
    upper = finite.max(axis=0)
    plot_center = 0.5 * (lower + upper)
    plot_radius = max(0.525 * float(np.max(upper - lower)), 1.0e-6)

    num_columns = 2 + len(predictions)
    figure = plt.figure(figsize=(4.0 * num_columns, 6.4), dpi=args.dpi)
    grid = figure.add_gridspec(2, num_columns, width_ratios=[1.35] + [1.0] * (num_columns - 1))
    image_axes = [
        figure.add_subplot(grid[row, 0])
        for row in range(2)
    ]
    # Each image axis contains a horizontal stereo pair.
    trajectory_axes = [
        figure.add_subplot(grid[:, column], projection="3d")
        for column in range(1, num_columns)
    ]

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
    gif_frames: list[Image.Image] = []

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
                axis.plot(
                    gt_xyz[:upto, 0],
                    gt_xyz[:upto, 1],
                    gt_xyz[:upto, 2],
                    color="black",
                    lw=2,
                    label="GT",
                )
                if column > 0:
                    values = pred_xyz[label]
                    axis.plot(
                        values[:upto, 0],
                        values[:upto, 1],
                        values[:upto, 2],
                        lw=2,
                        label=label,
                    )
                axis.scatter(
                    gt_xyz[frame_number, 0],
                    gt_xyz[frame_number, 1],
                    gt_xyz[frame_number, 2],
                    color="black",
                    s=24,
                )
                axis.set_xlim(plot_center[0] - plot_radius, plot_center[0] + plot_radius)
                axis.set_ylim(plot_center[1] - plot_radius, plot_center[1] + plot_radius)
                axis.set_zlim(plot_center[2] - plot_radius, plot_center[2] + plot_radius)
                axis.set_box_aspect((1, 1, 1))
                axis.view_init(elev=22, azim=-60)
                axis.set_title(label)
                axis.set_xlabel("x (mm)")
                axis.set_ylabel("y (mm)")
                axis.set_zlabel("z (mm)")
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
            if args.gif_output is not None and frame_number % args.gif_every == 0:
                gif_height = max(1, round(video_height * args.gif_width / video_width))
                gif_bgr = cv2.resize(
                    bgr,
                    (args.gif_width, gif_height),
                    interpolation=cv2.INTER_AREA,
                )
                gif_rgb = cv2.cvtColor(gif_bgr, cv2.COLOR_BGR2RGB)
                gif_frames.append(Image.fromarray(gif_rgb))
        if args.still_output is not None:
            args.still_output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.still_output, bbox_inches="tight")
    finally:
        writer.release()
        plt.close(figure)
    print(f"Saved: {args.output}")
    if args.gif_output is not None:
        if not gif_frames:
            raise RuntimeError("No frames were available for GIF output")
        args.gif_output.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = max(20, round(1000.0 * args.gif_every / args.fps))
        gif_frames[0].save(
            args.gif_output,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
            disposal=2,
        )
        print(f"Saved: {args.gif_output}")


if __name__ == "__main__":
    main()
