#!/usr/bin/env python3
"""Evaluation-only Method 4 trajectory and aligned point-error plots.

This script uses the same Horn/Sim(3) alignment implementation as the official
repository evaluator. Ground truth is read only here, after predictions exist.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.alignment import horn_align_sim3
from imcpe.io_pose import read_pose_txt


def _parse_prediction(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError("--prediction must be LABEL=ROOT")
    label, root = specification.split("=", 1)
    if not label or not root:
        raise ValueError("--prediction must be LABEL=ROOT")
    return label, Path(root)


def _aligned(gt_rows, pred_rows) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt = {row.frame_idx: row.t for row in gt_rows if np.isfinite(row.t).all()}
    pred = {row.frame_idx: row.t for row in pred_rows if np.isfinite(row.t).all()}
    frame_ids = np.asarray(sorted(set(gt) & set(pred)), dtype=int)
    if frame_ids.size < 3:
        raise RuntimeError("fewer than three common finite poses")
    gt_xyz = np.stack([gt[int(frame_id)] for frame_id in frame_ids])
    pred_xyz = np.stack([pred[int(frame_id)] for frame_id in frame_ids])
    rotation, translation, error, scale = horn_align_sim3(pred_xyz.T, gt_xyz.T)
    aligned = (scale * rotation @ pred_xyz.T + translation).T
    return frame_ids, gt_xyz, aligned, error


def _plot_sequence(
    *,
    data_root: Path,
    split: str,
    sequence: str,
    predictions: list[tuple[str, Path]],
    output_root: Path,
) -> None:
    gt_path = data_root / split / sequence / "pose.txt"
    gt_rows = read_pose_txt(gt_path)
    all_gt = np.stack([row.t for row in gt_rows if np.isfinite(row.t).all()])

    figure = plt.figure(figsize=(14, 6))
    trajectory_axis = figure.add_subplot(121, projection="3d")
    error_axis = figure.add_subplot(122)
    trajectory_axis.plot(
        all_gt[:, 0], all_gt[:, 1], all_gt[:, 2],
        color="black", linewidth=2.5, label="Ground truth",
    )

    for label, root in predictions:
        pose_path = root / split / sequence / "pose.txt"
        if not pose_path.is_file():
            print(f"Warning: missing {label}: {pose_path}")
            continue
        frame_ids, _, aligned, error = _aligned(gt_rows, read_pose_txt(pose_path))
        trajectory_axis.plot(*aligned.T, linewidth=1.6, label=label)
        error_axis.plot(frame_ids, error, linewidth=1.3, label=label)

    trajectory_axis.set_title(f"{sequence}: official-aligned trajectory")
    trajectory_axis.set_xlabel("x (mm)")
    trajectory_axis.set_ylabel("y (mm)")
    trajectory_axis.set_zlabel("z (mm)")
    trajectory_axis.grid(True, alpha=0.3)
    trajectory_axis.legend(fontsize=8)
    error_axis.set_title("Aligned point error versus frame")
    error_axis.set_xlabel("frame ID")
    error_axis.set_ylabel("Euclidean error (mm)")
    error_axis.grid(True, alpha=0.3)
    error_axis.legend(fontsize=8)
    figure.tight_layout()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{sequence}.png"
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot GT-aligned Method 4 trajectories using official alignment."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--sequence", action="append")
    parser.add_argument(
        "--prediction", action="append", required=True,
        help="LABEL=prediction/root; repeat for multiple methods",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    predictions = [_parse_prediction(item) for item in args.prediction]
    sequences = args.sequence
    if sequences is None:
        sequences = sorted(path.name for path in (args.data_root / args.split).iterdir() if path.is_dir())
    for sequence in sequences:
        _plot_sequence(
            data_root=args.data_root,
            split=args.split,
            sequence=sequence,
            predictions=predictions,
            output_root=args.output_root,
        )
    print(f"Saved {len(sequences)} trajectory diagnostics under: {args.output_root}")


if __name__ == "__main__":
    main()
