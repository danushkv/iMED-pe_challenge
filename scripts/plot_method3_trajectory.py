#!/usr/bin/env python3
"""Evaluation-only Horn-aligned trajectory plot, including sparse predictions."""

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


def _aligned(gt_rows, pred_rows) -> tuple[np.ndarray, np.ndarray]:
    gt = {row.frame_idx: row.t for row in gt_rows if np.isfinite(row.t).all()}
    pred = {row.frame_idx: row.t for row in pred_rows if np.isfinite(row.t).all()}
    ids = np.asarray(sorted(set(gt) & set(pred)), dtype=int)
    if len(ids) < 3:
        raise RuntimeError("fewer than three common finite poses")
    gt_xyz = np.stack([gt[int(frame_id)] for frame_id in ids])
    pred_xyz = np.stack([pred[int(frame_id)] for frame_id in ids])
    rotation, translation, _, scale = horn_align_sim3(pred_xyz.T, gt_xyz.T)
    aligned = (scale * rotation @ pred_xyz.T + translation).T
    return ids, aligned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-pose", type=Path, required=True)
    parser.add_argument(
        "--prediction", action="append", required=True,
        help="LABEL=path/to/pose.txt; repeat for multiple methods",
    )
    parser.add_argument("--title", default="Method 3 trajectory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gt_rows = read_pose_txt(args.gt_pose)
    gt = {row.frame_idx: row.t for row in gt_rows}
    fig = plt.figure(figsize=(9, 7))
    axis = fig.add_subplot(111, projection="3d")
    for specification in args.prediction:
        if "=" not in specification:
            raise ValueError("--prediction must be LABEL=PATH")
        label, path = specification.split("=", 1)
        ids, aligned = _aligned(gt_rows, read_pose_txt(Path(path)))
        gt_matched = np.stack([gt[int(frame_id)] for frame_id in ids])
        axis.plot(
            gt_matched[:, 0], gt_matched[:, 1], gt_matched[:, 2],
            color="black", alpha=0.25, linewidth=2,
        )
        axis.plot(aligned[:, 0], aligned[:, 1], aligned[:, 2], label=label)
    all_gt = np.stack([row.t for row in gt_rows if np.isfinite(row.t).all()])
    axis.plot(
        all_gt[:, 0], all_gt[:, 1], all_gt[:, 2],
        color="black", linewidth=2.5, label="Ground truth",
    )
    axis.set_title(args.title)
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_zlabel("z (mm)")
    axis.legend()
    axis.grid(True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
