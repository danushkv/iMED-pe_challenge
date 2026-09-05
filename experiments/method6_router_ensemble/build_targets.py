"""TRAIN-only trajectory-level GT targets for reliability supervision."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from imcpe.alignment import horn_align_sim3
from imcpe.io_pose import read_pose_txt

from .config import EXPERT_NAMES
from .load_experts import ObserverBundle


def _gt_centers(gt_path: Path, frame_ids: np.ndarray) -> np.ndarray:
    rows = {int(row.frame_idx): row for row in read_pose_txt(gt_path)}
    missing = [int(frame_id) for frame_id in frame_ids if int(frame_id) not in rows]
    if missing:
        raise RuntimeError(f"GT is missing frame IDs {missing} in {gt_path}")
    return np.stack([rows[int(frame_id)].t for frame_id in frame_ids]).astype(
        np.float64
    )


def build_train_targets(
    *,
    gt_path: Path,
    bundle: ObserverBundle,
    expert_centers_common: np.ndarray,
) -> dict[str, np.ndarray]:
    """Align once per complete trajectory, never independently per frame."""
    gt = _gt_centers(gt_path, bundle.frame_ids)
    errors = np.empty((len(bundle.frame_ids), len(EXPERT_NAMES)), dtype=np.float64)
    for expert_index, name in enumerate(EXPERT_NAMES):
        predicted = bundle.observers[name].centers
        valid = bundle.observers[name].valid & np.isfinite(predicted).all(axis=1)
        if np.count_nonzero(valid) < 3:
            errors[:, expert_index] = np.inf
            continue
        rotation, translation, _, scale = horn_align_sim3(
            predicted[valid].T, gt[valid].T
        )
        aligned = scale * (rotation @ predicted[valid].T) + translation
        errors[:, expert_index] = np.inf
        errors[valid, expert_index] = np.linalg.norm(aligned.T - gt[valid], axis=1)

    # A shared GT coordinate representation for the convex oracle. All common
    # observer centers receive the single Method-2B-to-GT trajectory Sim(3).
    reference = expert_centers_common[EXPERT_NAMES.index("2b")]
    rotation, translation, _, scale = horn_align_sim3(reference.T, gt.T)
    common_in_gt = (
        scale * np.einsum("ij,mnj->mni", rotation, expert_centers_common)
        + translation.reshape(1, 1, 3)
    )
    return {
        "gt_centers": gt,
        "expert_errors": errors,
        "best_expert": np.argmin(errors, axis=1).astype(np.int64),
        "expert_common_centers_in_gt": common_in_gt,
    }
