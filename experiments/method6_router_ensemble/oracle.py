#!/usr/bin/env python3
"""TRAIN-only hard and convex oracle upper bounds; never an inference method."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from imcpe.io_pose import PoseRow, write_pose_txt

from .config import EXPERT_NAMES
from .dataset import load_dataset


def closest_triangle_weights(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    candidates: list[np.ndarray] = [np.eye(3)[index] for index in range(3)]
    for first, second in ((0, 1), (0, 2), (1, 2)):
        direction = points[second] - points[first]
        denominator = float(direction @ direction)
        fraction = (
            float((target - points[first]) @ direction) / denominator
            if denominator > 1.0e-12
            else 0.0
        )
        fraction = float(np.clip(fraction, 0.0, 1.0))
        weights = np.zeros(3, dtype=np.float64)
        weights[first] = 1.0 - fraction
        weights[second] = fraction
        candidates.append(weights)
    gram = points @ points.T
    rhs = points @ target
    system = np.block(
        [[gram, np.ones((3, 1))], [np.ones((1, 3)), np.zeros((1, 1))]]
    )
    try:
        solution = np.linalg.solve(system, np.concatenate((rhs, [1.0])))[:3]
        if np.min(solution) >= -1.0e-9:
            candidates.append(np.maximum(solution, 0.0) / np.sum(np.maximum(solution, 0.0)))
    except np.linalg.LinAlgError:
        pass
    error = [np.linalg.norm(weights @ points - target) for weights in candidates]
    return candidates[int(np.argmin(error))]


def write_oracle(root: Path, split: str, record, centers: np.ndarray) -> None:
    rows = [
        PoseRow(
            frame_idx=int(frame_id),
            t=centers[index],
            q_xyzw=Rotation.from_matrix(record.rotation_source[index]).as_quat(),
        )
        for index, frame_id in enumerate(record.frame_ids)
    ]
    write_pose_txt(root / split / record.sequence / "pose.txt", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_dataset(args.dataset_root)
    if dataset.split != "train" or any(record.best_expert is None for record in dataset.records):
        raise ValueError("Oracle requires a TRAIN dataset with supervised targets")
    report = {"split": "train", "diagnostic_only": True, "sequences": {}}
    for record in dataset.records:
        assert record.best_expert is not None
        assert record.gt_centers is not None
        assert record.expert_common_centers_in_gt is not None
        indices = np.arange(len(record.frame_ids))
        hard = record.expert_centers[record.best_expert, indices]
        weights = np.stack(
            [
                closest_triangle_weights(
                    record.expert_common_centers_in_gt[:, index],
                    record.gt_centers[index],
                )
                for index in indices
            ]
        )
        soft = np.sum(weights.T[..., None] * record.expert_centers, axis=0)
        hard -= hard[0]
        soft -= soft[0]
        write_oracle(args.output_root / "oracle_hard", "train", record, hard)
        write_oracle(args.output_root / "oracle_soft", "train", record, soft)
        report["sequences"][record.sequence] = {
            "session": record.session,
            "num_frames": len(indices),
            "hard_usage": {
                name: float(np.mean(record.best_expert == expert_index))
                for expert_index, name in enumerate(EXPERT_NAMES)
            },
            "soft_mean_weight": {
                name: float(np.mean(weights[:, expert_index]))
                for expert_index, name in enumerate(EXPERT_NAMES)
            },
        }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "oracle_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved Method-6 oracle bounds: {args.output_root}")


if __name__ == "__main__":
    main()

