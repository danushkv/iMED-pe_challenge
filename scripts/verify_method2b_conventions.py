#!/usr/bin/env python3
"""Numerically verify Method 2B transforms without reading ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.io_pose import read_pose_txt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.raw_cache, allow_pickle=False) as archive:
        frame_ids = np.asarray(archive["frame_ids"], dtype=np.int64)
        reverse = np.asarray(archive["T_E1_from_E2"], dtype=np.float64)
        common = np.asarray(archive["T_E2_from_E1"], dtype=np.float64)
        relative = np.asarray(archive["T_E20_from_E2t"], dtype=np.float64)
        success = np.asarray(archive["pnp_success"], dtype=bool)
    rows = read_pose_txt(args.pose)
    if [row.frame_idx for row in rows] != frame_ids.tolist():
        raise RuntimeError("pose.txt frame IDs do not match the raw cache")
    pose_matrices = []
    for row in rows:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = Rotation.from_quat(row.q_xyzw).as_matrix()
        transform[:3, 3] = row.t
        pose_matrices.append(transform)
    pose_array = np.stack(pose_matrices)
    inverse_errors = []
    common_errors = []
    direction_errors = []
    for index in np.flatnonzero(success):
        inverse_errors.append(
            float(np.max(np.abs(reverse[index] @ common[index] - np.eye(4))))
        )
        common_errors.append(
            float(np.max(np.abs(np.linalg.inv(reverse[index]) - common[index])))
        )
        expected = common[0] @ np.linalg.inv(common[index])
        direction_errors.append(float(np.max(np.abs(expected - relative[index]))))
    values = {
        "inverse_consistency_error": max(inverse_errors, default=float("nan")),
        "common_transform_error": max(common_errors, default=float("nan")),
        "frame_zero_identity_error": float(np.max(np.abs(relative[0] - np.eye(4)))),
        "relative_direction_error": max(direction_errors, default=float("nan")),
        "pose_txt_vs_cache_error": float(np.max(np.abs(pose_array - relative))),
        "rotation_determinant_error": float(
            np.max(np.abs(np.linalg.det(relative[:, :3, :3]) - 1.0))
        ),
        "rotation_orthogonality_error": float(
            np.max(
                np.abs(
                    np.einsum(
                        "nji,njk->nik",
                        relative[:, :3, :3],
                        relative[:, :3, :3],
                    )
                    - np.eye(3)
                )
            )
        ),
    }
    for key, value in values.items():
        print(f"{key}={value:.3e}")
    if any(not np.isfinite(value) or value > 1e-8 for value in values.values()):
        raise SystemExit("Method 2B convention verification failed")
    print("Method 2B transform conventions verified.")


if __name__ == "__main__":
    main()
