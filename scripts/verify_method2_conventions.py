#!/usr/bin/env python3
"""Numerically verify Method 2 transform direction without using ground truth."""

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
        raw = np.asarray(archive["T_E2_from_E1"], dtype=np.float64)
        saved_relative = np.asarray(archive["T_E20_from_E2t"], dtype=np.float64)
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
    pose_matrices_array = np.stack(pose_matrices)
    frame_zero_error = float(np.max(np.abs(saved_relative[0] - np.eye(4))))
    pose_cache_error = float(np.max(np.abs(pose_matrices_array - saved_relative)))
    direction_errors = []
    for index in np.flatnonzero(success):
        expected = raw[0] @ np.linalg.inv(raw[index])
        direction_errors.append(float(np.max(np.abs(expected - saved_relative[index]))))
    maximum_direction_error = max(direction_errors, default=float("nan"))
    rotations = saved_relative[:, :3, :3]
    determinant_error = float(np.max(np.abs(np.linalg.det(rotations) - 1.0)))
    orthogonality_error = float(
        np.max(
            np.abs(
                np.einsum("nji,njk->nik", rotations, rotations) - np.eye(3)
            )
        )
    )
    print(f"frame_zero_identity_error={frame_zero_error:.3e}")
    print(f"pose_txt_vs_cache_error={pose_cache_error:.3e}")
    print(f"T0 @ inverse(Tt) direction_error={maximum_direction_error:.3e}")
    print(f"rotation_determinant_error={determinant_error:.3e}")
    print(f"rotation_orthogonality_error={orthogonality_error:.3e}")
    if (
        frame_zero_error > 1e-8
        or pose_cache_error > 1e-8
        or maximum_direction_error > 1e-8
        or determinant_error > 1e-8
        or orthogonality_error > 1e-8
    ):
        raise SystemExit("Method 2 convention verification failed")
    print("Method 2 transform conventions verified.")


if __name__ == "__main__":
    main()

