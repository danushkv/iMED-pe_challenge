from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.io_pose import read_pose_txt
from imcpe.methods.stereo_vo_anchored.anchor_estimator import load_anchor_cache
from imcpe.methods.stereo_vo_anchored.anchored_vo import fuse_anchored_trajectory
from imcpe.methods.stereo_vo_anchored.config import AnchorThresholds, FusionConfig


def pose_matrix(row) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(row.q_xyzw).as_matrix()
    matrix[:3, 3] = row.t
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Numerically verify Method 1.5 conventions.")
    parser.add_argument("--method1-pose", type=Path, required=True)
    parser.add_argument("--anchor-cache", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    args = parser.parse_args()

    rows = read_pose_txt(args.method1_pose)
    anchors = load_anchor_cache(args.anchor_cache)
    thresholds = AnchorThresholds.load_json(args.thresholds)
    matrices = np.asarray([pose_matrix(row) for row in rows])
    transition_index = 1
    local_previous_from_current = (
        np.linalg.inv(matrices[transition_index - 1]) @ matrices[transition_index]
    )
    recomposed = matrices[transition_index - 1] @ local_previous_from_current
    transition_error = float(np.max(np.abs(recomposed - matrices[transition_index])))

    result = fuse_anchored_trajectory(
        rows,
        anchors,
        thresholds,
        FusionConfig(anchor_interval=1, alpha=0.0),
    )
    print(f"Sequence: {anchors.sequence_name}")
    print(f"Checked transition: {rows[0].frame_idx} -> {rows[1].frame_idx}")
    print(f"Recomposition max error: {transition_error:.3e}")
    print(
        "R_anchor_rel(0) identity max error: "
        f"{np.max(np.abs(anchors.R_anchor_rel[0] - np.eye(3))):.3e}"
    )
    print(
        "alpha=0 Method 1 reproduction max error: "
        f"{result.summary['alpha_zero_max_matrix_error']:.3e}"
    )
    print(
        "maximum corrected SO(3) determinant/orthogonality error: "
        f"{result.summary['maximum_rotation_determinant_error']:.3e} / "
        f"{result.summary['maximum_rotation_orthogonality_error']:.3e}"
    )


if __name__ == "__main__":
    main()

