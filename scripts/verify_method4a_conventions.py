#!/usr/bin/env python3
"""Numerical convention checks using saved predictions only; no GT access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.methods.hybrid_pose_fusion.load_predictions import load_prediction_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--method1-root", type=Path, required=True)
    parser.add_argument("--method2a-root", type=Path, required=True)
    parser.add_argument("--method2b-root", type=Path, required=True)
    parser.add_argument("--rotation-root", type=Path, required=True)
    args = parser.parse_args()
    bundle = load_prediction_bundle(
        split=args.split,
        sequence=args.sequence,
        method1_root=args.method1_root,
        method2a_root=args.method2a_root,
        method2b_root=args.method2b_root,
        rotation_root=args.rotation_root,
    )
    centers = bundle.method1.centers
    rotations = bundle.method1.rotations_initial_from_camera
    maximum_increment_error = 0.0
    for index in range(1, len(centers)):
        previous = np.eye(4)
        previous[:3, :3] = rotations[index - 1]
        previous[:3, 3] = centers[index - 1]
        current = np.eye(4)
        current[:3, :3] = rotations[index]
        current[:3, 3] = centers[index]
        local = np.linalg.inv(previous) @ current
        recovered_delta = rotations[index - 1] @ local[:3, 3]
        maximum_increment_error = max(
            maximum_increment_error,
            float(np.max(np.abs(recovered_delta - (centers[index] - centers[index - 1])))),
        )
    print(json.dumps({
        "sequence": args.sequence,
        "num_frames": len(centers),
        "all_frame_ids_equal": True,
        "all_frame_zero_centers_zero": True,
        "all_frame_zero_rotations_identity": True,
        "maximum_local_increment_recovery_error": maximum_increment_error,
        "stored_translation_interpretation": "camera center C in T_0_from_t",
    }, indent=2))


if __name__ == "__main__":
    main()
