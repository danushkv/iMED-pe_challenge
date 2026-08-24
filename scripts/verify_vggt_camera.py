#!/usr/bin/env python3
"""User-run VGGT camera-only API and convention check (no ground truth access)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.methods.vggt_pose.data import load_vggt_inputs
from imcpe.methods.vggt_pose.model import VGGTCameraPredictor
from imcpe.methods.vggt_pose.pose_utils import (
    camera_center_world,
    homogeneous,
    invert_se3,
    relative_camera_transform,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the official VGGT camera branch on one synchronized four-view frame."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--preprocess-mode", choices=["crop", "pad"], default="crop")
    parser.add_argument("--model-id", default="facebook/VGGT-1B")
    parser.add_argument("--vggt-root", type=Path, default=ROOT / "vggt")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    inputs = load_vggt_inputs(args.data_root / args.split / args.sequence)
    i = args.frame_index
    if not 0 <= i < len(inputs.frame_ids):
        raise IndexError(f"frame-index must be in [0, {len(inputs.frame_ids) - 1}]")
    paths = [
        inputs.e1_l_images[i], inputs.e1_r_images[i],
        inputs.e2_l_images[i], inputs.e2_r_images[i],
    ]
    predictor = VGGTCameraPredictor(
        device=args.device,
        dtype=args.dtype,
        model_id=args.model_id,
        vggt_root=args.vggt_root,
    )
    prediction = predictor.predict(paths, args.preprocess_mode)
    transforms = np.stack([homogeneous(value) for value in prediction.extrinsics_cam_from_world])
    relative = relative_camera_transform(transforms[2], transforms[0])
    inverse_error = float(
        np.max(np.abs(relative @ invert_se3(relative) - np.eye(4)))
    )
    e1_baseline = float(
        np.linalg.norm(camera_center_world(transforms[0]) - camera_center_world(transforms[1]))
    )
    payload = {
        "model_id": args.model_id,
        "model_load_seconds": predictor.model_load_seconds,
        "frame_id": inputs.frame_ids[i],
        "input_paths": [str(path) for path in paths],
        "extrinsic_shape": list(prediction.extrinsics_cam_from_world.shape),
        "intrinsic_shape": list(prediction.intrinsics.shape),
        "input_height_width": list(prediction.input_shape),
        "inference_seconds": prediction.inference_seconds,
        "peak_gpu_memory_bytes": prediction.peak_gpu_memory_bytes,
        "e1_stereo_baseline_raw": e1_baseline,
        "T_E2L_from_E1L": relative.tolist(),
        "relative_inverse_identity_max_error": inverse_error,
    }
    print(json.dumps(payload, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
