#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.io_pose import write_pose_txt
from imcpe.methods.vggt_pose.config import Method3Config
from imcpe.methods.vggt_pose.data import load_vggt_inputs
from imcpe.methods.vggt_pose.inference import run_method3_sequence, select_frame_indices
from imcpe.methods.vggt_pose.model import VGGTCameraPredictor


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Method 3 zero-shot VGGT camera-only inference."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--variant", choices=["3a", "3b"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence", action="append", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--sample-count", type=int, default=None,
        help="Uniformly sample this many indices across each full sequence.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto"
    )
    parser.add_argument("--preprocess-mode", choices=["crop", "pad"], default="crop")
    parser.add_argument("--model-id", default="facebook/VGGT-1B")
    parser.add_argument("--vggt-root", type=Path, default=ROOT / "vggt")
    parser.add_argument("--failure-policy", choices=["previous", "error"], default="previous")
    args = parser.parse_args()

    split_dir = args.data_root / args.split
    sequence_dirs = (
        [split_dir / name for name in args.sequence]
        if args.sequence
        else sorted(path for path in split_dir.iterdir() if path.is_dir())
    )
    missing = [path for path in sequence_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing sequences: {missing}")
    config = Method3Config(
        variant=args.variant,
        preprocess_mode=args.preprocess_mode,
        dtype=args.dtype,
        failure_policy=args.failure_policy,
    )
    predictor = VGGTCameraPredictor(
        device=args.device,
        dtype=args.dtype,
        model_id=args.model_id,
        vggt_root=args.vggt_root,
    )
    run_summary: dict[str, object] = {
        "method": f"method{args.variant}",
        "split": args.split,
        "model_id": args.model_id,
        "vggt_local_commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
        "model_load_seconds": predictor.model_load_seconds,
        "config": asdict(config),
        "sample_count": args.sample_count,
        "max_frames": args.max_frames,
        "sequences": {},
    }
    total_estimated_full_seconds = 0.0
    wall_start = time.perf_counter()
    for sequence_dir in tqdm(sequence_dirs, desc=f"Method {args.variant.upper()}"):
        inputs = load_vggt_inputs(sequence_dir)
        selected = select_frame_indices(
            len(inputs.frame_ids),
            max_frames=args.max_frames,
            sample_count=args.sample_count,
        )
        result = run_method3_sequence(
            inputs=inputs,
            predictor=predictor,
            config=config,
            frame_indices=selected,
        )
        sequence_root = args.output_root / args.split / sequence_dir.name
        write_pose_txt(sequence_root / "pose.txt", result.rows)
        np.savez_compressed(
            sequence_root / "raw_vggt_outputs.npz",
            frame_ids=result.frame_ids,
            extrinsics_cam_from_world=result.raw_extrinsics_cam_from_world,
            intrinsics_processed=result.raw_intrinsics_processed,
            scale_factors=result.scale_factors,
            scale_reference_e1_baselines_raw=result.scale_reference_e1_baselines_raw,
            e1_stereo_baselines_raw=result.e1_stereo_baselines_raw,
            e2_stereo_baselines_raw=result.e2_stereo_baselines_raw,
            T_cross_current_scaled=result.T_cross_current_scaled,
            T_cross_reference_scaled=result.T_cross_reference_scaled,
            T_initial_from_current=result.T_initial_from_current,
            T_direct_e20_from_e2t=result.T_direct_e20_from_e2t,
            prediction_valid=result.prediction_valid,
        )
        median_seconds = result.summary["inference_seconds"]["median"]
        estimated_full = float(median_seconds) * len(inputs.frame_ids)
        total_estimated_full_seconds += estimated_full
        result.summary["num_total_sequence_frames"] = len(inputs.frame_ids)
        result.summary["num_selected_frames"] = len(selected)
        result.summary["estimated_full_sequence_camera_seconds"] = estimated_full
        payload = {
            "summary": result.summary,
            "frames": [asdict(item) for item in result.diagnostics],
        }
        sequence_root.mkdir(parents=True, exist_ok=True)
        (sequence_root / "diagnostics.json").write_text(
            json.dumps(payload, indent=2, default=_json_default) + "\n"
        )
        run_summary["sequences"][sequence_dir.name] = result.summary
        print(
            f"{sequence_dir.name}: selected={len(selected)}/{len(inputs.frame_ids)} "
            f"valid={100.0 * result.summary['valid_fraction']:.1f}% "
            f"median={median_seconds:.3f}s/inference "
            f"peak={result.summary['peak_gpu_memory_bytes'] / 2**30:.2f}GiB "
            f"scale-ref E1 baseline p50/p90="
            f"{result.summary['scale_reference_e1_baseline_raw']['median']:.4f}/"
            f"{result.summary['scale_reference_e1_baseline_raw']['p90']:.4f}"
        )
    run_summary["wall_seconds"] = time.perf_counter() - wall_start
    run_summary["estimated_full_split_camera_seconds"] = total_estimated_full_seconds
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / f"run_summary_{args.split}.json"
    summary_path.write_text(
        json.dumps(run_summary, indent=2, default=_json_default) + "\n"
    )
    print(f"Saved Method {args.variant.upper()}: {args.output_root}")
    print(f"Estimated full-split camera time: {total_estimated_full_seconds:.1f}s")


if __name__ == "__main__":
    main()
