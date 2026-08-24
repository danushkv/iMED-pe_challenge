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
from imcpe.methods.cross_stereo_pnp_reverse.config import ReverseMethod2Config
from imcpe.methods.cross_stereo_pnp_reverse.matching import Method2BFeatureMatcher
from imcpe.methods.cross_stereo_pnp_reverse.reverse_pnp import run_sequence_method2b
from imcpe.methods.cross_stereo_pnp_reverse.visualization import (
    save_method2b_debug_image,
)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Method 2B: independent E2-stereo to E1-left reverse PnP."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence", action="append", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--debug-frames", type=int, default=3)
    parser.add_argument("--min-shared-tracks", type=int, default=8)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=200.0)
    parser.add_argument("--disable-calibration-depth-bounds", action="store_true")
    parser.add_argument("--max-stereo-reprojection-error", type=float, default=2.0)
    parser.add_argument("--min-triangulation-angle-deg", type=float, default=0.0)
    parser.add_argument("--pnp-min-points", type=int, default=8)
    parser.add_argument("--pnp-min-inliers", type=int, default=8)
    parser.add_argument("--pnp-iterations", type=int, default=200)
    parser.add_argument("--pnp-reprojection-error", type=float, default=3.0)
    parser.add_argument("--pnp-confidence", type=float, default=0.999)
    parser.add_argument("--disable-refine-lm", action="store_true")
    parser.add_argument("--fail-on-stop-condition", action="store_true")
    parser.add_argument("--fail-on-invalid-calibration", action="store_true")
    args = parser.parse_args()

    split_dir = args.data_root / args.split
    sequence_dirs = (
        [split_dir / name for name in args.sequence]
        if args.sequence
        else sorted(path for path in split_dir.iterdir() if path.is_dir())
    )
    missing = [path for path in sequence_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing sequence directories: {missing}")
    if not sequence_dirs:
        raise RuntimeError(f"No sequences selected under {split_dir}")

    config = ReverseMethod2Config(
        min_shared_tracks=args.min_shared_tracks,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        use_calibration_depth_bounds=not args.disable_calibration_depth_bounds,
        max_stereo_reprojection_error_px=args.max_stereo_reprojection_error,
        min_triangulation_angle_deg=args.min_triangulation_angle_deg,
        pnp_min_points=args.pnp_min_points,
        pnp_min_inliers=args.pnp_min_inliers,
        pnp_iterations=args.pnp_iterations,
        pnp_reprojection_error_px=args.pnp_reprojection_error,
        pnp_confidence=args.pnp_confidence,
        refine_lm=not args.disable_refine_lm,
    )
    matcher = Method2BFeatureMatcher(
        device=args.device, max_num_keypoints=args.max_keypoints
    )
    run_summary: dict[str, object] = {
        "split": args.split,
        "max_frames": args.max_frames,
        "config": asdict(config),
        "sequences": {},
    }
    stop_triggered = False
    invalid_calibration_seen = False
    for sequence_dir in tqdm(sequence_dirs, desc="Method 2B sequences"):
        start = time.perf_counter()
        result = run_sequence_method2b(
            sequence_dir=sequence_dir,
            calibration_root=args.calibration_root,
            matcher=matcher,
            config=config,
            max_frames=args.max_frames,
            debug_frame_count=args.debug_frames,
        )
        elapsed = time.perf_counter() - start
        sequence_root = args.output_root / args.split / sequence_dir.name
        write_pose_txt(sequence_root / "pose.txt", result.rows)
        for debug in result.debug_frames:
            save_method2b_debug_image(
                debug,
                sequence_root / "debug" / f"matches_{debug.frame_idx:06d}.png",
            )
        np.savez_compressed(
            sequence_root / "raw_poses.npz",
            frame_ids=np.asarray([row.frame_idx for row in result.rows], dtype=np.int64),
            T_E1_from_E2=result.raw_T_e1_from_e2,
            T_E2_from_E1=result.raw_T_e2_from_e1,
            T_E20_from_E2t=result.relative_T_initial_from_current,
            pnp_success=result.raw_success,
        )
        payload = {
            "summary": {**result.summary, "runtime_s": elapsed},
            "frames": [asdict(item) for item in result.diagnostics],
        }
        (sequence_root / "diagnostics.json").write_text(
            json.dumps(payload, indent=2, default=_json_default) + "\n"
        )
        run_summary["sequences"][sequence_dir.name] = payload["summary"]
        print(f"\n{sequence_dir.name}: frames={result.summary['num_frames']}")
        for diagnostic in result.diagnostics:
            print(
                f"frame {diagnostic.frame_idx:06d}: "
                f"E2 stereo={diagnostic.e2_stereo_matches} "
                f"E2<->E1={diagnostic.e2_to_e1_matches} "
                f"shared={diagnostic.shared_e2_left_tracks} "
                f"valid3D={diagnostic.valid_triangulated_3d} "
                f"inliers={diagnostic.pnp_inliers} "
                f"ratio={diagnostic.pnp_inlier_ratio:.3f} "
                f"reproj={diagnostic.pnp_reprojection_error_median_px:.3f}px "
                f"|t|={diagnostic.common_translation_magnitude_baselines:.3f} "
                f"status={'OK' if diagnostic.pnp_success else diagnostic.failure_reason}"
            )
        print(
            f"SUMMARY: success={100.0 * result.summary['pnp_success_rate']:.1f}% "
            f"median_shared={result.summary['median_shared_3d_2d_tracks']:.1f} "
            f"median_inliers={result.summary['median_pnp_inliers']:.1f} "
            f"median_reproj={result.summary['median_pnp_reprojection_error_px']:.3f}px "
            f"max_jump={result.summary['maximum_common_translation_jump_ratio']:.2f} "
            f"calibration_valid={result.summary['calibration_valid']} "
            f"runtime={elapsed:.2f}s"
        )
        if result.summary["calibration_valid"] is False:
            invalid_calibration_seen = True
        if result.summary["geometry_stop_recommended"]:
            stop_triggered = True
            print("STOP CONDITIONS:")
            for reason in result.summary["geometry_stop_reasons"]:
                print(f"  - {reason}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / f"run_summary_{args.split}.json"
    summary_path.write_text(
        json.dumps(run_summary, indent=2, default=_json_default) + "\n"
    )
    print(f"Saved pure Method 2B run: {args.output_root}")
    print(f"Saved summary: {summary_path}")
    if invalid_calibration_seen and args.fail_on_invalid_calibration:
        raise SystemExit(3)
    if stop_triggered and args.fail_on_stop_condition:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
