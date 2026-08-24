#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.methods.cross_stereo_pnp_reverse.calibration import (
    calibrate_e2_session,
    discover_session_ids,
)
from imcpe.methods.cross_stereo_pnp_reverse.config import ReverseCalibrationConfig
from imcpe.methods.cross_stereo_pnp_reverse.matching import Method2BFeatureMatcher


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate fixed scale-free E2 stereo calibration per session."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--session", action="append", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "method_2b" / "calibration",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--ransac-threshold", type=float, default=1e-3)
    parser.add_argument("--min-matches", type=int, default=50)
    parser.add_argument("--min-essential-inliers", type=int, default=30)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--min-cheirality-fraction", type=float, default=0.50)
    parser.add_argument("--max-rotation-variation-deg", type=float, default=5.0)
    parser.add_argument("--max-translation-variation-deg", type=float, default=20.0)
    parser.add_argument("--min-stable-fraction", type=float, default=0.50)
    parser.add_argument("--max-pooled-points-per-pair", type=int, default=400)
    parser.add_argument("--max-epipolar-p95-px", type=float, default=2.0)
    parser.add_argument("--max-reprojection-median-px", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    splits = tuple(args.splits)
    sessions = (
        [str(value) for value in args.session]
        if args.session
        else discover_session_ids(args.data_root, splits)
    )
    if not sessions:
        raise RuntimeError("No sessions selected")
    config = ReverseCalibrationConfig(
        num_samples=args.num_samples,
        essential_ransac_threshold_normalized=args.ransac_threshold,
        min_matches=args.min_matches,
        min_essential_inliers=args.min_essential_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        min_cheirality_fraction=args.min_cheirality_fraction,
        max_rotation_deviation_deg=args.max_rotation_variation_deg,
        max_translation_deviation_deg=args.max_translation_variation_deg,
        min_stable_fraction=args.min_stable_fraction,
        max_pooled_points_per_pair=args.max_pooled_points_per_pair,
        max_epipolar_p95_px=args.max_epipolar_p95_px,
        max_reprojection_median_px=args.max_reprojection_median_px,
    )
    matcher = Method2BFeatureMatcher(
        device=args.device, max_num_keypoints=args.max_keypoints
    )
    summaries: dict[str, object] = {}
    for requested in sessions:
        normalized = str(requested).removeprefix("session_").zfill(3)
        target = args.output_dir / f"session_{normalized}.npz"
        if target.exists() and not args.overwrite:
            print(f"Skipping existing calibration: {target}")
            continue
        _, diagnostics = calibrate_e2_session(
            data_root=args.data_root,
            session_id=normalized,
            output_dir=args.output_dir,
            matcher=matcher,
            config=config,
            splits=splits,
        )
        summaries[normalized] = {
            key: diagnostics[key]
            for key in (
                "num_frames",
                "num_sampled",
                "num_individually_valid",
                "num_stable_pairs",
                "rotation_p50_deg",
                "rotation_p90_deg",
                "translation_direction_p50_deg",
                "translation_direction_p90_deg",
                "median_epipolar_error_px",
                "reprojection_error_median_px",
                "fixed_transform_appears_valid",
                "validity_failures",
            )
        }
        print(json.dumps(summaries[normalized], indent=2))
    print("METHOD 2B E2 CALIBRATION SUMMARY")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
