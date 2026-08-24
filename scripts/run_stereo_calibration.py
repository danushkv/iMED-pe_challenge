from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.matcher import ALikeLightGlueMatcher
from imcpe.methods.stereo_vo.calibration import CalibrationConfig, calibrate_session


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate one fixed, scale-free Endoscope2 stereo transform per session."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--session", required=True, help="Session ID, e.g. 001 or session_001")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "estimated_calibration")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--ransac-threshold", type=float, default=1e-3)
    parser.add_argument("--min-matches", type=int, default=50)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--min-cheirality-fraction", type=float, default=0.50)
    parser.add_argument("--max-rotation-variation-deg", type=float, default=5.0)
    parser.add_argument("--max-translation-variation-deg", type=float, default=20.0)
    parser.add_argument("--min-stable-fraction", type=float, default=0.50)
    parser.add_argument("--max-pooled-points-per-pair", type=int, default=400)
    parser.add_argument("--max-epipolar-p95-px", type=float, default=2.0)
    parser.add_argument("--max-reprojection-median-px", type=float, default=2.0)
    args = parser.parse_args()

    config = CalibrationConfig(
        num_samples=args.num_samples,
        essential_ransac_threshold=args.ransac_threshold,
        min_matches=args.min_matches,
        min_inlier_ratio=args.min_inlier_ratio,
        min_cheirality_fraction=args.min_cheirality_fraction,
        max_rotation_deviation_deg=args.max_rotation_variation_deg,
        max_translation_deviation_deg=args.max_translation_variation_deg,
        min_stable_fraction=args.min_stable_fraction,
        max_pooled_points_per_pair=args.max_pooled_points_per_pair,
        max_epipolar_p95_px=args.max_epipolar_p95_px,
        max_reprojection_median_px=args.max_reprojection_median_px,
    )
    matcher = ALikeLightGlueMatcher(device=args.device)
    _, diagnostics = calibrate_session(
        data_root=args.data_root,
        session_id=args.session,
        output_dir=args.output_dir,
        matcher=matcher,
        config=config,
        splits=tuple(args.splits),
    )
    summary_keys = [
        "session_id",
        "num_frames",
        "num_sampled",
        "num_individually_valid",
        "num_stable_pairs",
        "pooled_essential_inlier_ratio",
        "rotation_variation_median_deg",
        "rotation_variation_p90_deg",
        "translation_direction_variation_median_deg",
        "translation_direction_variation_p90_deg",
        "pooled_inlier_epipolar_sampson_median_approx_px",
        "cheirality_fraction",
        "reprojection_error_median_px",
        "reprojection_error_p95_px",
        "fixed_transform_appears_valid",
        "validity_failures",
    ]
    print("\nSESSION CALIBRATION SUMMARY")
    print(json.dumps({key: diagnostics[key] for key in summary_keys}, indent=2))


if __name__ == "__main__":
    main()
