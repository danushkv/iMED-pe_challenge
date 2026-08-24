#!/usr/bin/env python3
"""Session stereo calibration with official sparse XFeat correspondences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import XFeatConfig
from xfeat_wrapper import XFeatSparseMatcher
from imcpe.methods.cross_stereo_pnp.config import SessionCalibrationConfig
from imcpe.methods.cross_stereo_pnp.session_calibration import (
    calibrate_e1_session,
    discover_session_ids,
)
from imcpe.methods.cross_stereo_pnp_reverse.calibration import calibrate_e2_session
from imcpe.methods.cross_stereo_pnp_reverse.config import ReverseCalibrationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--camera", choices=["e1", "e2"], required=True)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--session", action="append", default=None)
    parser.add_argument("--xfeat-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-features", type=int, default=2048)
    parser.add_argument("--min-cosine", type=float, default=-1.0)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    splits = tuple(args.splits)
    sessions = (
        args.session
        if args.session
        else discover_session_ids(args.data_root, splits)
    )
    matcher = XFeatSparseMatcher(
        xfeat_root=args.xfeat_root,
        weights=args.weights,
        device=args.device,
        config=XFeatConfig(
            max_features=args.max_features,
            min_cosine_similarity=args.min_cosine,
        ),
    )
    summaries: dict[str, object] = {}
    for requested in sessions:
        session = str(requested).removeprefix("session_").zfill(3)
        target = args.output_dir / f"session_{session}.npz"
        if target.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {target}; use a new output directory"
            )
        common = dict(
            data_root=args.data_root,
            session_id=session,
            output_dir=args.output_dir,
            matcher=matcher,
            splits=splits,
        )
        if args.camera == "e1":
            _, diagnostics = calibrate_e1_session(
                **common,
                config=SessionCalibrationConfig(num_samples=args.num_samples),
            )
        else:
            _, diagnostics = calibrate_e2_session(
                **common,
                config=ReverseCalibrationConfig(num_samples=args.num_samples),
            )
        diagnostics["correspondence_frontend"] = matcher.metadata()
        diagnostics["no_ground_truth_used"] = True
        diagnostics_path = args.output_dir / f"session_{session}_diagnostics.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
        summaries[session] = {
            "num_frames": diagnostics.get("num_frames"),
            "num_sampled": diagnostics.get("num_sampled"),
            "num_individually_valid": diagnostics.get("num_individually_valid"),
            "num_stable_pairs": diagnostics.get("num_stable_pairs"),
            "rotation_p50_deg": diagnostics.get(
                "rotation_p50_deg",
                diagnostics.get("rotation_variation_median_deg"),
            ),
            "rotation_p90_deg": diagnostics.get(
                "rotation_p90_deg",
                diagnostics.get("rotation_variation_p90_deg"),
            ),
            "translation_direction_p50_deg": diagnostics.get(
                "translation_direction_p50_deg",
                diagnostics.get("translation_direction_variation_median_deg"),
            ),
            "translation_direction_p90_deg": diagnostics.get(
                "translation_direction_p90_deg",
                diagnostics.get("translation_direction_variation_p90_deg"),
            ),
            "reprojection_error_median_px": diagnostics.get(
                "reprojection_error_median_px"
            ),
            "fixed_transform_appears_valid": diagnostics.get(
                "fixed_transform_appears_valid"
            ),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "xfeat_calibration_summary.json"
    output.write_text(
        json.dumps(
            {
                "camera": args.camera,
                "splits": list(splits),
                "frontend": matcher.metadata(),
                "sessions": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved XFeat {args.camera.upper()} calibration: {args.output_dir}")
    print(f"Saved summary: {output}")


if __name__ == "__main__":
    main()
