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

from imcpe.io_pose import read_pose_txt, write_pose_txt
from imcpe.methods.stereo_vo.features import SharedFeatureMatcher
from imcpe.methods.stereo_vo.visualization import (
    save_debug_match_image,
    save_trajectory_plot,
)
from imcpe.methods.stereo_vo.vo import StereoVOConfig, run_sequence_stereo_vo
from imcpe.pose_metrics import evaluate_sequence


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _baseline_path(baseline_root: Path | None, split: str, sequence_name: str) -> Path | None:
    if baseline_root is None:
        return None
    candidate = baseline_root / split / sequence_name / "pose.txt"
    return candidate if candidate.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Endoscope2 session-calibrated stereo visual odometry."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="Sequence name to process; repeat to select several. Default: full split.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit every selected sequence; intended for the sanity run.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--debug-frames", type=int, default=3)
    parser.add_argument("--baseline-root", type=Path, default=None)
    parser.add_argument("--min-shared-tracks", type=int, default=8)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=200.0)
    parser.add_argument("--max-stereo-reprojection-error", type=float, default=2.0)
    parser.add_argument("--pnp-min-points", type=int, default=8)
    parser.add_argument("--pnp-min-inliers", type=int, default=8)
    parser.add_argument("--pnp-iterations", type=int, default=200)
    parser.add_argument("--pnp-reprojection-error", type=float, default=3.0)
    parser.add_argument("--pnp-confidence", type=float, default=0.999)
    args = parser.parse_args()

    split_dir = args.data_root / args.split
    if args.sequence:
        sequence_dirs = [split_dir / name for name in args.sequence]
        missing = [path for path in sequence_dirs if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing sequence directories: {missing}")
    else:
        sequence_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if not sequence_dirs:
        raise RuntimeError(f"No sequences selected under {split_dir}")

    config = StereoVOConfig(
        min_shared_tracks=args.min_shared_tracks,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        max_stereo_reprojection_error_px=args.max_stereo_reprojection_error,
        pnp_min_points=args.pnp_min_points,
        pnp_min_inliers=args.pnp_min_inliers,
        pnp_iterations=args.pnp_iterations,
        pnp_reprojection_error_px=args.pnp_reprojection_error,
        pnp_confidence=args.pnp_confidence,
    )
    matcher = SharedFeatureMatcher(
        device=args.device,
        max_num_keypoints=args.max_keypoints,
    )
    split_output = args.output_root / args.split
    split_output.mkdir(parents=True, exist_ok=True)
    run_summary: dict[str, object] = {
        "split": args.split,
        "max_frames": args.max_frames,
        "config": asdict(config),
        "sequences": {},
    }

    for sequence_dir in tqdm(sequence_dirs, desc="Sequences"):
        start = time.perf_counter()
        result = run_sequence_stereo_vo(
            sequence_dir=sequence_dir,
            calibration_root=args.calibration_root,
            matcher=matcher,
            config=config,
            max_frames=args.max_frames,
            debug_frame_count=args.debug_frames,
        )
        sequence_output = split_output / sequence_dir.name
        sequence_output.mkdir(parents=True, exist_ok=True)
        write_pose_txt(sequence_output / "pose.txt", result.rows)

        gt_rows_all = read_pose_txt(sequence_dir / "pose.txt")
        predicted_ids = {row.frame_idx for row in result.rows}
        gt_rows = [row for row in gt_rows_all if row.frame_idx in predicted_ids]
        metrics = evaluate_sequence(gt_rows, result.rows)
        baseline_rows = None
        baseline_path = _baseline_path(args.baseline_root, args.split, sequence_dir.name)
        baseline_metrics = None
        if baseline_path is not None:
            baseline_rows_all = read_pose_txt(baseline_path)
            baseline_rows = [row for row in baseline_rows_all if row.frame_idx in predicted_ids]
            baseline_metrics = evaluate_sequence(gt_rows, baseline_rows)

        for debug in result.debug_frames:
            save_debug_match_image(
                debug,
                sequence_output / "debug" / f"matches_{debug.previous_frame_idx:06d}_{debug.frame_idx:06d}.png",
            )
        save_trajectory_plot(
            gt_rows=gt_rows,
            stereo_vo_rows=result.rows,
            baseline_rows=baseline_rows,
            output_path=sequence_output / "trajectory.png",
            title=sequence_dir.name,
        )

        elapsed = time.perf_counter() - start
        diagnostics_payload = {
            "summary": {**result.summary, "runtime_s": elapsed},
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "frames": [asdict(diagnostic) for diagnostic in result.diagnostics],
        }
        (sequence_output / "diagnostics.json").write_text(
            json.dumps(diagnostics_payload, indent=2, default=_json_default) + "\n"
        )
        run_summary["sequences"][sequence_dir.name] = diagnostics_payload["summary"]

        print(f"\n{sequence_dir.name}: frames={result.summary['num_frames']}")
        for diagnostic in result.diagnostics:
            print(
                f"frame {diagnostic.frame_idx:06d}: "
                f"stereo matches={diagnostic.stereo_matches} "
                f"temporal matches={diagnostic.temporal_matches} "
                f"shared tracks={diagnostic.shared_tracks} "
                f"valid 3D={diagnostic.valid_3d_points} "
                f"PnP inliers={diagnostic.pnp_inliers} "
                f"ratio={diagnostic.pnp_inlier_ratio:.3f} "
                f"reproj={diagnostic.pnp_reprojection_error_median_px:.3f}px "
                f"status={'OK' if diagnostic.pnp_success else diagnostic.failure_reason}"
            )
        print(
            f"SUMMARY: success={100.0 * result.summary['pnp_success_rate']:.1f}% "
            f"median_inliers={result.summary['median_pnp_inliers']:.1f} "
            f"ATE={metrics['mean_ate']:.4f}mm runtime={elapsed:.2f}s"
        )

    (split_output / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, default=_json_default) + "\n"
    )
    print(f"Wrote run summary: {split_output / 'run_summary.json'}")


if __name__ == "__main__":
    main()

