#!/usr/bin/env python3
"""Run Point-6 Method 2A with exact-index ALIKED-source RoMa tracks."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.roma_2a.roma_matcher import ExactSourceRoMaMatcher
from experiments.roma_2a.roma_wrapper import OfficialRoMa
from experiments.roma_2a.source_points import ALIKEDSourceDetector
from imcpe.io_pose import write_pose_txt
from imcpe.methods.cross_stereo_pnp.config import Method2Config
from imcpe.methods.cross_stereo_pnp.cross_pnp import run_sequence_method2
from imcpe.methods.cross_stereo_pnp.visualization import save_method2_debug_image


def _json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def _sequence_dirs(args: argparse.Namespace) -> list[Path]:
    selected = sorted(
        path for path in (args.data_root / args.split).iterdir() if path.is_dir()
    )
    if args.session:
        prefixes = tuple(
            f"session_{str(value).removeprefix('session_').zfill(3)}_"
            for value in args.session
        )
        selected = [path for path in selected if path.name.startswith(prefixes)]
    if args.sequence:
        names = set(args.sequence)
        selected = [path for path in selected if path.name in names]
    if not selected:
        raise RuntimeError("No sequences selected")
    return selected


def _finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _median(values: list[float]) -> float:
    array = _finite(values)
    return float(np.median(array)) if len(array) else math.nan


def _track_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "aliked_source_points",
        "valid_stereo_warps",
        "valid_cross_warps",
        "stereo_certainty_surviving",
        "cross_certainty_surviving",
        "stereo_cycle_surviving",
        "cross_cycle_surviving",
        "joint_three_view_tracks",
        "stereo_certainty_median",
        "cross_certainty_median",
        "stereo_cycle_error_median_px",
        "cross_cycle_error_median_px",
        "stereo_forward_seconds",
        "stereo_reverse_seconds",
        "cross_forward_seconds",
        "cross_reverse_seconds",
        "peak_gpu_memory_gib",
    )
    summary: dict[str, Any] = {"num_roma_frames": len(history)}
    for field in fields:
        summary[f"median_{field}"] = _median(
            [float(item[field]) for item in history]
        ) if history else math.nan
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--roma-root", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-source-points", type=int, default=2048)
    parser.add_argument("--certainty-threshold", type=float, default=None)
    parser.add_argument("--cycle-threshold-px", type=float, default=None)
    parser.add_argument(
        "--disable-symmetric-reverse-reuse",
        action="store_true",
        help=(
            "Regression-only legacy mode: run a separate target-to-source "
            "RoMa inference for cycle checking instead of reusing the reverse "
            "field returned by the official symmetric forward inference."
        ),
    )
    parser.add_argument("--session", action="append", default=None)
    parser.add_argument("--sequence", action="append", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--debug-frames", type=int, default=3)
    parser.add_argument(
        "--frame-zero-fallback-root",
        type=Path,
        default=None,
        help=(
            "Optional submission-safe original Method-2A root. Pure TRAIN runs "
            "should omit this. Used only when RoMa frame-zero PnP fails."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geometry = Method2Config()
    roma = OfficialRoMa(
        roma_root=args.roma_root,
        device=args.device,
        torch_home=args.torch_home,
    )
    detector = ALIKEDSourceDetector(args.max_source_points, args.device)
    matcher = ExactSourceRoMaMatcher(
        roma=roma,
        detector=detector,
        certainty_threshold=args.certainty_threshold,
        cycle_threshold_px=args.cycle_threshold_px,
        reuse_symmetric_reverse=not args.disable_symmetric_reverse_reuse,
    )

    run_summary: dict[str, Any] = {
        "method": "method2a_roma_exact_source",
        "split": args.split,
        "calibration_root": str(args.calibration_root),
        "calibration_unchanged": True,
        "frontend": roma.metadata(),
        "source_detector": {
            "name": "ALIKED",
            "max_source_points": args.max_source_points,
            "LightGlue_used": False,
        },
        "certainty_threshold": args.certainty_threshold,
        "cycle_threshold_px": args.cycle_threshold_px,
        "symmetric_reverse_reuse": not args.disable_symmetric_reverse_reuse,
        "track_certainty": "min(stereo certainty, cross certainty)",
        "source_identity": "exact shared ALIKED E1-L index",
        "nearest_neighbour_association_used": False,
        "geometry_config": asdict(geometry),
        "geometry_unchanged": True,
        "no_ground_truth_used": True,
        "frame_zero_fallback_root": (
            str(args.frame_zero_fallback_root)
            if args.frame_zero_fallback_root is not None
            else None
        ),
        "sequences": {},
    }

    for sequence_dir in tqdm(_sequence_dirs(args), desc="RoMa-2A"):
        sequence_root = args.output_root / args.split / sequence_dir.name
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        history_start = len(matcher.history)
        started = time.perf_counter()
        try:
            result = run_sequence_method2(
                sequence_dir=sequence_dir,
                calibration_root=args.calibration_root,
                matcher=matcher,
                config=geometry,
                max_frames=args.max_frames,
                debug_frame_count=args.debug_frames,
            )
        except RuntimeError as error:
            is_frame_zero_failure = "frame-zero" in str(error) or "frame zero" in str(error)
            if args.frame_zero_fallback_root is None or not is_frame_zero_failure:
                raise
            fallback_sequence = (
                args.frame_zero_fallback_root / args.split / sequence_dir.name
            )
            if not (fallback_sequence / "pose.txt").is_file():
                raise RuntimeError(
                    "Frame-zero fallback requested but missing: "
                    f"{fallback_sequence / 'pose.txt'}"
                ) from error
            if sequence_root.exists():
                raise RuntimeError(
                    f"Refusing to overwrite fallback destination: {sequence_root}"
                ) from error
            shutil.copytree(fallback_sequence, sequence_root)
            fallback_summary = {
                "sequence": sequence_dir.name,
                "frame_zero_fallback_used": True,
                "fallback_reason": str(error),
                "fallback_source": str(fallback_sequence),
                "roma_track_frames_before_failure": matcher.history[history_start:],
                "no_ground_truth_used": True,
            }
            run_summary["sequences"][sequence_dir.name] = fallback_summary
            (sequence_root / "roma_fallback.json").write_text(
                json.dumps(fallback_summary, indent=2, default=_json_default) + "\n"
            )
            print(
                f"{sequence_dir.name}: RoMa frame zero failed; copied explicit "
                f"Method-2A fallback from {fallback_sequence}"
            )
            continue

        elapsed = time.perf_counter() - started
        history = matcher.history[history_start:]
        track_summary = _track_summary(history)
        peak_gib = (
            torch.cuda.max_memory_allocated() / (1024.0**3)
            if torch.cuda.is_available() and str(args.device).startswith("cuda")
            else 0.0
        )

        write_pose_txt(sequence_root / "pose.txt", result.pure_rows)
        np.savez_compressed(
            sequence_root / "raw_poses.npz",
            frame_ids=np.asarray(
                [row.frame_idx for row in result.pure_rows], dtype=np.int64
            ),
            T_E2_from_E1=result.raw_T_e2_from_e1,
            T_E20_from_E2t=result.relative_T_initial_from_current,
            pnp_success=result.raw_success,
        )
        for debug in result.debug_frames:
            save_method2_debug_image(
                debug,
                sequence_root / "debug" / f"matches_{debug.frame_idx:06d}.png",
            )

        summary = {
            **result.summary,
            **track_summary,
            "total_runtime_seconds": elapsed,
            "peak_gpu_memory_gib": peak_gib,
            "no_ground_truth_used": True,
            "frame_zero_fallback_used": False,
        }
        payload = {
            "summary": summary,
            "frames": [asdict(item) for item in result.diagnostics],
            "track_frames": history,
        }
        (sequence_root / "diagnostics.json").write_text(
            json.dumps(payload, indent=2, default=_json_default) + "\n"
        )
        run_summary["sequences"][sequence_dir.name] = summary
        print(
            f"{sequence_dir.name}: success={100.0 * summary['pnp_success_rate']:.1f}% "
            f"source={summary['median_aliked_source_points']:.1f} "
            f"tracks={summary['median_joint_three_view_tracks']:.1f} "
            f"valid3D={summary['median_shared_3d_2d_tracks']:.1f} "
            f"inliers={summary['median_pnp_inliers']:.1f} "
            f"ratio={summary['median_pnp_inlier_ratio']:.3f} "
            f"reproj={summary['median_pnp_reprojection_error_px']:.3f}px "
            f"runtime={elapsed:.2f}s peak={peak_gib:.2f}GiB"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / f"run_summary_{args.split}.json"
    summary_path.write_text(
        json.dumps(run_summary, indent=2, default=_json_default) + "\n"
    )
    print(f"Saved RoMa-2A: {args.output_root}")


if __name__ == "__main__":
    main()
