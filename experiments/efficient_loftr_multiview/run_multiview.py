#!/usr/bin/env python3
"""Run Point 5B pure EfficientLoFTR multi-view Method 2A or Method 2B."""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EfficientLoFTRConfig, MultiViewConfig
from loftr_wrapper import EfficientLoFTRMatcher
from multiview_matcher import PureLoFTRMultiViewMatcher
from imcpe.io_pose import write_pose_txt
from imcpe.methods.cross_stereo_pnp.config import Method2Config
from imcpe.methods.cross_stereo_pnp.cross_pnp import run_sequence_method2
from imcpe.methods.cross_stereo_pnp.visualization import save_method2_debug_image
from imcpe.methods.cross_stereo_pnp_reverse.config import ReverseMethod2Config
from imcpe.methods.cross_stereo_pnp_reverse.reverse_pnp import run_sequence_method2b
from imcpe.methods.cross_stereo_pnp_reverse.visualization import save_method2b_debug_image


def _json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def _sequence_dirs(args: argparse.Namespace) -> list[Path]:
    selected = sorted(path for path in (args.data_root / args.split).iterdir() if path.is_dir())
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


def _p90(values: list[float]) -> float:
    array = _finite(values)
    return float(np.percentile(array, 90)) if len(array) else math.nan


def _track_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        "median_stereo_raw_matches": ("stereo_raw_matches", _median),
        "median_stereo_filtered_matches": ("stereo_filtered_matches", _median),
        "median_cross_raw_matches": ("cross_raw_matches", _median),
        "median_cross_filtered_matches": ("cross_filtered_matches", _median),
        "median_three_view_tracks": ("three_view_tracks", _median),
        "median_association_rate_vs_stereo": ("association_rate_vs_stereo", _median),
        "median_association_rate_vs_cross": ("association_rate_vs_cross", _median),
        "median_association_distance_px": ("association_distance_median_px", _median),
        "p90_association_distance_px": ("association_distance_p90_px", _p90),
        "median_combined_confidence": ("combined_confidence_median", _median),
        "median_stereo_inference_ms": ("stereo_inference_seconds", lambda x: 1000.0 * _median(x)),
        "median_cross_inference_ms": ("cross_inference_seconds", lambda x: 1000.0 * _median(x)),
    }
    summary: dict[str, Any] = {"num_multiview_frames": len(history)}
    for output_name, (input_name, reducer) in fields.items():
        summary[output_name] = reducer([float(item[input_name]) for item in history]) if history else math.nan
    return summary


def _runtime_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    pairs = int(after["matched_pairs"]) - int(before["matched_pairs"])
    seconds = float(after["inference_seconds"]) - float(before["inference_seconds"])
    return {
        "matched_pairs": pairs,
        "inference_seconds": seconds,
        "inference_ms_per_pair": 1000.0 * seconds / max(pairs, 1),
    }


def main(fixed_direction: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    if fixed_direction is None:
        parser.add_argument("--direction", choices=("2a", "2b"), required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--loftr-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-type", choices=("full", "opt"), default="full")
    parser.add_argument("--precision", choices=("fp32", "mp", "fp16"), default="fp32")
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, choices=(1.0, 2.0, 3.0), required=True)
    parser.add_argument("--session", action="append", default=None)
    parser.add_argument("--sequence", action="append", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--debug-frames", type=int, default=3)
    parser.add_argument(
        "--frame-zero-fallback-root",
        type=Path,
        default=None,
        help=(
            "Optional submission-safe original Method-2 root. If and only if "
            "LoFTR-MV cannot estimate frame zero, copy that sequence prediction "
            "and mark the fallback explicitly. Pure runs should omit this option."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    direction = fixed_direction if fixed_direction is not None else args.direction

    loftr = EfficientLoFTRMatcher(
        loftr_root=args.loftr_root,
        checkpoint=args.checkpoint,
        device=args.device,
        config=EfficientLoFTRConfig(model_type=args.model_type, precision=args.precision),
    )
    matcher = PureLoFTRMultiViewMatcher(
        loftr=loftr,
        config=MultiViewConfig(
            confidence_threshold=args.confidence_threshold,
            association_tolerance_px=args.epsilon,
        ),
    )
    geometry: Method2Config | ReverseMethod2Config = (
        Method2Config() if direction == "2a" else ReverseMethod2Config()
    )
    run_summary: dict[str, Any] = {
        "method": f"method{direction}_efficient_loftr_multiview",
        "split": args.split,
        "direction": direction,
        "calibration_root": str(args.calibration_root),
        "calibration_unchanged": True,
        "frontend": loftr.metadata(),
        "confidence_threshold": args.confidence_threshold,
        "epsilon_original_pixels": args.epsilon,
        "source_association": "mutual nearest one-to-one; distance then confidence tie break",
        "triangulation_source_coordinate": "actual stereo-left LoFTR coordinate",
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

    for sequence_dir in tqdm(_sequence_dirs(args), desc=f"LoFTR-MV {direction.upper()}"):
        sequence_root = args.output_root / args.split / sequence_dir.name
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        history_start = len(matcher.history)
        runtime_before = loftr.runtime.snapshot()
        start = time.perf_counter()
        try:
            if direction == "2a":
                result = run_sequence_method2(
                    sequence_dir=sequence_dir,
                    calibration_root=args.calibration_root,
                    matcher=matcher,
                    config=geometry,
                    max_frames=args.max_frames,
                    debug_frame_count=args.debug_frames,
                )
                rows = result.pure_rows
            else:
                result = run_sequence_method2b(
                    sequence_dir=sequence_dir,
                    calibration_root=args.calibration_root,
                    matcher=matcher,
                    config=geometry,
                    max_frames=args.max_frames,
                    debug_frame_count=args.debug_frames,
                )
                rows = result.rows
        except RuntimeError as error:
            is_frame_zero_failure = "frame-zero" in str(error) or "frame zero" in str(error)
            if args.frame_zero_fallback_root is None or not is_frame_zero_failure:
                raise
            fallback_sequence = args.frame_zero_fallback_root / args.split / sequence_dir.name
            if not (fallback_sequence / "pose.txt").is_file():
                raise RuntimeError(
                    f"Frame-zero fallback requested but missing: {fallback_sequence / 'pose.txt'}"
                ) from error
            if sequence_root.exists():
                raise RuntimeError(
                    f"Refusing to overwrite existing fallback destination: {sequence_root}"
                ) from error
            shutil.copytree(fallback_sequence, sequence_root)
            fallback_summary = {
                "sequence": sequence_dir.name,
                "frame_zero_fallback_used": True,
                "fallback_reason": str(error),
                "fallback_source": str(fallback_sequence),
                "loftr_track_frames_before_failure": matcher.history[history_start:],
                "no_ground_truth_used": True,
            }
            run_summary["sequences"][sequence_dir.name] = fallback_summary
            marker = sequence_root / "loftr_mv_fallback.json"
            marker.write_text(json.dumps(fallback_summary, indent=2, default=_json_default) + "\n")
            print(
                f"{sequence_dir.name}: LoFTR-MV frame zero failed; copied explicit "
                f"Method {direction.upper()} fallback from {fallback_sequence}"
            )
            continue
        elapsed = time.perf_counter() - start
        history = matcher.history[history_start:]
        track_summary = _track_summary(history)
        valid_3d = _median([float(item.valid_triangulated_3d) for item in result.diagnostics])
        peak_gib = (
            torch.cuda.max_memory_allocated() / (1024.0**3)
            if torch.cuda.is_available() and str(args.device).startswith("cuda")
            else 0.0
        )

        write_pose_txt(sequence_root / "pose.txt", rows)
        if direction == "2a":
            np.savez_compressed(
                sequence_root / "raw_poses.npz",
                frame_ids=np.asarray([row.frame_idx for row in rows], dtype=np.int64),
                T_E2_from_E1=result.raw_T_e2_from_e1,
                T_E20_from_E2t=result.relative_T_initial_from_current,
                pnp_success=result.raw_success,
            )
            for debug in result.debug_frames:
                save_method2_debug_image(
                    debug, sequence_root / "debug" / f"matches_{debug.frame_idx:06d}.png"
                )
        else:
            np.savez_compressed(
                sequence_root / "raw_poses.npz",
                frame_ids=np.asarray([row.frame_idx for row in rows], dtype=np.int64),
                T_E1_from_E2=result.raw_T_e1_from_e2,
                T_E2_from_E1=result.raw_T_e2_from_e1,
                T_E20_from_E2t=result.relative_T_initial_from_current,
                pnp_success=result.raw_success,
            )
            for debug in result.debug_frames:
                save_method2b_debug_image(
                    debug, sequence_root / "debug" / f"matches_{debug.frame_idx:06d}.png"
                )

        summary = {
            **result.summary,
            **track_summary,
            "median_valid_triangulated_3d": valid_3d,
            "total_runtime_seconds": elapsed,
            "loftr_runtime": _runtime_delta(runtime_before, loftr.runtime.snapshot()),
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
            f"stereo={summary['median_stereo_filtered_matches']:.1f} "
            f"cross={summary['median_cross_filtered_matches']:.1f} "
            f"tracks={summary['median_three_view_tracks']:.1f} "
            f"rates={100.0 * summary['median_association_rate_vs_stereo']:.1f}%/"
            f"{100.0 * summary['median_association_rate_vs_cross']:.1f}% "
            f"distance={summary['median_association_distance_px']:.3f}px "
            f"valid3D={summary['median_valid_triangulated_3d']:.1f} "
            f"inliers={summary['median_pnp_inliers']:.1f} "
            f"ratio={summary['median_pnp_inlier_ratio']:.3f} "
            f"reproj={summary['median_pnp_reprojection_error_px']:.3f}px "
            f"runtime={elapsed:.2f}s peak={peak_gib:.2f}GiB"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / f"run_summary_{args.split}.json"
    summary_path.write_text(json.dumps(run_summary, indent=2, default=_json_default) + "\n")
    print(f"Saved LoFTR-MV Method {direction.upper()}: {args.output_root}")


if __name__ == "__main__":
    main()
