#!/usr/bin/env python3
"""Run L1: ALIKED stereo geometry with EfficientLoFTR cross-camera matches."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import AssociationConfig, EfficientLoFTRConfig
from hybrid_matcher import L1HybridMatcher
from loftr_wrapper import EfficientLoFTRMatcher
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


def _finite_median(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else math.nan


def _finite_p90(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, 90)) if array.size else math.nan


def _association_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {
            "num_cross_pairs": 0,
            "median_raw_loftr_matches": math.nan,
            "median_filtered_loftr_matches": math.nan,
            "median_associated_matches": math.nan,
            "median_association_rate": math.nan,
            "median_association_distance_px": math.nan,
            "p90_association_distance_px": math.nan,
            "median_associated_confidence": math.nan,
            "median_loftr_inference_ms": math.nan,
        }
    return {
        "num_cross_pairs": len(history),
        "median_raw_loftr_matches": _finite_median([x["raw_loftr_matches"] for x in history]),
        "median_filtered_loftr_matches": _finite_median([x["filtered_loftr_matches"] for x in history]),
        "median_associated_matches": _finite_median([x["associated_matches"] for x in history]),
        "median_association_rate": _finite_median([x["association_rate"] for x in history]),
        "median_association_distance_px": _finite_median(
            [x["association_distance_median_px"] for x in history]
        ),
        "p90_association_distance_px": _finite_p90(
            [x["association_distance_p90_px"] for x in history]
        ),
        "median_associated_confidence": _finite_median(
            [x["associated_confidence_median"] for x in history]
        ),
        "median_loftr_inference_ms": 1000.0
        * _finite_median([x["loftr_inference_seconds"] for x in history]),
    }


def _runtime_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    pairs = int(after["matched_pairs"]) - int(before["matched_pairs"])
    seconds = float(after["inference_seconds"]) - float(before["inference_seconds"])
    return {
        "matched_pairs": pairs,
        "inference_seconds": seconds,
        "inference_ms_per_pair": 1000.0 * seconds / max(pairs, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L1: original ALIKED stereo + EfficientLoFTR cross-camera matching."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--direction", choices=("2a", "2b"), required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--loftr-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-type", choices=("full", "opt"), default="full")
    parser.add_argument("--precision", choices=("fp32", "mp", "fp16"), default="fp32")
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--association-tolerance", type=float, default=1.0)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--session", action="append", default=None)
    parser.add_argument("--sequence", action="append", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--debug-frames", type=int, default=3)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    loftr = EfficientLoFTRMatcher(
        loftr_root=args.loftr_root,
        checkpoint=args.checkpoint,
        device=args.device,
        config=EfficientLoFTRConfig(
            model_type=args.model_type,
            precision=args.precision,
        ),
    )
    matcher = L1HybridMatcher(
        loftr=loftr,
        device=args.device,
        max_num_keypoints=args.max_keypoints,
        association=AssociationConfig(
            tolerance_px=args.association_tolerance,
            confidence_threshold=args.confidence_threshold,
        ),
    )
    config: Method2Config | ReverseMethod2Config = (
        Method2Config() if args.direction == "2a" else ReverseMethod2Config()
    )
    run_summary: dict[str, Any] = {
        "method": f"method{args.direction}_l1_efficient_loftr_cross_only",
        "split": args.split,
        "direction": args.direction,
        "calibration_root": str(args.calibration_root),
        "stereo_frontend": "existing ALIKED + LightGlue",
        "cross_camera_frontend": loftr.metadata(),
        "confidence_threshold": args.confidence_threshold,
        "association_tolerance_px": args.association_tolerance,
        "association": "mutual nearest, one-to-one, in original image pixels",
        "geometry_config": asdict(config),
        "geometry_unchanged": True,
        "no_ground_truth_used": True,
        "sequences": {},
    }

    for sequence_dir in tqdm(
        _sequence_dirs(args), desc=f"L1 Method {args.direction.upper()}"
    ):
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        history_start = len(matcher.cross_history)
        runtime_before = loftr.runtime.snapshot()
        start = time.perf_counter()
        if args.direction == "2a":
            result = run_sequence_method2(
                sequence_dir=sequence_dir,
                calibration_root=args.calibration_root,
                matcher=matcher,
                config=config,
                max_frames=args.max_frames,
                debug_frame_count=args.debug_frames,
            )
            rows = result.pure_rows
        else:
            result = run_sequence_method2b(
                sequence_dir=sequence_dir,
                calibration_root=args.calibration_root,
                matcher=matcher,
                config=config,
                max_frames=args.max_frames,
                debug_frame_count=args.debug_frames,
            )
            rows = result.rows
        elapsed = time.perf_counter() - start
        history = matcher.cross_history[history_start:]
        association_summary = _association_summary(history)
        loftr_runtime = _runtime_delta(runtime_before, loftr.runtime.snapshot())
        peak_gib = (
            torch.cuda.max_memory_allocated() / (1024.0**3)
            if torch.cuda.is_available() and str(args.device).startswith("cuda")
            else 0.0
        )

        sequence_root = args.output_root / args.split / sequence_dir.name
        write_pose_txt(sequence_root / "pose.txt", rows)
        if args.direction == "2a":
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
            **association_summary,
            "median_valid_triangulated_3d": _finite_median(
                [float(item.valid_triangulated_3d) for item in result.diagnostics]
            ),
            "total_runtime_seconds": elapsed,
            "loftr_runtime": loftr_runtime,
            "peak_gpu_memory_gib": peak_gib,
            "no_ground_truth_used": True,
        }
        payload = {
            "summary": summary,
            "frames": [asdict(item) for item in result.diagnostics],
            "association_frames": history,
        }
        (sequence_root / "diagnostics.json").write_text(
            json.dumps(payload, indent=2, default=_json_default) + "\n"
        )
        run_summary["sequences"][sequence_dir.name] = summary
        print(
            f"{sequence_dir.name}: success={100.0 * summary['pnp_success_rate']:.1f}% "
            f"LoFTR={summary['median_filtered_loftr_matches']:.1f} "
            f"associated={summary['median_associated_matches']:.1f} "
            f"rate={100.0 * summary['median_association_rate']:.1f}% "
            f"distance={summary['median_association_distance_px']:.3f}px "
            f"valid3D={summary['median_valid_triangulated_3d']:.1f} "
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
    print(f"Saved L1 Method {args.direction.upper()}: {args.output_root}")


if __name__ == "__main__":
    main()
