#!/usr/bin/env python3
"""Run isolated X1/X2/X3 Method-2A or Method-2B ablations."""

from __future__ import annotations

import argparse
import json
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

from config import XFeatConfig
from xfeat_wrapper import XFeatSparseMatcher
from imcpe.io_pose import write_pose_txt
from imcpe.methods.cross_stereo_pnp.config import Method2Config
from imcpe.methods.cross_stereo_pnp.cross_pnp import run_sequence_method2
from imcpe.methods.cross_stereo_pnp.visualization import save_method2_debug_image
from imcpe.methods.cross_stereo_pnp_reverse.config import ReverseMethod2Config
from imcpe.methods.cross_stereo_pnp_reverse.reverse_pnp import run_sequence_method2b
from imcpe.methods.cross_stereo_pnp_reverse.visualization import (
    save_method2b_debug_image,
)
from imcpe.methods.stereo_vo.features import SharedFeatureMatcher


def _json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def _sequence_dirs(args: argparse.Namespace) -> list[Path]:
    split_root = args.data_root / args.split
    selected = sorted(path for path in split_root.iterdir() if path.is_dir())
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


def _runtime_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    extracted = int(after["extracted_images"]) - int(before["extracted_images"])
    pairs = int(after["matched_pairs"]) - int(before["matched_pairs"])
    extraction = float(after["extraction_seconds"]) - float(before["extraction_seconds"])
    matching = float(after["matching_seconds"]) - float(before["matching_seconds"])
    return {
        "extracted_images": extracted,
        "matched_pairs": pairs,
        "extraction_seconds": extraction,
        "matching_seconds": matching,
        "extraction_ms_per_image": 1000.0 * extraction / max(extracted, 1),
        "matching_ms_per_pair": 1000.0 * matching / max(pairs, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--direction", choices=["2a", "2b"], required=True)
    parser.add_argument("--variant", choices=["x1", "x2", "x3"], required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--xfeat-root", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-features", type=int, default=2048)
    parser.add_argument("--min-cosine", type=float, default=-1.0)
    parser.add_argument("--session", action="append", default=None)
    parser.add_argument("--sequence", action="append", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--debug-frames", type=int, default=3)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    frontend = "aliked_lightglue" if args.variant == "x1" else "xfeat_sparse"
    if frontend == "xfeat_sparse":
        if args.xfeat_root is None:
            raise ValueError("--xfeat-root is required for X2/X3")
        matcher: Any = XFeatSparseMatcher(
            xfeat_root=args.xfeat_root,
            weights=args.weights,
            device=args.device,
            config=XFeatConfig(
                max_features=args.max_features,
                min_cosine_similarity=args.min_cosine,
            ),
        )
        frontend_metadata = matcher.metadata()
    else:
        matcher = SharedFeatureMatcher(
            device=args.device, max_num_keypoints=args.max_features
        )
        frontend_metadata = {
            "implementation": "existing ALIKED + LightGlue",
            "max_features": args.max_features,
        }
    config: Method2Config | ReverseMethod2Config = (
        Method2Config() if args.direction == "2a" else ReverseMethod2Config()
    )
    run_summary: dict[str, Any] = {
        "split": args.split,
        "direction": args.direction,
        "variant": args.variant,
        "inference_frontend": frontend,
        "calibration_root": str(args.calibration_root),
        "frontend": frontend_metadata,
        "geometry_config": asdict(config),
        "geometry_unchanged": True,
        "sequences": {},
    }

    for sequence_dir in tqdm(
        _sequence_dirs(args), desc=f"{args.variant.upper()} Method {args.direction.upper()}"
    ):
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        before = (
            matcher.runtime.snapshot()
            if isinstance(matcher, XFeatSparseMatcher)
            else None
        )
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
                    debug,
                    sequence_root / "debug" / f"matches_{debug.frame_idx:06d}.png",
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
                    debug,
                    sequence_root / "debug" / f"matches_{debug.frame_idx:06d}.png",
                )
        frontend_runtime = None
        if isinstance(matcher, XFeatSparseMatcher):
            assert before is not None
            frontend_runtime = _runtime_delta(before, matcher.runtime.snapshot())
        summary = {
            **result.summary,
            "variant": args.variant,
            "inference_frontend": frontend,
            "calibration_root": str(args.calibration_root),
            "total_runtime_seconds": elapsed,
            "frontend_runtime": frontend_runtime,
            "peak_gpu_memory_gib": peak_gib,
            "no_ground_truth_used": True,
        }
        payload = {
            "summary": summary,
            "frames": [asdict(item) for item in result.diagnostics],
        }
        (sequence_root / "diagnostics.json").write_text(
            json.dumps(payload, indent=2, default=_json_default) + "\n"
        )
        run_summary["sequences"][sequence_dir.name] = summary
        print(
            f"{sequence_dir.name}: success={100.0 * summary['pnp_success_rate']:.1f}% "
            f"shared={summary['median_shared_3d_2d_tracks']:.1f} "
            f"inliers={summary['median_pnp_inliers']:.1f} "
            f"reproj={summary['median_pnp_reprojection_error_px']:.3f}px "
            f"runtime={elapsed:.2f}s peak={peak_gib:.2f}GiB"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / f"run_summary_{args.split}.json"
    summary_path.write_text(
        json.dumps(run_summary, indent=2, default=_json_default) + "\n"
    )
    print(f"Saved {args.variant.upper()} Method {args.direction.upper()}: {args.output_root}")


if __name__ == "__main__":
    main()
