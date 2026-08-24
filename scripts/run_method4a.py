#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.io_pose import write_pose_txt
from imcpe.methods.hybrid_pose_fusion.config import Method4Config
from imcpe.methods.hybrid_pose_fusion.load_predictions import load_prediction_bundle
from imcpe.methods.hybrid_pose_fusion.optimize import fuse_method4


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Method 4A robust fusion from saved M1/M2A/M2B predictions."
    )
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--method1-root", type=Path, required=True)
    parser.add_argument("--method2a-root", type=Path, required=True)
    parser.add_argument("--method2b-root", type=Path, required=True)
    parser.add_argument(
        "--rotation-root", type=Path, required=True,
        help="Method 1.5A root for final rotations; Method 1 is allowed for train ablations.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence", action="append", default=None)
    parser.add_argument("--mode", choices=["4a0", "4a1"], required=True)
    parser.add_argument("--confidence-version", choices=["v0", "v1", "v2"], default="v1")
    parser.add_argument("--lambda-vo", type=float, default=1.0)
    parser.add_argument("--absolute-weight-scale", type=float, default=1.0)
    parser.add_argument("--vo-weight-scale", type=float, default=1.0)
    parser.add_argument("--huber-delta-steps", type=float, default=2.5)
    parser.add_argument("--alignment-iterations", type=int, default=8)
    parser.add_argument("--alignment-huber-mad", type=float, default=2.5)
    parser.add_argument("--optimizer-max-nfev", type=int, default=200)
    args = parser.parse_args()

    split_root = args.method2b_root / args.split
    sequences = (
        args.sequence
        if args.sequence
        else sorted(path.name for path in split_root.iterdir() if path.is_dir())
    )
    if not sequences:
        raise RuntimeError(f"No sequences selected under {split_root}")
    config = Method4Config(
        mode=args.mode,
        confidence_version=args.confidence_version,
        lambda_vo=args.lambda_vo,
        absolute_weight_scale=args.absolute_weight_scale,
        vo_weight_scale=args.vo_weight_scale,
        huber_delta_steps=args.huber_delta_steps,
        alignment_iterations=args.alignment_iterations,
        alignment_huber_mad=args.alignment_huber_mad,
        optimizer_max_nfev=args.optimizer_max_nfev,
    )
    run_summary: dict[str, object] = {
        "split": args.split,
        "mode": args.mode,
        "config": config.__dict__,
        "prediction_roots": {
            "method1": str(args.method1_root),
            "method2a": str(args.method2a_root),
            "method2b": str(args.method2b_root),
            "rotation": str(args.rotation_root),
        },
        "sequences": {},
    }
    wall_start = time.perf_counter()
    for sequence in tqdm(sequences, desc=f"Method {args.mode.upper()}"):
        bundle = load_prediction_bundle(
            split=args.split,
            sequence=sequence,
            method1_root=args.method1_root,
            method2a_root=args.method2a_root,
            method2b_root=args.method2b_root,
            rotation_root=args.rotation_root,
        )
        result = fuse_method4(bundle, config)
        output = args.output_root / args.split / sequence
        write_pose_txt(output / "pose.txt", result.rows)
        np.savez_compressed(
            output / "raw_fusion.npz",
            frame_ids=bundle.method2b.frame_ids,
            centers_a_aligned=result.centers_a_aligned,
            centers_b=result.centers_b,
            centers_vo_aligned=result.centers_vo_aligned,
            delta_vo_aligned=result.delta_vo_aligned,
            centers_consensus=result.centers_consensus,
            centers_final=result.centers_final,
            weight_a=result.confidence.weight_a,
            weight_b=result.confidence.weight_b,
            weight_vo=result.vo_weights,
            position_disagreement=result.confidence.position_disagreement,
            rotation_disagreement_deg=result.confidence.rotation_disagreement_deg,
            robust_z=result.confidence.robust_z,
            anchor_a_residual=result.optimization.anchor_a_residuals,
            anchor_b_residual=result.optimization.anchor_b_residuals,
            vo_residual=result.optimization.vo_residuals,
        )
        frames = []
        for index, frame_id in enumerate(bundle.method2b.frame_ids):
            frames.append(
                {
                    "frame_idx": int(frame_id),
                    "position_disagreement": float(result.confidence.position_disagreement[index]),
                    "position_disagreement_normalized": float(
                        result.confidence.position_disagreement_normalized[index]
                    ),
                    "rotation_disagreement_deg": float(
                        result.confidence.rotation_disagreement_deg[index]
                    ),
                    "robust_disagreement_z": float(result.confidence.robust_z[index]),
                    "weight_a": float(result.confidence.weight_a[index]),
                    "weight_b": float(result.confidence.weight_b[index]),
                    "a_inlier_ratio": float(result.confidence.a_inlier_ratio[index]),
                    "b_inlier_ratio": float(result.confidence.b_inlier_ratio[index]),
                    "a_reprojection_px": float(result.confidence.a_reprojection_px[index]),
                    "b_reprojection_px": float(result.confidence.b_reprojection_px[index]),
                    "anchor_a_residual": float(result.optimization.anchor_a_residuals[index]),
                    "anchor_b_residual": float(result.optimization.anchor_b_residuals[index]),
                    "vo_increment_magnitude": (
                        0.0 if index == 0 else float(np.linalg.norm(result.delta_vo_aligned[index - 1]))
                    ),
                    "vo_weight": 0.0 if index == 0 else float(result.vo_weights[index - 1]),
                    "vo_residual": (
                        0.0 if index == 0 else float(result.optimization.vo_residuals[index - 1])
                    ),
                }
            )
        payload = {"summary": result.summary, "frames": frames}
        output.mkdir(parents=True, exist_ok=True)
        (output / "diagnostics.json").write_text(
            json.dumps(payload, indent=2, default=_json_default) + "\n"
        )
        run_summary["sequences"][sequence] = result.summary
        print(
            f"{sequence}: frames={len(result.rows)} "
            f"A/B pos p50={result.summary['position_disagreement']['median']:.4f} "
            f"downweighted={result.summary['num_anchor_frames_downweighted_below_0_5']} "
            f"VO residual p50={result.summary['vo_residual']['median']:.4f} "
            f"opt={result.summary['optimization_seconds']:.4f}s"
        )
    run_summary["wall_seconds"] = time.perf_counter() - wall_start
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / f"run_summary_{args.split}.json"
    path.write_text(json.dumps(run_summary, indent=2, default=_json_default) + "\n")
    print(f"Saved Method {args.mode.upper()}: {args.output_root}")


if __name__ == "__main__":
    main()
