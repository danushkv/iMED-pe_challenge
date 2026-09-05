#!/usr/bin/env python3
"""Fuse saved absolute-pose observers with frozen Method-1 Stereo-VO edges."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from imcpe.io_pose import PoseRow, write_pose_txt

from .config import Method4CConfig
from .confidence import observer_weights
from .consensus import build_consensus
from .diagnostics import (
    alignment_summary,
    extended_distribution,
    pairwise_disagreements,
)
from .load_observers import load_observer_bundle, parse_named_root
from .optimize import (
    align_vo_displacements,
    optimize_multi_observer_positions,
    vo_weights,
)
from .robust_alignment import (
    align_method1_to_reference,
    align_observers_to_reference,
)


def _json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument(
        "--observer",
        type=parse_named_root,
        action="append",
        required=True,
        help="NAME=ROOT; repeat for every enabled absolute observer",
    )
    parser.add_argument("--reference-observer", default="2b")
    parser.add_argument("--method1-root", type=Path, required=True)
    parser.add_argument("--rotation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence", action="append", default=None)
    parser.add_argument("--mode", choices=("4c0", "4c1"), required=True)
    parser.add_argument(
        "--weighting", choices=("v0", "v1", "method4a-v1"), required=True
    )
    parser.add_argument("--lambda-vo", type=float, default=5.0)
    parser.add_argument("--absolute-weight-scale", type=float, default=1.0)
    parser.add_argument("--vo-weight-scale", type=float, default=1.0)
    parser.add_argument("--huber-delta-steps", type=float, default=2.5)
    parser.add_argument("--alignment-iterations", type=int, default=8)
    parser.add_argument("--alignment-huber-mad", type=float, default=2.5)
    parser.add_argument("--optimizer-max-nfev", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observer_roots = dict(args.observer)
    if len(observer_roots) != len(args.observer):
        raise ValueError("Observer names must be unique")
    if args.reference_observer not in observer_roots:
        raise ValueError("Reference observer was not supplied")
    if len(observer_roots) < 2:
        raise ValueError("Method 4C requires at least two absolute observers")
    if args.mode == "4c0" and args.weighting != "v0":
        raise ValueError("Method 4C-0 is the unweighted geometric median; use v0")

    config = Method4CConfig(
        mode=args.mode,
        weighting=args.weighting,
        lambda_vo=args.lambda_vo,
        absolute_weight_scale=args.absolute_weight_scale,
        vo_weight_scale=args.vo_weight_scale,
        huber_delta_steps=args.huber_delta_steps,
        alignment_iterations=args.alignment_iterations,
        alignment_huber_mad=args.alignment_huber_mad,
        optimizer_max_nfev=args.optimizer_max_nfev,
    )
    reference_split = observer_roots[args.reference_observer] / args.split
    sequences = (
        args.sequence
        if args.sequence
        else sorted(
            path.name
            for path in reference_split.iterdir()
            if path.is_dir() and (path / "pose.txt").is_file()
        )
    )
    if not sequences:
        raise RuntimeError(f"No saved sequences found under {reference_split}")

    run_summary: dict[str, Any] = {
        "method": "method4c_consensus",
        "split": args.split,
        "config": config.__dict__,
        "observer_roots": {name: str(root) for name, root in observer_roots.items()},
        "observer_order": list(observer_roots),
        "reference_observer": args.reference_observer,
        "method1_root": str(args.method1_root),
        "rotation_root": str(args.rotation_root),
        "ground_truth_loaded_during_inference": False,
        "sequences": {},
    }
    wall_started = time.perf_counter()
    for sequence in tqdm(sequences, desc=f"Method {args.mode.upper()}-{args.weighting}"):
        bundle = load_observer_bundle(
            split=args.split,
            sequence=sequence,
            observer_roots=observer_roots,
            reference_name=args.reference_observer,
            method1_root=args.method1_root,
            rotation_root=args.rotation_root,
        )
        aligned = align_observers_to_reference(
            bundle,
            iterations=config.alignment_iterations,
            huber_mad=config.alignment_huber_mad,
        )
        consensus = build_consensus(
            aligned.centers,
            aligned.valid,
            max_iterations=config.weiszfeld_iterations,
            tolerance=config.weiszfeld_tolerance,
        )
        weights = observer_weights(
            bundle=bundle,
            aligned=aligned,
            consensus=consensus,
            version=config.weighting,
        )
        similarity_vo = align_method1_to_reference(
            bundle,
            aligned.reference_centers,
            iterations=config.alignment_iterations,
            huber_mad=config.alignment_huber_mad,
        )
        centers_vo, delta_vo = align_vo_displacements(bundle, similarity_vo)
        edge_weights = vo_weights(bundle)
        optimization = optimize_multi_observer_positions(
            observer_centers=aligned.centers,
            observer_valid=aligned.valid,
            observer_weights=weights.weights,
            delta_vo=delta_vo,
            weight_vo=edge_weights,
            trajectory_step_scale=aligned.trajectory_step_scale,
            config=config,
        )

        rotations = bundle.rotation_source.rotations_initial_from_camera
        determinant_error = np.abs(np.linalg.det(rotations) - 1.0)
        orthogonality_error = np.linalg.norm(
            np.transpose(rotations, (0, 2, 1)) @ rotations - np.eye(3),
            axis=(1, 2),
        )
        if np.max(determinant_error) > 1.0e-7 or np.max(orthogonality_error) > 1.0e-7:
            raise RuntimeError("Frozen rotation source contains invalid SO(3) matrices")
        rows = [
            PoseRow(
                frame_idx=int(frame_id),
                # Verified repository convention: pose.txt is T_0_from_t and
                # stores camera center C_t directly as its translation.
                t=optimization.centers[index].copy(),
                q_xyzw=Rotation.from_matrix(rotations[index]).as_quat(),
            )
            for index, frame_id in enumerate(bundle.frame_ids)
        ]
        if not np.allclose(rows[0].t, 0.0, atol=1.0e-10, rtol=0.0):
            raise RuntimeError("Method 4C frame-zero center is not zero")
        if not np.allclose(rows[0].q_xyzw, [0, 0, 0, 1], atol=1.0e-7, rtol=0.0):
            raise RuntimeError("Method 4C frame-zero rotation is not identity")

        pairwise = pairwise_disagreements(aligned)
        normalized_spread = consensus.spread / max(
            aligned.trajectory_step_scale, 1.0e-9
        )
        summary = {
            "num_frames": len(rows),
            "num_observers": len(aligned.names),
            "observer_names": list(aligned.names),
            "observer_valid_frames": {
                name: int(np.count_nonzero(aligned.valid[index]))
                for index, name in enumerate(aligned.names)
            },
            "config": config.__dict__,
            "alignment": alignment_summary(aligned, args.reference_observer),
            "alignment_vo_to_reference": {
                "scale": similarity_vo.scale,
                "rotation": similarity_vo.rotation,
                "translation": similarity_vo.translation,
                "residual": extended_distribution(similarity_vo.residuals),
            },
            "pairwise_disagreement": pairwise,
            "trajectory_step_scale": aligned.trajectory_step_scale,
            "consensus_spread": extended_distribution(consensus.spread),
            "consensus_spread_normalized": extended_distribution(normalized_spread),
            "consensus_maximum_disagreement": extended_distribution(
                consensus.maximum_disagreement
            ),
            "consensus_pairwise_median_disagreement": extended_distribution(
                consensus.pairwise_median_disagreement
            ),
            "observer_weight_below_0_5": {
                name: int(np.count_nonzero(weights.weights[index] < 0.5))
                for index, name in enumerate(aligned.names)
            },
            "frame_spread_weight_below_0_5": int(
                np.count_nonzero(1.0 / (1.0 + weights.frame_robust_z**2) < 0.5)
            ),
            "optimization_seconds": optimization.optimization_seconds,
            "optimizer_success": optimization.optimizer_success,
            "optimizer_message": optimization.optimizer_message,
            "optimizer_cost": optimization.optimizer_cost,
            "optimizer_nfev": optimization.optimizer_nfev,
            "observer_final_residual": {
                name: extended_distribution(optimization.observer_residuals[index])
                for index, name in enumerate(aligned.names)
            },
            "vo_residual": extended_distribution(optimization.vo_residuals),
            "num_zero_weight_vo_edges": int(np.count_nonzero(edge_weights == 0.0)),
            "no_absolute_max_error": optimization.no_absolute_max_error,
            "lambda_zero_max_error": optimization.lambda_zero_max_error,
            "maximum_rotation_determinant_error": float(np.max(determinant_error)),
            "maximum_rotation_orthogonality_error": float(
                np.max(orthogonality_error)
            ),
            "no_ground_truth_used": True,
        }

        output = args.output_root / args.split / sequence
        output.mkdir(parents=True, exist_ok=True)
        write_pose_txt(output / "pose.txt", rows)
        np.savez_compressed(
            output / "raw_fusion.npz",
            frame_ids=bundle.frame_ids,
            observer_names=np.asarray(aligned.names),
            observer_centers=aligned.centers,
            observer_valid=aligned.valid,
            observer_weights=weights.weights,
            geometric_median=consensus.centers,
            consensus_spread=consensus.spread,
            consensus_spread_normalized=normalized_spread,
            centers_vo_aligned=centers_vo,
            delta_vo_aligned=delta_vo,
            vo_weights=edge_weights,
            centers_initial=optimization.initial_consensus,
            centers_final=optimization.centers,
            observer_final_residuals=optimization.observer_residuals,
            vo_residuals=optimization.vo_residuals,
        )
        frames = []
        for frame_index, frame_id in enumerate(bundle.frame_ids):
            frames.append(
                {
                    "frame_idx": int(frame_id),
                    "valid_observers": [
                        name
                        for index, name in enumerate(aligned.names)
                        if aligned.valid[index, frame_index]
                    ],
                    "observer_residual_to_geometric_median": {
                        name: float(consensus.observer_residuals[index, frame_index])
                        for index, name in enumerate(aligned.names)
                    },
                    "observer_weight": {
                        name: float(weights.weights[index, frame_index])
                        for index, name in enumerate(aligned.names)
                    },
                    "spread": float(consensus.spread[frame_index]),
                    "spread_normalized": float(normalized_spread[frame_index]),
                    "maximum_disagreement": float(
                        consensus.maximum_disagreement[frame_index]
                    ),
                    "pairwise_median_disagreement": float(
                        consensus.pairwise_median_disagreement[frame_index]
                    ),
                    "vo_increment_magnitude": (
                        0.0
                        if frame_index == 0
                        else float(np.linalg.norm(delta_vo[frame_index - 1]))
                    ),
                    "vo_weight": (
                        0.0 if frame_index == 0 else float(edge_weights[frame_index - 1])
                    ),
                    "vo_residual": (
                        0.0
                        if frame_index == 0
                        else float(optimization.vo_residuals[frame_index - 1])
                    ),
                }
            )
        (output / "diagnostics.json").write_text(
            json.dumps(
                {"summary": summary, "frames": frames},
                indent=2,
                default=_json_default,
                allow_nan=True,
            )
            + "\n"
        )
        run_summary["sequences"][sequence] = summary
        print(
            f"{sequence}: observers={len(aligned.names)} "
            f"spread p50={summary['consensus_spread']['median']:.4f} "
            f"VO residual p50={summary['vo_residual']['median']:.4f} "
            f"opt={summary['optimization_seconds']:.4f}s"
        )

    run_summary["wall_seconds"] = time.perf_counter() - wall_started
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / f"run_summary_{args.split}.json"
    summary_path.write_text(
        json.dumps(run_summary, indent=2, default=_json_default, allow_nan=True) + "\n"
    )
    print(f"Saved Method 4C: {args.output_root}")


if __name__ == "__main__":
    main()
