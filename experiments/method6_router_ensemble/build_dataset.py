#!/usr/bin/env python3
"""Build inference features and optional TRAIN-only reliability targets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from .build_features import build_inference_features
from .build_targets import build_train_targets
from .config import FusionConfig
from .load_experts import (
    discover_sequences,
    load_bundle,
    load_calibration_diagnostics,
    parse_named_root,
    session_id,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--expert", type=parse_named_root, action="append", required=True)
    parser.add_argument("--method1-root", type=Path, required=True)
    parser.add_argument("--rotation-root", type=Path, required=True)
    parser.add_argument("--e1-calibration-diagnostics", type=Path, required=True)
    parser.add_argument("--e2-calibration-diagnostics", type=Path, required=True)
    parser.add_argument("--with-train-targets", action="store_true")
    parser.add_argument("--sequence", action="append", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.with_train_targets and args.split != "train":
        raise ValueError("GT router targets are allowed only for --split train")
    expert_roots = dict(args.expert)
    if len(expert_roots) != len(args.expert):
        raise ValueError("Expert names must be unique")
    sequences = args.sequence or discover_sequences(expert_roots["2b"], args.split)
    fusion = FusionConfig()
    manifest: dict[str, object] = {
        "split": args.split,
        "targets": bool(args.with_train_targets),
        "ground_truth_use": (
            "TRAIN supervised router labels only"
            if args.with_train_targets
            else "none"
        ),
        "expert_roots": {name: str(path) for name, path in expert_roots.items()},
        "method1_root": str(args.method1_root),
        "rotation_root": str(args.rotation_root),
        "feature_schema": None,
        "sequences": [],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    expected_names: tuple[str, ...] | None = None
    expected_groups: tuple[str, ...] | None = None
    wall_started = time.perf_counter()
    for sequence in tqdm(sequences, desc=f"Method 6 dataset {args.split}"):
        session = session_id(sequence)
        bundle = load_bundle(
            split=args.split,
            sequence=sequence,
            expert_roots=expert_roots,
            method1_root=args.method1_root,
            rotation_root=args.rotation_root,
        )
        inference = build_inference_features(
            bundle=bundle,
            e1_calibration=load_calibration_diagnostics(
                args.e1_calibration_diagnostics, session
            ),
            e2_calibration=load_calibration_diagnostics(
                args.e2_calibration_diagnostics, session
            ),
            fusion=fusion,
        )
        if expected_names is None:
            expected_names = inference.feature_names
            expected_groups = inference.feature_groups
        if inference.feature_names != expected_names or inference.feature_groups != expected_groups:
            raise RuntimeError("Feature schema changed between sequences")
        targets = (
            build_train_targets(
                gt_path=args.data_root / args.split / sequence / "pose.txt",
                bundle=bundle,
                expert_centers_common=inference.expert_centers,
            )
            if args.with_train_targets
            else {}
        )
        relative = Path("sequences") / f"{sequence}.npz"
        path = args.output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            frame_ids=inference.frame_ids,
            features=inference.values,
            expert_centers=inference.expert_centers,
            expert_valid=inference.expert_valid,
            rotation_source=inference.rotation_source,
            delta_vo=inference.delta_vo,
            vo_weights=inference.vo_weights,
            trajectory_step_scale=np.asarray(inference.trajectory_step_scale),
            **targets,
        )
        manifest["sequences"].append(
            {
                "sequence": sequence,
                "session": session,
                "num_frames": int(len(inference.frame_ids)),
                "npz": str(relative),
                "has_targets": bool(args.with_train_targets),
            }
        )
    manifest["feature_schema"] = {
        "names": list(expected_names or ()),
        "groups": list(expected_groups or ()),
        "num_features": len(expected_names or ()),
        "r1_features": sum(group == "r1" for group in (expected_groups or ())),
        "r2_features": sum(group == "r2" for group in (expected_groups or ())),
    }
    manifest["feature_build_wall_seconds"] = time.perf_counter() - wall_started
    manifest["feature_build_ms_per_frame"] = 1000.0 * float(
        manifest["feature_build_wall_seconds"]
    ) / max(
        sum(int(item["num_frames"]) for item in manifest["sequences"]), 1
    )
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Saved Method-6 {args.split} dataset: {args.output_root} "
        f"sequences={len(sequences)} features={len(expected_names or ())}"
    )


if __name__ == "__main__":
    main()
