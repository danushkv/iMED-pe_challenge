#!/usr/bin/env python3
"""Inference-only routed prediction with a frozen all-TRAIN router."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from .config import FusionConfig
from .dataset import load_dataset
from .output import write_routed_outputs
from .router import mask_invalid_probabilities, smooth_probabilities
from .routed_optimizer import route_and_optimize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_dataset(args.dataset_root)
    with (args.router_root / "router.pkl").open("rb") as stream:
        artifact = pickle.load(stream)
    schema_payload = json.dumps(
        {"names": dataset.feature_names, "groups": dataset.feature_groups}, sort_keys=True
    )
    if hashlib.sha256(schema_payload.encode()).hexdigest() != artifact["feature_schema_sha256"]:
        raise RuntimeError("Inference feature schema differs from the frozen router")
    router = artifact["router"]
    fusion = FusionConfig(**artifact["fusion_config"])
    sequence_metrics = {}
    started = time.perf_counter()
    router_seconds = 0.0
    smoothing_seconds = 0.0
    routing_seconds = 0.0
    total_frames = 0
    for record in dataset.records:
        total_frames += len(record.frame_ids)
        phase = time.perf_counter()
        raw_probabilities = mask_invalid_probabilities(
            router.predict_proba(record.features), record.expert_valid
        )
        router_seconds += time.perf_counter() - phase
        phase = time.perf_counter()
        probabilities = mask_invalid_probabilities(
            smooth_probabilities(raw_probabilities, router.config.smoothing_alpha),
            record.expert_valid,
        )
        smoothing_seconds += time.perf_counter() - phase
        phase = time.perf_counter()
        routed = route_and_optimize(
            expert_centers=record.expert_centers,
            expert_valid=record.expert_valid,
            probabilities=probabilities,
            delta_vo=record.delta_vo,
            vo_weights=record.vo_weights,
            trajectory_step_scale=record.trajectory_step_scale,
            fusion=fusion,
        )
        routing_seconds += time.perf_counter() - phase
        sequence_metrics[record.sequence] = write_routed_outputs(
            output_root=args.output_root,
            record=record,
            probabilities=probabilities,
            routed=routed,
            fold="final_all_train",
        )
    report = {
        "method": "method6_final",
        "split": dataset.split,
        "router_config": artifact["router_config"],
        "fusion_config": artifact["fusion_config"],
        "router_inference_and_optimization_seconds": time.perf_counter() - started,
        "router_prediction_seconds": router_seconds,
        "probability_smoothing_seconds": smoothing_seconds,
        "routed_fusion_seconds": routing_seconds,
        "num_frames": total_frames,
        "router_prediction_ms_per_frame": 1000.0 * router_seconds / max(total_frames, 1),
        "probability_smoothing_ms_per_frame": 1000.0 * smoothing_seconds / max(total_frames, 1),
        "routed_fusion_ms_per_frame": 1000.0 * routing_seconds / max(total_frames, 1),
        "ground_truth_loaded": False,
        "sequences": sequence_metrics,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / f"run_summary_{dataset.split}.json").write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n"
    )
    print(f"Saved final Method-6 predictions: {args.output_root}")


if __name__ == "__main__":
    main()
