#!/usr/bin/env python3
"""Strict physical-session LOSO router training and routed prediction."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from .config import FusionConfig, RouterConfig
from .dataset import feature_indices, load_dataset, stack_records
from .output import write_routed_outputs
from .router import (
    mask_invalid_probabilities,
    router_metrics,
    smooth_probabilities,
    train_router,
)
from .routed_optimizer import route_and_optimize


def _json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", choices=("logistic", "histgb"), required=True)
    parser.add_argument("--feature-set", choices=("r1", "r2", "r3"), required=True)
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=1.0,
        help="1 means no smoothing; otherwise EMA current-sample coefficient",
    )
    parser.add_argument("--hard-target", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_dataset(args.dataset_root)
    if dataset.split != "train" or any(record.expert_errors is None for record in dataset.records):
        raise ValueError("LOSO requires a TRAIN dataset with targets")
    router_config = RouterConfig(
        model=args.model,
        feature_set=args.feature_set,
        smoothing_alpha=args.smoothing_alpha,
        soft_target=not args.hard_target,
    )
    fusion = FusionConfig()
    indices = feature_indices(dataset, args.feature_set)
    all_probabilities: list[np.ndarray] = []
    all_errors: list[np.ndarray] = []
    all_features: list[np.ndarray] = []
    folds: dict[str, Any] = {}
    started = time.perf_counter()
    for held_session in dataset.sessions:
        development = [record for record in dataset.records if record.session != held_session]
        held = [record for record in dataset.records if record.session == held_session]
        development_sessions = sorted({record.session for record in development})
        if held_session in development_sessions:
            raise RuntimeError("Held session leaked into development set")
        train_x, train_errors = stack_records(development, indices)
        router = train_router(
            features=np.concatenate([record.features for record in development]),
            errors=train_errors,
            feature_indices=indices,
            feature_names=dataset.feature_names,
            config=router_config,
        )
        fold_probabilities: list[np.ndarray] = []
        fold_errors: list[np.ndarray] = []
        fold_sequence_metrics: dict[str, Any] = {}
        for record in held:
            probabilities = mask_invalid_probabilities(
                smooth_probabilities(
                    mask_invalid_probabilities(
                        router.predict_proba(record.features), record.expert_valid
                    ),
                    args.smoothing_alpha,
                ),
                record.expert_valid,
            )
            routed = route_and_optimize(
                expert_centers=record.expert_centers,
                expert_valid=record.expert_valid,
                probabilities=probabilities,
                delta_vo=record.delta_vo,
                vo_weights=record.vo_weights,
                trajectory_step_scale=record.trajectory_step_scale,
                fusion=fusion,
            )
            fold_sequence_metrics[record.sequence] = write_routed_outputs(
                output_root=args.output_root,
                record=record,
                probabilities=probabilities,
                routed=routed,
                fold=held_session,
            )
            assert record.expert_errors is not None
            fold_probabilities.append(probabilities)
            fold_errors.append(record.expert_errors)
            all_features.append(record.features[:, indices])
        probabilities_fold = np.concatenate(fold_probabilities)
        errors_fold = np.concatenate(fold_errors)
        folds[held_session] = {
            "held_session": held_session,
            "development_sessions": development_sessions,
            "num_development_frames": int(len(train_x)),
            "tau_mm": router.tau_mm,
            "router_metrics": router_metrics(probabilities_fold, errors_fold),
            "sequences": fold_sequence_metrics,
        }
        all_probabilities.append(probabilities_fold)
        all_errors.append(errors_fold)
        print(
            f"held session_{held_session}: "
            f"accuracy={folds[held_session]['router_metrics']['accuracy']:.3f} "
            f"balanced={folds[held_session]['router_metrics']['balanced_accuracy']:.3f} "
            f"entropy={folds[held_session]['router_metrics']['mean_entropy']:.3f}"
        )
    probabilities_all = np.concatenate(all_probabilities)
    errors_all = np.concatenate(all_errors)
    features_all = np.concatenate(all_features)
    correlations: dict[str, dict[str, float]] = {}
    for local_index, global_index in enumerate(indices):
        feature = features_all[:, local_index]
        finite = np.isfinite(feature)
        correlations[dataset.feature_names[global_index]] = {}
        for expert_index, expert_name in enumerate(("2a", "2b", "loftr")):
            probability = probabilities_all[:, expert_index]
            value = (
                float(np.corrcoef(feature[finite], probability[finite])[0, 1])
                if np.count_nonzero(finite) > 2
                and np.std(feature[finite]) > 0
                and np.std(probability[finite]) > 0
                else float("nan")
            )
            correlations[dataset.feature_names[global_index]][expert_name] = value
    report = {
        "method": "method6_router_loso",
        "router_config": router_config.to_dict(),
        "fusion_config": fusion.to_dict(),
        "num_features": int(len(indices)),
        "feature_names": [dataset.feature_names[index] for index in indices],
        "strict_session_loso": True,
        "ground_truth_use": "development-session router supervision and held-session evaluation only",
        "folds": folds,
        "aggregate_router_metrics": router_metrics(probabilities_all, errors_all),
        "feature_probability_correlation": correlations,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "loso_router_summary.json").write_text(
        json.dumps(report, indent=2, default=_json_default, allow_nan=True) + "\n"
    )
    print(f"Saved Method-6 LOSO predictions: {args.output_root}")


if __name__ == "__main__":
    main()
