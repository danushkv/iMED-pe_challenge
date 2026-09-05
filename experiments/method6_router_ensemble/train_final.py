#!/usr/bin/env python3
"""Fit and serialize one frozen router on all TRAIN sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from .config import FusionConfig, RouterConfig
from .dataset import feature_indices, load_dataset, stack_records
from .router import train_router


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", choices=("logistic", "histgb"), required=True)
    parser.add_argument("--feature-set", choices=("r1", "r2", "r3"), required=True)
    parser.add_argument("--smoothing-alpha", type=float, required=True)
    parser.add_argument("--hard-target", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_dataset(args.dataset_root)
    if dataset.split != "train":
        raise ValueError("Final router must be fitted on the TRAIN dataset")
    config = RouterConfig(
        model=args.model,
        feature_set=args.feature_set,
        smoothing_alpha=args.smoothing_alpha,
        soft_target=not args.hard_target,
    )
    indices = feature_indices(dataset, args.feature_set)
    _, errors = stack_records(dataset.records, indices)
    features = np.concatenate([record.features for record in dataset.records])
    router = train_router(
        features=features,
        errors=errors,
        feature_indices=indices,
        feature_names=dataset.feature_names,
        config=config,
    )
    schema_payload = json.dumps(
        {"names": dataset.feature_names, "groups": dataset.feature_groups},
        sort_keys=True,
    )
    artifact = {
        "router": router,
        "router_config": config.to_dict(),
        "fusion_config": FusionConfig().to_dict(),
        "feature_names": dataset.feature_names,
        "feature_groups": dataset.feature_groups,
        "feature_schema_sha256": hashlib.sha256(schema_payload.encode()).hexdigest(),
        "train_sessions": dataset.sessions,
        "num_train_sequences": len(dataset.records),
        "num_train_frames": int(sum(len(record.frame_ids) for record in dataset.records)),
        "no_hidden_or_test_data_used": True,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "router.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    (args.output_root / "config.json").write_text(
        json.dumps({key: value for key, value in artifact.items() if key != "router"}, indent=2)
        + "\n"
    )
    print(f"Saved frozen Method-6 router: {args.output_root}")


if __name__ == "__main__":
    main()
