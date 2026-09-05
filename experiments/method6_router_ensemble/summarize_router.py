#!/usr/bin/env python3
"""Print compact router-only LOSO diagnostics from one or more runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_named(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Expected NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=parse_named, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    print("Run | model | features | alpha | accuracy | balanced | logloss | entropy")
    for name, root in args.run:
        payload = json.loads((root / "loso_router_summary.json").read_text())
        config = payload["router_config"]
        metrics = payload["aggregate_router_metrics"]
        report[name] = {
            "root": str(root),
            "config": config,
            "num_features": payload["num_features"],
            "aggregate_router_metrics": metrics,
        }
        print(
            f"{name:18s} | {config['model']:8s} | {config['feature_set']:2s} | "
            f"{config['smoothing_alpha']:.1f} | {metrics['accuracy']:.4f} | "
            f"{metrics['balanced_accuracy']:.4f} | {metrics['log_loss']:.4f} | "
            f"{metrics['mean_entropy']:.4f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

