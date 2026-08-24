#!/usr/bin/env python3
"""Freeze the train-selected Method 1.5 settings before test evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-interval", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--max-correction-degrees", type=float, required=True)
    parser.add_argument("--max-disagreement-degrees", type=float)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--selection-results", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/method_1_5/final_config.yaml"),
    )
    return parser.parse_args()


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    if any(character in text for character in (":", "#", "\n")):
        return repr(text)
    return text


def main() -> None:
    args = parse_args()
    if args.anchor_interval <= 0:
        raise ValueError("--anchor-interval must be positive")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    if args.max_correction_degrees <= 0.0:
        raise ValueError("--max-correction-degrees must be positive")
    if not args.thresholds.is_file():
        raise FileNotFoundError(args.thresholds)
    if not args.selection_results.is_file():
        raise FileNotFoundError(args.selection_results)

    values = {
        "method": "method_1_5A_rotation_anchored_stereo_vo",
        "selected_on_split": "train",
        "anchor_interval": args.anchor_interval,
        "alpha": args.alpha,
        "max_correction_degrees": args.max_correction_degrees,
        "max_disagreement_degrees": args.max_disagreement_degrees,
        "thresholds_file": str(args.thresholds),
        "selection_results": str(args.selection_results),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "ground_truth_used_during_inference": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}: {_yaml_scalar(value)}\n")
    print(f"Saved frozen configuration to {args.output}")


if __name__ == "__main__":
    main()

