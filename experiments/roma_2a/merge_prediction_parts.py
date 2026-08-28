#!/usr/bin/env python3
"""Merge disjoint parallel RoMa prediction roots without touching inference."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--part", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    destination_split = args.output_root / args.split
    destination_split.mkdir(parents=True, exist_ok=True)
    merged_sequences: dict[str, object] = {}
    metadata: dict[str, object] | None = None

    for part_root in args.part:
        summary_path = part_root / f"run_summary_{args.split}.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing part summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        if summary.get("split") != args.split:
            raise RuntimeError(f"Split mismatch in {summary_path}")
        if metadata is None:
            metadata = {key: value for key, value in summary.items() if key != "sequences"}
        else:
            for key in (
                "method",
                "split",
                "calibration_root",
                "certainty_threshold",
                "cycle_threshold_px",
                "symmetric_reverse_reuse",
                "geometry_config",
            ):
                if metadata.get(key) != summary.get(key):
                    raise RuntimeError(
                        f"Part configuration mismatch for {key}: "
                        f"{metadata.get(key)!r} vs {summary.get(key)!r}"
                    )

        source_split = part_root / args.split
        for sequence_name, sequence_summary in summary.get("sequences", {}).items():
            if sequence_name in merged_sequences:
                raise RuntimeError(f"Duplicate sequence across parts: {sequence_name}")
            source_sequence = source_split / sequence_name
            destination_sequence = destination_split / sequence_name
            if not (source_sequence / "pose.txt").is_file():
                raise FileNotFoundError(
                    f"Missing prediction pose: {source_sequence / 'pose.txt'}"
                )
            if destination_sequence.exists():
                raise RuntimeError(
                    f"Refusing to overwrite existing sequence: {destination_sequence}"
                )
            shutil.copytree(source_sequence, destination_sequence)
            merged_sequences[sequence_name] = sequence_summary

    if metadata is None:
        raise RuntimeError("No part metadata loaded")
    merged = {
        **metadata,
        "merged_from_parts": [str(path) for path in args.part],
        "num_merged_sequences": len(merged_sequences),
        "sequences": dict(sorted(merged_sequences.items())),
    }
    summary_path = args.output_root / f"run_summary_{args.split}.json"
    summary_path.write_text(json.dumps(merged, indent=2, allow_nan=True) + "\n")
    print(f"Merged {len(merged_sequences)} sequences into: {args.output_root}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
