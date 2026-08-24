#!/usr/bin/env python3
"""Validate the public dataset contract without performing inference."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.data_io import load_sequence  # noqa: E402


SESSION_PATTERN = re.compile(r"^(session_\d+)_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", action="append", choices=("train", "test"))
    args = parser.parse_args()

    splits = args.split or [name for name in ("train", "test") if (args.data_root / name).is_dir()]
    if not splits:
        raise SystemExit(f"No train/test directories under {args.data_root}")

    sessions: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for split in splits:
        sequence_dirs = sorted(path for path in (args.data_root / split).iterdir() if path.is_dir())
        for sequence_dir in sequence_dirs:
            match = SESSION_PATTERN.match(sequence_dir.name)
            if match is None:
                raise RuntimeError(f"Cannot parse session from {sequence_dir.name}")
            sequence = load_sequence(sequence_dir)
            sessions[f"{split}/{match.group(1)}"].append(
                (sequence.sequence_name, len(sequence.frame_ids))
            )

    total_sequences = 0
    total_frames = 0
    for session, sequences in sorted(sessions.items()):
        frames = sum(count for _, count in sequences)
        total_sequences += len(sequences)
        total_frames += frames
        print(f"{session}: {len(sequences)} sequences, {frames} frames")
        for name, count in sequences:
            print(f"  {name}: {count}")
    print(f"OK: {total_sequences} sequences, {total_frames} synchronized timestamps")


if __name__ == "__main__":
    main()

