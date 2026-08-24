#!/usr/bin/env python3
"""Discover physical acquisition sessions without hardcoded session IDs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from imcpe.io_pose import read_pose_txt


SESSION_PATTERN = re.compile(r"^(session_\d+)(?:_|$)")


@dataclass(frozen=True)
class SequenceRecord:
    split: str
    session_id: str
    sequence: str
    path: Path
    num_frames: int
    frame_ids: tuple[int, ...]

    def to_json(self) -> dict[str, object]:
        result = asdict(self)
        result["path"] = str(self.path)
        result["frame_ids"] = list(self.frame_ids)
        return result


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    sequences: tuple[SequenceRecord, ...]

    @property
    def num_frames(self) -> int:
        return sum(item.num_frames for item in self.sequences)

    def to_json(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "num_sequences": len(self.sequences),
            "num_frames": self.num_frames,
            "sequences": [item.to_json() for item in self.sequences],
        }


def session_id_from_sequence(sequence: str) -> str:
    match = SESSION_PATTERN.match(sequence)
    if match is None:
        raise ValueError(
            f"sequence {sequence!r} does not begin with the required session_<digits> prefix"
        )
    return match.group(1)


def discover_sessions(data_root: Path, split: str) -> tuple[SessionRecord, ...]:
    split_root = data_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"dataset split does not exist: {split_root}")

    grouped: dict[str, list[SequenceRecord]] = {}
    seen_sequences: set[str] = set()
    for sequence_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        sequence = sequence_dir.name
        if sequence in seen_sequences:
            raise RuntimeError(f"duplicate sequence directory: {sequence}")
        seen_sequences.add(sequence)
        session_id = session_id_from_sequence(sequence)
        pose_path = sequence_dir / "pose.txt"
        if not pose_path.is_file():
            raise FileNotFoundError(f"missing evaluation pose file: {pose_path}")
        rows = read_pose_txt(pose_path)
        frame_ids = tuple(int(row.frame_idx) for row in rows)
        if len(frame_ids) != len(set(frame_ids)):
            raise RuntimeError(f"duplicate frame IDs in {pose_path}")
        grouped.setdefault(session_id, []).append(
            SequenceRecord(
                split=split,
                session_id=session_id,
                sequence=sequence,
                path=sequence_dir,
                num_frames=len(rows),
                frame_ids=frame_ids,
            )
        )

    sessions = tuple(
        SessionRecord(session_id=session_id, sequences=tuple(records))
        for session_id, records in sorted(grouped.items())
    )
    if not sessions:
        raise RuntimeError(f"no sequence directories discovered under {split_root}")
    assigned = [item.sequence for session in sessions for item in session.sequences]
    if len(assigned) != len(set(assigned)) or set(assigned) != seen_sequences:
        raise RuntimeError("session discovery did not assign every sequence exactly once")
    return sessions


def folds_from_sessions(sessions: tuple[SessionRecord, ...]) -> list[dict[str, object]]:
    session_ids = [session.session_id for session in sessions]
    folds: list[dict[str, object]] = []
    for held_out in session_ids:
        development = [session_id for session_id in session_ids if session_id != held_out]
        if held_out in development:
            raise RuntimeError("held-out session leaked into development sessions")
        folds.append({
            "fold": f"holdout_{held_out}",
            "held_out_session": held_out,
            "development_sessions": development,
        })
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    sessions = discover_sessions(args.data_root, args.split)
    result = {
        "split": args.split,
        "num_sessions": len(sessions),
        "num_sequences": sum(len(session.sequences) for session in sessions),
        "sessions": [session.to_json() for session in sessions],
        "folds": folds_from_sessions(sessions),
        "sanity": {
            "every_sequence_assigned_exactly_once": True,
            "development_holdout_disjoint": True,
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    for session in sessions:
        print(
            f"{session.session_id}: {len(session.sequences)} sequences, "
            f"{session.num_frames} frames"
        )
        for sequence in session.sequences:
            print(f"  {sequence.sequence}: {sequence.num_frames} frames")
    print(f"LOSO folds: {len(sessions)}")
    print(f"Saved: {args.json_out}")


if __name__ == "__main__":
    main()
