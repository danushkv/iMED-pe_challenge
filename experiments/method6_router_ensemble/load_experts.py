"""Load saved expert predictions and diagnostics without opening ground truth."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.method4c_consensus.load_observers import (
    ObserverBundle,
    load_observer_bundle,
    parse_named_root,
)


SESSION_PATTERN = re.compile(r"^session_(\d+)(?:_|$)")


def session_id(sequence: str) -> str:
    match = SESSION_PATTERN.match(sequence)
    if match is None:
        raise ValueError(f"Cannot discover physical session from {sequence!r}")
    return match.group(1)


def discover_sequences(reference_root: Path, split: str) -> list[str]:
    root = reference_root / split
    if not root.is_dir():
        raise FileNotFoundError(root)
    sequences = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "pose.txt").is_file()
    )
    if not sequences:
        raise RuntimeError(f"No saved predictions under {root}")
    return sequences


def load_bundle(
    *,
    split: str,
    sequence: str,
    expert_roots: dict[str, Path],
    method1_root: Path,
    rotation_root: Path,
) -> ObserverBundle:
    expected = {"2a", "2b", "loftr"}
    if set(expert_roots) != expected:
        raise ValueError(f"Expected expert roots {sorted(expected)}")
    return load_observer_bundle(
        split=split,
        sequence=sequence,
        observer_roots=expert_roots,
        reference_name="2b",
        method1_root=method1_root,
        rotation_root=rotation_root,
    )


def load_calibration_diagnostics(root: Path, session: str) -> dict[str, Any]:
    path = root / f"session_{session}_diagnostics.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    if str(payload.get("session_id", session)).zfill(len(session)) != session:
        raise RuntimeError(f"Session mismatch in {path}")
    return payload


__all__ = [
    "ObserverBundle",
    "discover_sequences",
    "load_bundle",
    "load_calibration_diagnostics",
    "parse_named_root",
    "session_id",
]

