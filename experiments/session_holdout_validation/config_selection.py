"""Leak-free development-session selection for future candidate predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidateSelection:
    output_name: str
    candidates: dict[str, Path]


def load_candidate_manifest(path: Path) -> CandidateSelection:
    payload = json.loads(path.read_text())
    if payload.get("objective", "session_balanced_mean_ate") != "session_balanced_mean_ate":
        raise ValueError("only session_balanced_mean_ate selection is supported")
    candidates = {
        str(name): Path(root) for name, root in payload.get("candidates", {}).items()
    }
    if len(candidates) < 2:
        raise ValueError("candidate manifest must contain at least two candidates")
    return CandidateSelection(
        output_name=str(payload["output_name"]),
        candidates=candidates,
    )


def select_on_development_sessions(
    *,
    candidate_session_metrics: dict[str, dict[str, dict[str, Any]]],
    development_sessions: list[str],
) -> tuple[str, dict[str, float]]:
    """Select without ever reading the held-out session's metrics."""
    scores: dict[str, float] = {}
    for candidate, session_metrics in candidate_session_metrics.items():
        values = [
            float(session_metrics[session_id]["aggregate"]["mean_ate"])
            for session_id in development_sessions
        ]
        scores[candidate] = float(np.mean(values))
    selected = min(scores, key=lambda name: (scores[name], name))
    return selected, scores
