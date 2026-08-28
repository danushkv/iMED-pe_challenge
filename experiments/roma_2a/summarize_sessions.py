#!/usr/bin/env python3
"""Session-balanced aggregation of an official-metric comparison JSON."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


SESSION_PATTERN = re.compile(r"^(session_\d+)_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison = json.loads(args.comparison.read_text())
    methods = comparison.get("methods", {})
    if args.method not in methods:
        raise KeyError(
            f"Method {args.method!r} not present; available={sorted(methods)}"
        )
    method = methods[args.method]
    per_sequence = method.get("per_sequence", {})
    if not per_sequence:
        raise RuntimeError(f"No per-sequence metrics for {args.method}")

    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for sequence_name, metrics in sorted(per_sequence.items()):
        match = SESSION_PATTERN.match(sequence_name)
        if match is None:
            raise RuntimeError(f"Cannot parse session from {sequence_name!r}")
        ate = float(metrics["mean_ate"])
        if not np.isfinite(ate):
            raise RuntimeError(f"Non-finite ATE for {sequence_name}: {ate}")
        grouped[match.group(1)].append((sequence_name, ate))

    sessions: dict[str, object] = {}
    session_means: list[float] = []
    for session_name, rows in sorted(grouped.items()):
        values = np.asarray([value for _, value in rows], dtype=np.float64)
        mean_ate = float(np.mean(values))
        session_means.append(mean_ate)
        sessions[session_name] = {
            "num_sequences": len(rows),
            "mean_ate_mm": mean_ate,
            "median_ate_mm": float(np.median(values)),
            "std_ate_mm": float(np.std(values)),
            "sequences": {
                sequence_name: {"mean_ate_mm": ate}
                for sequence_name, ate in rows
            },
        }

    session_array = np.asarray(session_means, dtype=np.float64)
    worst_name = max(
        sessions,
        key=lambda name: float(sessions[name]["mean_ate_mm"]),
    )
    best_name = min(
        sessions,
        key=lambda name: float(sessions[name]["mean_ate_mm"]),
    )
    balanced_mean = float(np.mean(session_array))
    session_std = float(np.std(session_array))
    report = {
        "split": comparison.get("split"),
        "method": args.method,
        "prediction_root": method.get("prediction_root"),
        "num_sessions": len(sessions),
        "num_sequences": len(per_sequence),
        "ordinary_sequence_weighted_mean_ate_mm": float(
            method.get("global", {}).get("mean_ate", np.nan)
        ),
        "session_balanced_mean_ate_mm": balanced_mean,
        "median_session_mean_ate_mm": float(np.median(session_array)),
        "std_across_session_means_mm": session_std,
        "coefficient_of_variation_across_sessions": (
            session_std / balanced_mean if balanced_mean > 0 else float("nan")
        ),
        "worst_session": {
            "session": worst_name,
            "mean_ate_mm": sessions[worst_name]["mean_ate_mm"],
        },
        "best_session": {
            "session": best_name,
            "mean_ate_mm": sessions[best_name]["mean_ate_mm"],
        },
        "sessions": sessions,
    }

    print(
        f"{args.method}: overall={report['ordinary_sequence_weighted_mean_ate_mm']:.4f} "
        f"session-balanced={balanced_mean:.4f} "
        f"worst={report['worst_session']['mean_ate_mm']:.4f} "
        f"({worst_name}) CV={report['coefficient_of_variation_across_sessions']:.4f}"
    )
    print("Session | sequences | mean ATE")
    for session_name, payload in sessions.items():
        print(
            f"{session_name:12s} | {payload['num_sequences']:9d} | "
            f"{payload['mean_ate_mm']:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
