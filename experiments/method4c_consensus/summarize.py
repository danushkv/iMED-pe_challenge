#!/usr/bin/env python3
"""Session-balanced summary for every method in a Method 4C comparison."""

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text())
    report = {"split": comparison.get("split"), "methods": {}}

    print("Method | overall | session-balanced | worst | CV | frame p95")
    for method, payload in comparison["methods"].items():
        grouped: dict[str, list[float]] = defaultdict(list)
        for sequence, metrics in payload["per_sequence"].items():
            match = SESSION_PATTERN.match(sequence)
            if match is None:
                raise RuntimeError(f"Cannot parse physical session from {sequence}")
            grouped[match.group(1)].append(float(metrics["mean_ate"]))
        sessions = {
            session: {
                "num_sequences": len(values),
                "mean_ate_mm": float(np.mean(values)),
            }
            for session, values in sorted(grouped.items())
        }
        values = np.asarray(
            [item["mean_ate_mm"] for item in sessions.values()], dtype=np.float64
        )
        worst = max(sessions, key=lambda name: sessions[name]["mean_ate_mm"])
        mean = float(np.mean(values))
        std = float(np.std(values))
        result = {
            "prediction_root": payload.get("prediction_root"),
            "ordinary_mean_ate_mm": float(payload["global"]["mean_ate"]),
            "session_balanced_mean_ate_mm": mean,
            "worst_session": worst,
            "worst_session_ate_mm": float(sessions[worst]["mean_ate_mm"]),
            "std_across_sessions_mm": std,
            "coefficient_of_variation": std / mean if mean > 0 else float("nan"),
            "p95_aligned_position_error_mm": float(
                payload["global"]["p95_aligned_position_error_mm"]
            ),
            "sessions": sessions,
        }
        report["methods"][method] = result
        print(
            f"{method:18s} | {result['ordinary_mean_ate_mm']:.4f} | "
            f"{mean:.4f} | {result['worst_session_ate_mm']:.4f} ({worst}) | "
            f"{result['coefficient_of_variation']:.4f} | "
            f"{result['p95_aligned_position_error_mm']:.4f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

