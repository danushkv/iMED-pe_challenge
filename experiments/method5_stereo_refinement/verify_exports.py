#!/usr/bin/env python3
"""Verify exported calibrations preserve schema and valid SE(3) geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--refinement-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.refinement_summary.read_text())
    sessions: dict[str, object] = {}
    passed = True
    required = {
        "session_id", "K_left", "K_right", "E", "R_LR", "t_LR_unit",
        "T_R_from_L", "suggested_min_depth", "suggested_max_depth",
        "diagnostic_statistics_json",
    }
    for name, item in sorted(summary["sessions"].items()):
        session = name.removeprefix("session_")
        original_path = args.original_root / f"session_{session}.npz"
        candidate_path = args.candidate_root / f"session_{session}.npz"
        with np.load(original_path, allow_pickle=False) as original, np.load(
            candidate_path, allow_pickle=False
        ) as candidate:
            missing = sorted(required - set(candidate.files))
            R = np.asarray(candidate["R_LR"], dtype=np.float64)
            t = np.asarray(candidate["t_LR_unit"], dtype=np.float64).reshape(3)
            T = np.asarray(candidate["T_R_from_L"], dtype=np.float64)
            det_error = abs(float(np.linalg.det(R)) - 1.0)
            orthogonality_error = float(np.max(np.abs(R.T @ R - np.eye(3))))
            baseline_error = abs(
                float(np.linalg.norm(t))
                - float(np.linalg.norm(np.asarray(original["t_LR_unit"])))
            )
            transform_error = float(
                np.max(np.abs(T[:3, :3] - R))
                + np.max(np.abs(T[:3, 3] - t))
            )
            invalid_unchanged_error = 0.0
            if not item["correction_valid"]:
                invalid_unchanged_error = max(
                    float(np.max(np.abs(R - np.asarray(original["R_LR"])))),
                    float(np.max(np.abs(t - np.asarray(original["t_LR_unit"]).reshape(3)))),
                )
        valid = (
            not missing
            and det_error < 1e-8
            and orthogonality_error < 1e-8
            and baseline_error < 1e-10
            and transform_error < 1e-10
            and invalid_unchanged_error < 1e-12
        )
        passed &= valid
        sessions[name] = {
            "passed": valid,
            "missing_keys": missing,
            "rotation_determinant_error": det_error,
            "rotation_orthogonality_error": orthogonality_error,
            "baseline_norm_error": baseline_error,
            "transform_consistency_error": transform_error,
            "invalid_export_unchanged_error": invalid_unchanged_error,
        }
    report = {"passed": passed, "sessions": sessions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

