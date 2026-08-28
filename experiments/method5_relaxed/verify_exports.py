#!/usr/bin/env python3
"""Verify Method 5-R selected calibrations and fallback behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--selected-root", type=Path, required=True)
    parser.add_argument("--relaxed-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.relaxed_summary.read_text())
    required = {
        "session_id", "K_left", "K_right", "E", "R_LR", "t_LR_unit",
        "T_R_from_L", "suggested_min_depth", "suggested_max_depth",
        "diagnostic_statistics_json",
    }
    report: dict[str, object] = {"passed": True, "sessions": {}}
    for session_name, decision in sorted(summary["sessions"].items()):
        session = session_name.removeprefix("session_")
        original_path = args.original_root / f"session_{session}.npz"
        selected_path = args.selected_root / f"session_{session}.npz"
        with np.load(original_path, allow_pickle=False) as original, np.load(
            selected_path, allow_pickle=False
        ) as selected:
            missing = sorted(required - set(selected.files))
            R = np.asarray(selected["R_LR"], dtype=np.float64)
            t = np.asarray(selected["t_LR_unit"], dtype=np.float64).reshape(3)
            T = np.asarray(selected["T_R_from_L"], dtype=np.float64)
            det_error = abs(float(np.linalg.det(R)) - 1.0)
            orthogonality_error = float(np.max(np.abs(R.T @ R - np.eye(3))))
            baseline_error = abs(
                float(np.linalg.norm(t))
                - float(np.linalg.norm(np.asarray(original["t_LR_unit"])))
            )
            transform_error = max(
                float(np.max(np.abs(T[:3, :3] - R))),
                float(np.max(np.abs(T[:3, 3] - t))),
            )
            if decision["relaxed_valid"]:
                strict_diagnostics_path = (
                    Path(decision["source_strict_run"])
                    / "diagnostics"
                    / f"session_{session}.json"
                )
                strict = json.loads(strict_diagnostics_path.read_text())
                expected_R = np.asarray(strict["full"]["R_LR"], dtype=np.float64)
                expected_t = np.asarray(strict["full"]["t_LR"], dtype=np.float64)
                selection_error = max(
                    float(np.max(np.abs(R - expected_R))),
                    float(np.max(np.abs(t - expected_t))),
                )
                changed_from_original = max(
                    float(np.max(np.abs(R - np.asarray(original["R_LR"])))),
                    float(
                        np.max(
                            np.abs(t - np.asarray(original["t_LR_unit"]).reshape(3))
                        )
                    ),
                )
                selection_ok = selection_error < 1e-12 and changed_from_original > 1e-12
            else:
                selection_error = max(
                    float(np.max(np.abs(R - np.asarray(original["R_LR"])))),
                    float(
                        np.max(
                            np.abs(t - np.asarray(original["t_LR_unit"]).reshape(3))
                        )
                    ),
                )
                changed_from_original = selection_error
                selection_ok = selection_error < 1e-12
            valid = bool(
                not missing
                and det_error < 1e-8
                and orthogonality_error < 1e-8
                and baseline_error < 1e-10
                and transform_error < 1e-10
                and selection_ok
            )
        report["passed"] = bool(report["passed"] and valid)
        report["sessions"][session_name] = {
            "passed": valid,
            "relaxed_valid": decision["relaxed_valid"],
            "missing_keys": missing,
            "rotation_determinant_error": det_error,
            "rotation_orthogonality_error": orthogonality_error,
            "baseline_norm_error": baseline_error,
            "transform_consistency_error": transform_error,
            "change_from_original": changed_from_original,
            "fallback_or_selection_error": selection_error,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
