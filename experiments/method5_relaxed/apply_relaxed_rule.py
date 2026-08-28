#!/usr/bin/env python3
"""Apply Method 5-R's global validity rule to unchanged Method-5 fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.method5_stereo_refinement.geometry import skew


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _load_runs(roots: list[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    sessions: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for root in roots:
        summary_path = root / "refinement_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing strict refinement summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        summaries.append(summary)
        for session_name in summary["sessions"]:
            if session_name in sessions:
                raise RuntimeError(f"Duplicate strict fit for {session_name}")
            diagnostics_path = root / "diagnostics" / f"{session_name}.json"
            if not diagnostics_path.is_file():
                raise FileNotFoundError(f"Missing strict diagnostics: {diagnostics_path}")
            sessions[session_name] = {
                "root": root,
                "strict_summary": summary["sessions"][session_name],
                "diagnostics": json.loads(diagnostics_path.read_text()),
                "config": summary["config"],
            }
    return sessions, summaries


def _relaxed_flags(payload: dict[str, Any]) -> dict[str, bool]:
    diagnostics = payload["diagnostics"]
    config = payload["config"]
    fit = diagnostics["full"]
    split = diagnostics["split_half"]
    cross = diagnostics["cross_sequence"]
    optimizer_converged = bool(fit["optimizer_success"])
    within_bounds = bool(
        fit["delta_rotation_deg"] <= config["max_rotation_correction_deg"]
        and fit["delta_translation_direction_deg"]
        <= config["max_translation_direction_correction_deg"]
    )
    split_half_stable = bool(
        len(split.get("fits", [])) == 2
        and all(item.get("optimizer_success", False) for item in split["fits"])
        and split["rotation_disagreement_deg"]
        <= config["split_rotation_stability_deg"]
        and split["translation_direction_disagreement_deg"]
        <= config["split_translation_stability_deg"]
    )
    cross_sequence_stable = bool(
        cross.get("num_fits", 0) < 2
        or (
            cross["rotation_disagreement_p90_deg"]
            <= config["sequence_rotation_stability_p90_deg"]
            and cross["translation_direction_disagreement_p90_deg"]
            <= config["sequence_translation_stability_p90_deg"]
        )
    )
    objective_improved = bool(
        np.isfinite(fit["initial_cost"])
        and np.isfinite(fit["final_cost"])
        and fit["final_cost"] < fit["initial_cost"]
    )
    return {
        "optimizer_converged": optimizer_converged,
        "within_bounds": within_bounds,
        "split_half_stable": split_half_stable,
        "cross_sequence_stable": cross_sequence_stable,
        "total_objective_improved": objective_improved,
        "relaxed_valid": all(
            (
                optimizer_converged,
                within_bounds,
                split_half_stable,
                cross_sequence_stable,
                objective_improved,
            )
        ),
    }


def _export(
    original_path: Path,
    destination: Path,
    diagnostics: dict[str, Any],
    relaxed_valid: bool,
    relaxed_record: dict[str, Any],
) -> None:
    with np.load(original_path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    if relaxed_valid:
        R = np.asarray(diagnostics["full"]["R_LR"], dtype=np.float64)
        t = np.asarray(diagnostics["full"]["t_LR"], dtype=np.float64).reshape(3)
    else:
        R = np.asarray(payload["R_LR"], dtype=np.float64)
        t = np.asarray(payload["t_LR_unit"], dtype=np.float64).reshape(3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R
    transform[:3, 3] = t
    payload["R_LR"] = R
    payload["t_LR_unit"] = t
    payload["T_R_from_L"] = transform
    payload["E"] = skew(t) @ R
    original_diagnostics: dict[str, Any] = {}
    if "diagnostic_statistics_json" in payload:
        try:
            original_diagnostics = json.loads(
                str(payload["diagnostic_statistics_json"].item())
            )
        except (json.JSONDecodeError, ValueError):
            original_diagnostics = {"unparsed_original_diagnostics": True}
    original_diagnostics["method5_relaxed"] = relaxed_record
    payload["diagnostic_statistics_json"] = np.array(
        json.dumps(original_diagnostics, sort_keys=True, default=_json_default)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-run",
        type=Path,
        action="append",
        required=True,
        help="Directory containing refinement_summary.json and diagnostics/",
    )
    parser.add_argument("--original-calibration-root", type=Path, required=True)
    parser.add_argument("--require-session", action="append", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sessions, source_summaries = _load_runs(args.strict_run)
    required = (
        {f"session_{str(value).removeprefix('session_').zfill(3)}" for value in args.require_session}
        if args.require_session
        else set()
    )
    missing = sorted(required - set(sessions))
    if missing:
        raise RuntimeError(f"Missing required session fits: {missing}")
    output_calibration = args.output_root / "calibration"
    output_diagnostics = args.output_root / "diagnostics"
    output_diagnostics.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "method": "method5_relaxed_global_validity",
        "rule": {
            "optimizer_converged": True,
            "inside_existing_hard_bounds": True,
            "existing_split_half_stability": True,
            "existing_cross_sequence_stability_where_available": True,
            "total_robust_objective_improved": True,
            "per_view_reprojection_veto_used": False,
            "ground_truth_used": False,
        },
        "strict_runs": [str(path) for path in args.strict_run],
        "sessions": {},
    }
    for session_name, payload in sorted(sessions.items()):
        session = session_name.removeprefix("session_")
        diagnostics = payload["diagnostics"]
        fit = diagnostics["full"]
        flags = _relaxed_flags(payload)
        initial_cost = float(fit["initial_cost"])
        final_cost = float(fit["final_cost"])
        record = {
            "session": session_name,
            **flags,
            "strict_valid": bool(diagnostics["correction_valid"]),
            "strict_validity_failures": diagnostics["validity_failures"],
            "delta_rotation_deg": float(fit["delta_rotation_deg"]),
            "delta_translation_direction_deg": float(
                fit["delta_translation_direction_deg"]
            ),
            "objective_before": initial_cost,
            "objective_after": final_cost,
            "relative_objective_improvement": (
                (initial_cost - final_cost) / max(abs(initial_cost), 1e-12)
            ),
            "all_view_reprojection_before": diagnostics["original_reprojection"]["all_views"],
            "all_view_reprojection_after": fit["reprojection"]["all_views"],
            "per_view_reprojection_before": {
                name: diagnostics["original_reprojection"][name]
                for name in ("e1_left", "e1_right", "e2_left")
            },
            "per_view_reprojection_after": {
                name: fit["reprojection"][name]
                for name in ("e1_left", "e1_right", "e2_left")
            },
            "split_half": {
                "rotation_disagreement_deg": diagnostics["split_half"]["rotation_disagreement_deg"],
                "translation_direction_disagreement_deg": diagnostics["split_half"]["translation_direction_disagreement_deg"],
            },
            "cross_sequence": {
                "num_fits": diagnostics["cross_sequence"]["num_fits"],
                "rotation_disagreement_p90_deg": diagnostics["cross_sequence"]["rotation_disagreement_p90_deg"],
                "translation_direction_disagreement_p90_deg": diagnostics["cross_sequence"]["translation_direction_disagreement_p90_deg"],
            },
            "source_strict_run": str(payload["root"]),
        }
        _export(
            args.original_calibration_root / f"session_{session}.npz",
            output_calibration / f"session_{session}.npz",
            diagnostics,
            flags["relaxed_valid"],
            record,
        )
        (output_diagnostics / f"session_{session}.json").write_text(
            json.dumps(record, indent=2, default=_json_default) + "\n"
        )
        report["sessions"][session_name] = record
        print(
            f"{session_name}: strict={record['strict_valid']} "
            f"relaxed={record['relaxed_valid']} "
            f"objective={initial_cost:.6f}->{final_cost:.6f}",
            flush=True,
        )
    report["counts"] = {
        "num_sessions": len(sessions),
        "strict_valid": sum(item["strict_valid"] for item in report["sessions"].values()),
        "relaxed_valid": sum(item["relaxed_valid"] for item in report["sessions"].values()),
        "strict_rejected_relaxed_accepted": sum(
            (not item["strict_valid"]) and item["relaxed_valid"]
            for item in report["sessions"].values()
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "relaxed_summary.json"
    output.write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
    print(f"Saved Method 5-R summary: {output}")
    print(f"Saved selected calibrations: {output_calibration}")


if __name__ == "__main__":
    main()

