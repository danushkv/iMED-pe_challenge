#!/usr/bin/env python3
"""Run leave-one-session-out evaluation on existing prediction trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from imcpe.pose_metrics import RPE_DELTAS, evaluate_split

from calibration_analysis import calibration_correlations, load_calibration_table
from config_selection import load_candidate_manifest, select_on_development_sessions
from discover_sessions import discover_sessions, folds_from_sessions
from evaluate_fold import aggregate_sequence_metrics, evaluate_session, validate_prediction_coverage
from plots import plot_calibration_analysis, plot_session_ate
from summarize import hidden_context, summarize_loso, write_tables_and_report


def _named_path(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError("expected NAME=PATH")
    name, path = specification.split("=", 1)
    if not name or not path:
        raise ValueError("expected NAME=PATH")
    return name, Path(path)


def _hidden_score(specification: str) -> tuple[str, float]:
    name, value = specification.split("=", 1)
    return name, float(value)


def _ordinary_sanity(
    *,
    data_root: Path,
    split: str,
    prediction_roots: dict[str, Path],
    evaluated: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    keys = ["mean_ate", "median_ate", "std_ate"] + [
        key
        for delta in RPE_DELTAS
        for key in (f"trans_rpe_{delta}frame", f"rot_rpe_deg_{delta}frame")
    ]
    for method, root in prediction_roots.items():
        _, official = evaluate_split(data_root, root, split)
        per_sequence = {
            sequence: metrics
            for session_metrics in evaluated[method].values()
            for sequence, metrics in session_metrics["per_sequence"].items()
        }
        grouped = aggregate_sequence_metrics(per_sequence)
        differences = {
            key: abs(float(official[key]) - float(grouped[key])) for key in keys
        }
        result[method] = {
            "official_metrics": official,
            "session_grouped_then_flattened_metrics": grouped,
            "maximum_absolute_metric_difference": float(max(differences.values())),
            "passed": bool(max(differences.values()) <= 1.0e-10),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument(
        "--prediction", action="append", required=True,
        help="NAME=prediction/root; repeat for each fixed method",
    )
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument(
        "--candidate-manifest", type=Path, action="append", default=[],
        help="Optional future development-only configuration-selection manifest",
    )
    parser.add_argument("--e1-calibration-root", type=Path)
    parser.add_argument("--e2-calibration-root", type=Path)
    parser.add_argument("--calibration-analysis-json", type=Path)
    parser.add_argument("--method2a-name", default="method2a")
    parser.add_argument("--method2b-name", default="method2b")
    parser.add_argument(
        "--hidden-score", action="append", default=[],
        help="NAME=MM external context only; never used by selection",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    prediction_roots = dict(_named_path(item) for item in args.prediction)
    if len(prediction_roots) != len(args.prediction):
        raise ValueError("prediction names must be unique")
    if args.baseline_name not in prediction_roots:
        raise ValueError(f"baseline method {args.baseline_name!r} was not supplied")

    sessions = discover_sessions(args.data_root, args.split)
    session_by_id = {session.session_id: session for session in sessions}
    fold_definitions = folds_from_sessions(sessions)
    coverage = {
        method: validate_prediction_coverage(
            prediction_root=root, split=args.split, sessions=sessions
        )
        for method, root in prediction_roots.items()
    }

    evaluated: dict[str, dict[str, dict[str, Any]]] = {
        method: {
            session.session_id: evaluate_session(
                session=session, prediction_root=root, split=args.split
            )
            for session in sessions
        }
        for method, root in prediction_roots.items()
    }

    folds: dict[str, dict[str, Any]] = {}
    for definition in fold_definitions:
        held_out = str(definition["held_out_session"])
        development = list(definition["development_sessions"])
        if held_out in development or set(development) & {held_out}:
            raise RuntimeError("development/held-out leakage")
        folds[held_out] = {
            **definition,
            "methods": {
                method: evaluated[method][held_out] for method in prediction_roots
            },
            "selected_configurations": {},
        }

    selected_roots: dict[str, dict[str, Path]] = {}
    for manifest_path in args.candidate_manifest:
        manifest = load_candidate_manifest(manifest_path)
        if manifest.output_name in prediction_roots or manifest.output_name in selected_roots:
            raise ValueError(f"duplicate output method name: {manifest.output_name}")
        for candidate_root in manifest.candidates.values():
            validate_prediction_coverage(
                prediction_root=candidate_root, split=args.split, sessions=sessions
            )
        candidate_metrics = {
            candidate: {
                session.session_id: evaluate_session(
                    session=session, prediction_root=root, split=args.split
                )
                for session in sessions
            }
            for candidate, root in manifest.candidates.items()
        }
        selected_roots[manifest.output_name] = manifest.candidates
        for held_out, fold in folds.items():
            # The selector receives a view containing development sessions only;
            # held-out metrics are not even present in its input mapping.
            development_only = {
                candidate: {
                    session_id: metrics
                    for session_id, metrics in session_metrics.items()
                    if session_id in fold["development_sessions"]
                }
                for candidate, session_metrics in candidate_metrics.items()
            }
            selected, development_scores = select_on_development_sessions(
                candidate_session_metrics=development_only,
                development_sessions=fold["development_sessions"],
            )
            fold["methods"][manifest.output_name] = candidate_metrics[selected][held_out]
            fold["selected_configurations"][manifest.output_name] = {
                "selected_candidate": selected,
                "development_scores": development_scores,
                "held_out_session_not_used_for_selection": True,
            }

    method_names = list(prediction_roots) + list(selected_roots)
    summary = summarize_loso(
        folds=folds, method_names=method_names, baseline_name=args.baseline_name
    )
    calibration_table = load_calibration_table(
        session_ids=list(session_by_id),
        session_method_metrics=summary["session_table"],
        e1_root=args.e1_calibration_root,
        e2_root=args.e2_calibration_root,
        analysis_json=args.calibration_analysis_json,
        method2a_name=args.method2a_name,
        method2b_name=args.method2b_name,
    )
    calibration = {
        "session_table": calibration_table,
        "correlations": calibration_correlations(calibration_table),
        "note": "Diagnostic only; no thresholds or configurations are fitted.",
    }
    hidden_scores = dict(_hidden_score(item) for item in args.hidden_score)
    discovery = {
        "num_sessions": len(sessions),
        "num_folds": len(folds),
        "sessions": [session.to_json() for session in sessions],
        "folds": fold_definitions,
    }
    result: dict[str, Any] = {
        "split": args.split,
        "baseline_name": args.baseline_name,
        "discovery": discovery,
        "prediction_roots": {name: str(root) for name, root in prediction_roots.items()},
        "folds": folds,
        "summary": summary,
        "calibration_analysis": calibration,
        "hidden_external_context": hidden_context(
            hidden_scores=hidden_scores, summary=summary
        ) if hidden_scores else None,
        "sanity": {
            "every_sequence_belongs_to_exactly_one_session": True,
            "development_and_held_out_disjoint": True,
            "held_out_ground_truth_evaluation_only": True,
            "session_specific_image_only_calibration_allowed": True,
            "prediction_coverage": coverage,
            "official_overall_reproduction": _ordinary_sanity(
                data_root=args.data_root,
                split=args.split,
                prediction_roots=prediction_roots,
                evaluated=evaluated,
            ),
        },
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    result_path = args.output_root / "loso_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    (args.output_root / "folds.json").write_text(json.dumps(discovery, indent=2) + "\n")
    write_tables_and_report(result, args.output_root)
    plot_session_ate(summary, args.output_root / "plots")
    plot_calibration_analysis(calibration, args.output_root / "plots")

    print(f"Discovered {len(sessions)} sessions and {len(folds)} LOSO folds")
    print("Method | released overall | LOSO mean | LOSO worst")
    for method, metrics in summary["methods"].items():
        print(
            f"{method:12s} | {metrics['sequence_weighted_mean_ate']:.4f} | "
            f"{metrics['session_balanced_mean_ate']:.4f} | "
            f"{metrics['worst_session_ate']:.4f} ({metrics['worst_session']})"
        )
    if hidden_scores:
        print("Hidden leaderboard scores (external context only):")
        for method, score in hidden_scores.items():
            print(f"  {method}: {score:.3f} mm")
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
