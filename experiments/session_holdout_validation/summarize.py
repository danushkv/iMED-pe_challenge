"""Aggregate and serialize LOSO metrics without changing predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_fold import aggregate_sequence_metrics


def summarize_loso(
    *,
    folds: dict[str, dict[str, Any]],
    method_names: list[str],
    baseline_name: str,
) -> dict[str, Any]:
    session_table: dict[str, dict[str, Any]] = {}
    for session_id, fold in folds.items():
        session_table[session_id] = {
            method: fold["methods"][method]["aggregate"] for method in method_names
        }

    method_summary: dict[str, dict[str, Any]] = {}
    for method in method_names:
        session_ates = {
            session_id: float(values[method]["mean_ate"])
            for session_id, values in session_table.items()
        }
        ate_values = np.asarray(list(session_ates.values()), dtype=np.float64)
        all_sequences = {
            sequence: metrics
            for fold in folds.values()
            for sequence, metrics in fold["methods"][method]["per_sequence"].items()
        }
        sequence_aggregate = aggregate_sequence_metrics(all_sequences)
        summary: dict[str, Any] = {
            "session_balanced_mean_ate": float(np.mean(ate_values)),
            "median_session_ate": float(np.median(ate_values)),
            "std_across_sessions": float(np.std(ate_values)),
            "worst_session_ate": float(np.max(ate_values)),
            "worst_session": max(session_ates, key=session_ates.get),
            "best_session_ate": float(np.min(ate_values)),
            "best_session": min(session_ates, key=session_ates.get),
            "coefficient_of_variation": float(np.std(ate_values) / np.mean(ate_values)),
            "sequence_weighted_mean_ate": float(sequence_aggregate["mean_ate"]),
            "frame_weighted_mean_ate": float(sequence_aggregate["frame_weighted_mean_ate"]),
            "session_ate": session_ates,
        }
        if method != baseline_name and baseline_name in method_names:
            baseline = {
                session_id: float(values[baseline_name]["mean_ate"])
                for session_id, values in session_table.items()
            }
            deltas = {session_id: session_ates[session_id] - baseline[session_id] for session_id in session_ates}
            regressions = {
                session_id: value for session_id, value in deltas.items() if value > 0
            }
            improvements = {
                session_id: -value for session_id, value in deltas.items() if value < 0
            }
            summary.update({
                "sessions_improved_vs_baseline": len(improvements),
                "sessions_worsened_vs_baseline": len(regressions),
                "maximum_regression_vs_baseline": (
                    float(max(regressions.values())) if regressions else 0.0
                ),
                "maximum_regression_session": (
                    max(regressions, key=regressions.get) if regressions else None
                ),
                "maximum_improvement_vs_baseline": (
                    float(max(improvements.values())) if improvements else 0.0
                ),
                "maximum_improvement_session": (
                    max(improvements, key=improvements.get) if improvements else None
                ),
                "delta_vs_baseline_by_session": deltas,
            })
        method_summary[method] = summary

    released_ranking = sorted(
        method_names, key=lambda name: method_summary[name]["sequence_weighted_mean_ate"]
    )
    loso_mean_ranking = sorted(
        method_names, key=lambda name: method_summary[name]["session_balanced_mean_ate"]
    )
    loso_worst_ranking = sorted(
        method_names, key=lambda name: method_summary[name]["worst_session_ate"]
    )
    return {
        "session_table": session_table,
        "methods": method_summary,
        "rankings": {
            "released_overall_ate": released_ranking,
            "loso_session_balanced_mean_ate": loso_mean_ranking,
            "loso_worst_session_ate": loso_worst_ranking,
        },
    }


def hidden_context(
    *,
    hidden_scores: dict[str, float],
    summary: dict[str, Any],
) -> dict[str, Any]:
    available = [name for name in hidden_scores if name in summary["methods"]]
    hidden_ranking = sorted(available, key=hidden_scores.get)

    def inversions(ranking: list[str]) -> int:
        filtered = [name for name in ranking if name in available]
        hidden_position = {name: index for index, name in enumerate(hidden_ranking)}
        count = 0
        for left in range(len(filtered)):
            for right in range(left + 1, len(filtered)):
                if hidden_position[filtered[left]] > hidden_position[filtered[right]]:
                    count += 1
        return count

    rankings = summary["rankings"]
    return {
        "note": "External context only; never used for selection or thresholds.",
        "scores_mm": hidden_scores,
        "hidden_ranking": hidden_ranking,
        "released_ranking_pairwise_disagreements": inversions(
            rankings["released_overall_ate"]
        ),
        "loso_mean_ranking_pairwise_disagreements": inversions(
            rankings["loso_session_balanced_mean_ate"]
        ),
        "loso_worst_ranking_pairwise_disagreements": inversions(
            rankings["loso_worst_session_ate"]
        ),
    }


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}" if np.isfinite(float(value)) else "nan"
    except (TypeError, ValueError):
        return "nan"


def write_tables_and_report(result: dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    methods = list(summary["methods"])
    sessions = list(summary["session_table"])

    with (output_root / "session_method_ate.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["session", *methods])
        for session_id in sessions:
            writer.writerow([
                session_id,
                *[summary["session_table"][session_id][name]["mean_ate"] for name in methods],
            ])

    with (output_root / "per_sequence_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "session", "sequence", "method", "mean_ate", "median_ate", "std_ate",
            "trans_rpe_1", "rot_rpe_1", "trans_rpe_10", "rot_rpe_10",
            "trans_rpe_20", "rot_rpe_20", "trans_rpe_40", "rot_rpe_40",
        ])
        for session_id, fold in result["folds"].items():
            for method, method_result in fold["methods"].items():
                for sequence, metrics in method_result["per_sequence"].items():
                    writer.writerow([
                        session_id, sequence, method,
                        metrics["mean_ate"], metrics["median_ate"], metrics["std_ate"],
                        metrics["trans_rpe_1frame"], metrics["rot_rpe_deg_1frame"],
                        metrics["trans_rpe_10frame"], metrics["rot_rpe_deg_10frame"],
                        metrics["trans_rpe_20frame"], metrics["rot_rpe_deg_20frame"],
                        metrics["trans_rpe_40frame"], metrics["rot_rpe_deg_40frame"],
                    ])

    calibration = result.get("calibration_analysis", {}).get("session_table", {})
    if calibration:
        with (output_root / "calibration_session_table.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "session",
                "e1_rotation_p90_deg", "e1_translation_direction_p90_deg",
                "e1_reprojection_error_median_px", "e1_valid", "method2a_ate_mm",
                "e2_rotation_p90_deg", "e2_translation_direction_p90_deg",
                "e2_reprojection_error_median_px", "e2_valid", "method2b_ate_mm",
            ])
            for session_id, row in calibration.items():
                e1 = row.get("e1_stereo", {})
                e2 = row.get("e2_stereo", {})
                writer.writerow([
                    session_id,
                    e1.get("rotation_p90_deg", np.nan),
                    e1.get("translation_direction_p90_deg", np.nan),
                    e1.get("reprojection_error_median_px", np.nan),
                    e1.get("valid"),
                    row.get("method2a_ate_mm", np.nan),
                    e2.get("rotation_p90_deg", np.nan),
                    e2.get("translation_direction_p90_deg", np.nan),
                    e2.get("reprojection_error_median_px", np.nan),
                    e2.get("valid"),
                    row.get("method2b_ate_mm", np.nan),
                ])

    lines = ["# Leave-one-session-out validation", "", f"Split: `{result['split']}`", ""]
    lines.extend(["## Sessions", ""])
    for session in result["discovery"]["sessions"]:
        lines.append(
            f"- `{session['session_id']}`: {session['num_sequences']} sequences, "
            f"{session['num_frames']} frames"
        )

    lines.extend([
        "", "## Robustness summary", "",
        "| Method | Released overall | LOSO mean | LOSO median | LOSO std | Worst session | Best session | CV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method, values in summary["methods"].items():
        lines.append(
            f"| {method} | {_fmt(values['sequence_weighted_mean_ate'])} | "
            f"{_fmt(values['session_balanced_mean_ate'])} | {_fmt(values['median_session_ate'])} | "
            f"{_fmt(values['std_across_sessions'])} | {_fmt(values['worst_session_ate'])} "
            f"({values['worst_session']}) | {_fmt(values['best_session_ate'])} "
            f"({values['best_session']}) | {_fmt(values['coefficient_of_variation'])} |"
        )

    lines.extend(["", "## Robustness versus baseline", ""])
    for method, values in summary["methods"].items():
        if method == result.get("baseline_name", "baseline"):
            continue
        if "sessions_improved_vs_baseline" not in values:
            continue
        lines.append(
            f"- {method}: improved {values['sessions_improved_vs_baseline']} sessions, "
            f"worsened {values['sessions_worsened_vs_baseline']}; maximum improvement "
            f"{_fmt(values['maximum_improvement_vs_baseline'])} mm on "
            f"{values['maximum_improvement_session']}; maximum regression "
            f"{_fmt(values['maximum_regression_vs_baseline'])} mm on "
            f"{values['maximum_regression_session']}."
        )

    lines.extend(["", "## Session x method mean ATE", ""])
    lines.append("| Session | " + " | ".join(methods) + " |")
    lines.append("|---|" + "---:|" * len(methods))
    for session_id in sessions:
        lines.append(
            f"| {session_id} | " + " | ".join(
                _fmt(summary["session_table"][session_id][name]["mean_ate"])
                for name in methods
            ) + " |"
        )

    lines.extend(["", "## Per-fold official metrics", ""])
    for session_id in sessions:
        lines.extend([
            f"### Held out: {session_id}", "",
            "| Method | Mean ATE | Median ATE | ATE std | tRPE1 | rRPE1 | tRPE10 | rRPE10 | tRPE20 | rRPE20 | tRPE40 | rRPE40 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for method in methods:
            values = summary["session_table"][session_id][method]
            lines.append(
                f"| {method} | {_fmt(values['mean_ate'])} | {_fmt(values['median_ate'])} | "
                f"{_fmt(values['std_ate'])} | {_fmt(values['trans_rpe_1frame'])} | "
                f"{_fmt(values['rot_rpe_deg_1frame'])} | {_fmt(values['trans_rpe_10frame'])} | "
                f"{_fmt(values['rot_rpe_deg_10frame'])} | {_fmt(values['trans_rpe_20frame'])} | "
                f"{_fmt(values['rot_rpe_deg_20frame'])} | {_fmt(values['trans_rpe_40frame'])} | "
                f"{_fmt(values['rot_rpe_deg_40frame'])} |"
            )
        lines.append("")

    lines.extend(["", "## Rankings", ""])
    for label, ranking in summary["rankings"].items():
        lines.append(f"- {label}: " + " < ".join(ranking))

    if result.get("calibration_analysis"):
        lines.extend(["", "## Calibration correlations", ""])
        for name, values in result["calibration_analysis"]["correlations"].items():
            lines.append(
                f"- {name}: r={_fmt(values['pearson_r'])}, n={values['num_sessions']}"
            )
        lines.extend(["", "Calibration values and Method-2 session ATEs are saved in `calibration_session_table.csv`."])
    if result.get("hidden_external_context"):
        hidden = result["hidden_external_context"]
        lines.extend(["", "## Hidden leaderboard — external context only", ""])
        for method, value in hidden["scores_mm"].items():
            lines.append(f"- {method}: {value:.3f} mm")
        lines.append(
            "- Pairwise ranking disagreements with hidden: released="
            f"{hidden['released_ranking_pairwise_disagreements']}, "
            f"LOSO mean={hidden['loso_mean_ranking_pairwise_disagreements']}, "
            f"LOSO worst={hidden['loso_worst_ranking_pairwise_disagreements']}"
        )

    lines.extend([
        "", "## Leakage policy", "",
        "Held-out pose.txt files are opened only by evaluation after predictions and any development-only candidate selection are fixed. Session-specific image-only calibration remains a valid inference-time operation.",
        "",
    ])
    (output_root / "SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate LOSO tables from results JSON.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    write_tables_and_report(json.loads(args.results.read_text()), args.output_root)


if __name__ == "__main__":
    main()
