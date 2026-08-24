#!/usr/bin/env python3
"""Create the final Method 4A report from already-saved evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


METHOD_ORDER = (
    "baseline", "method1", "method1_5", "method2a", "method2b", "method4a0", "method4a1",
)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _fmt(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}" if _finite(value) else "nan"


def _stats(values: list[float]) -> dict[str, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"median": np.nan, "p90": np.nan, "maximum": np.nan}
    return {
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "maximum": float(np.max(finite)),
    }


def _comparison_rows(comparison: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    methods = [name for name in METHOD_ORDER if name in comparison["global"]]
    rows: list[dict[str, Any]] = []
    for sequence, metrics in sorted(comparison["per_sequence"].items()):
        row: dict[str, Any] = {"sequence": sequence}
        for method in methods:
            row[f"{method}_ate_mm"] = metrics.get(f"{method}_mean_ate_mm", np.nan)
        component_values = [
            row.get(f"{name}_ate_mm", np.nan)
            for name in ("method1", "method1_5", "method2a", "method2b")
        ]
        valid_components = [float(value) for value in component_values if _finite(value)]
        best_component = min(valid_components) if valid_components else np.nan
        row["best_component_ate_mm"] = best_component
        row["method4a1_minus_best_component_mm"] = (
            float(row["method4a1_ate_mm"]) - best_component
            if _finite(row.get("method4a1_ate_mm")) and _finite(best_component)
            else np.nan
        )
        row["method4a1_minus_method2b_mm"] = (
            float(row["method4a1_ate_mm"]) - float(row["method2b_ate_mm"])
            if _finite(row.get("method4a1_ate_mm")) and _finite(row.get("method2b_ate_mm"))
            else np.nan
        )
        row["method4a1_downweighted_frames"] = metrics.get("method4a1_downweighted_frames", 0)
        row["method4a1_optimization_seconds"] = metrics.get(
            "method4a1_optimization_seconds", np.nan
        )
        rows.append(row)
    return methods, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["sequence"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _diagnostic_summary(run_summary: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    position_medians: list[float] = []
    position_p90s: list[float] = []
    rotation_medians: list[float] = []
    rotation_p90s: list[float] = []
    vo_medians: list[float] = []
    vo_p90s: list[float] = []
    runtimes: list[float] = []
    downweighted_rows: list[dict[str, Any]] = []
    optimizer_failures: list[str] = []

    for sequence, summary in run_summary["sequences"].items():
        position_medians.append(float(summary["position_disagreement"]["median"]))
        position_p90s.append(float(summary["position_disagreement"]["p90"]))
        rotation_medians.append(float(summary["rotation_disagreement_deg"]["median"]))
        rotation_p90s.append(float(summary["rotation_disagreement_deg"]["p90"]))
        vo_medians.append(float(summary["vo_residual"]["median"]))
        vo_p90s.append(float(summary["vo_residual"]["p90"]))
        runtimes.append(float(summary["optimization_seconds"]))
        if not bool(summary.get("optimizer_success", False)):
            optimizer_failures.append(sequence)
        for frame_id in summary.get("downweighted_frame_ids", []):
            downweighted_rows.append({"sequence": sequence, "frame_id": int(frame_id)})

    result = {
        "num_sequences": len(run_summary["sequences"]),
        "position_disagreement_sequence_median": _stats(position_medians),
        "position_disagreement_sequence_p90": _stats(position_p90s),
        "rotation_disagreement_deg_sequence_median": _stats(rotation_medians),
        "rotation_disagreement_deg_sequence_p90": _stats(rotation_p90s),
        "vo_residual_sequence_median": _stats(vo_medians),
        "vo_residual_sequence_p90": _stats(vo_p90s),
        "optimization_seconds": {
            **_stats(runtimes),
            "total": float(np.sum(runtimes)),
        },
        "total_downweighted_frames": len(downweighted_rows),
        "optimizer_failures": optimizer_failures,
    }
    return result, downweighted_rows


def _render_markdown(
    *,
    comparison: dict[str, Any],
    methods: list[str],
    rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> str:
    lines = ["# Method 4A final released-data report", ""]
    lines.extend(["## Overall metrics", "", "| Method | Mean ATE | Median ATE | ATE std |", "|---|---:|---:|---:|"])
    for method in methods:
        metrics = comparison["global"][method]
        lines.append(
            f"| {method} | {_fmt(metrics.get('mean_ate'))} | "
            f"{_fmt(metrics.get('median_ate'))} | {_fmt(metrics.get('std_ate'))} |"
        )

    lines.extend(["", "## RPE", "", "| Method | d | Translation (mm) | Rotation (deg) |", "|---|---:|---:|---:|"])
    for method in methods:
        metrics = comparison["global"][method]
        for delta in (1, 10, 20, 40):
            lines.append(
                f"| {method} | {delta} | {_fmt(metrics.get(f'trans_rpe_{delta}frame'))} | "
                f"{_fmt(metrics.get(f'rot_rpe_deg_{delta}frame'))} |"
            )

    deltas_best = [float(row["method4a1_minus_best_component_mm"]) for row in rows if _finite(row["method4a1_minus_best_component_mm"])]
    deltas_b = [float(row["method4a1_minus_method2b_mm"]) for row in rows if _finite(row["method4a1_minus_method2b_mm"])]
    lines.extend([
        "", "## Improvement counts", "",
        f"- Method 4A-1 improved over the best individual component on {sum(value < 0 for value in deltas_best)}/{len(deltas_best)} sequences.",
        f"- Method 4A-1 worsened versus the best individual component on {sum(value > 0 for value in deltas_best)}/{len(deltas_best)} sequences.",
        f"- Method 4A-1 improved over Method 2B on {sum(value < 0 for value in deltas_b)}/{len(deltas_b)} sequences.",
    ])

    ranked = sorted(
        [row for row in rows if _finite(row["method4a1_minus_best_component_mm"])],
        key=lambda row: float(row["method4a1_minus_best_component_mm"]),
    )
    lines.extend(["", "## Largest changes versus best component", "", "| Sequence | Method 4A-1 ATE | Best component ATE | Difference |", "|---|---:|---:|---:|"])
    for row in (ranked[:5] + ranked[-5:]):
        lines.append(
            f"| {row['sequence']} | {_fmt(row['method4a1_ate_mm'])} | "
            f"{_fmt(row['best_component_ate_mm'])} | "
            f"{_fmt(row['method4a1_minus_best_component_mm'])} |"
        )

    lines.extend(["", "## Inference-only diagnostics", ""])
    runtime = diagnostics["optimization_seconds"]
    lines.extend([
        f"- Sequences: {diagnostics['num_sequences']}",
        f"- Total optimization time: {_fmt(runtime['total'], 3)} seconds",
        f"- Median optimization time/sequence: {_fmt(runtime['median'], 4)} seconds",
        f"- P90 optimization time/sequence: {_fmt(runtime['p90'], 4)} seconds",
        f"- Total frames downweighted below 0.5: {diagnostics['total_downweighted_frames']}",
        f"- Optimizer failures: {len(diagnostics['optimizer_failures'])}",
        f"- Median of sequence-median A/B position disagreements: {_fmt(diagnostics['position_disagreement_sequence_median']['median'])}",
        f"- Median of sequence-median A/B rotation disagreements: {_fmt(diagnostics['rotation_disagreement_deg_sequence_median']['median'])} deg",
        f"- Median of sequence-median VO residuals: {_fmt(diagnostics['vo_residual_sequence_median']['median'])}",
    ])

    lines.extend(["", "## Per-sequence ATE", "", "| Sequence | M1 | M1.5 | M2A | M2B | 4A-0 | 4A-1 | 4A-1 - best |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(
            f"| {row['sequence']} | {_fmt(row.get('method1_ate_mm'))} | "
            f"{_fmt(row.get('method1_5_ate_mm'))} | {_fmt(row.get('method2a_ate_mm'))} | "
            f"{_fmt(row.get('method2b_ate_mm'))} | {_fmt(row.get('method4a0_ate_mm'))} | "
            f"{_fmt(row.get('method4a1_ate_mm'))} | "
            f"{_fmt(row.get('method4a1_minus_best_component_mm'))} |"
        )
    lines.extend([
        "", "## Interpretation", "",
        "Method 4A-1 is frozen at the TRAIN-selected configuration. Ground truth was used only by evaluation/reporting after inference. Diagnostic rankings must not be used for further TEST-set tuning.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize frozen Method 4A results.")
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    comparison = json.loads(args.comparison.read_text())
    run_summary = json.loads(args.run_summary.read_text())
    methods, rows = _comparison_rows(comparison)
    diagnostics, downweighted_rows = _diagnostic_summary(run_summary)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_root / "per_sequence_ate.csv", rows)
    _write_csv(args.output_root / "downweighted_frames.csv", downweighted_rows)
    (args.output_root / "aggregate_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    report = _render_markdown(
        comparison=comparison,
        methods=methods,
        rows=rows,
        diagnostics=diagnostics,
    )
    (args.output_root / "REPORT.md").write_text(report)
    print(f"Saved report: {args.output_root / 'REPORT.md'}")
    print(f"Saved per-sequence table: {args.output_root / 'per_sequence_ate.csv'}")
    print(f"Saved downweighted frames: {args.output_root / 'downweighted_frames.csv'}")


if __name__ == "__main__":
    main()
