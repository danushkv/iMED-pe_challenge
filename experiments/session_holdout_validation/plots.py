"""Session-balanced LOSO and calibration diagnostic plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _session_matrix(summary: dict[str, Any]) -> tuple[list[str], list[str], np.ndarray]:
    sessions = list(summary["session_table"])
    methods = list(summary["methods"])
    matrix = np.asarray([
        [summary["session_table"][session][method]["mean_ate"] for session in sessions]
        for method in methods
    ], dtype=float)
    return sessions, methods, matrix


def plot_session_ate(summary: dict[str, Any], output_root: Path) -> None:
    sessions, methods, matrix = _session_matrix(summary)
    output_root.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(sessions), dtype=float)
    width = 0.82 / max(len(methods), 1)
    figure, axis = plt.subplots(figsize=(max(9, 1.8 * len(sessions)), 5.5))
    for index, method in enumerate(methods):
        offset = (index - (len(methods) - 1) / 2.0) * width
        axis.bar(x + offset, matrix[index], width=width, label=method)
    axis.set_xticks(x, sessions, rotation=25, ha="right")
    axis.set_ylabel("Session mean ATE (mm)")
    axis.set_title("Leave-one-session-out performance")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(fontsize=8, ncol=min(4, len(methods)))
    figure.tight_layout()
    figure.savefig(output_root / "session_ate_grouped.png", dpi=190, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(
        figsize=(max(7, 1.5 * len(sessions)), max(4, 0.75 * len(methods)))
    )
    image = axis.imshow(matrix, cmap="viridis", aspect="auto")
    axis.set_xticks(np.arange(len(sessions)), sessions, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(methods)), methods)
    axis.set_title("Session x method mean ATE (mm)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", color="white" if matrix[row, column] > np.nanmedian(matrix) else "black", fontsize=8)
    figure.colorbar(image, ax=axis, label="Mean ATE (mm)")
    figure.tight_layout()
    figure.savefig(output_root / "session_method_heatmap.png", dpi=190, bbox_inches="tight")
    plt.close(figure)


def plot_calibration_analysis(
    calibration: dict[str, Any],
    output_root: Path,
) -> None:
    rows = calibration.get("session_table", {})
    if not rows:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("e1_stereo", "translation_direction_p90_deg", "method2a_ate_mm", "E1 translation direction p90 (deg)"),
        ("e1_stereo", "rotation_p90_deg", "method2a_ate_mm", "E1 rotation p90 (deg)"),
        ("e1_stereo", "reprojection_error_median_px", "method2a_ate_mm", "E1 reprojection median (px)"),
        ("e2_stereo", "translation_direction_p90_deg", "method2b_ate_mm", "E2 translation direction p90 (deg)"),
        ("e2_stereo", "rotation_p90_deg", "method2b_ate_mm", "E2 rotation p90 (deg)"),
        ("e2_stereo", "reprojection_error_median_px", "method2b_ate_mm", "E2 reprojection median (px)"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, (stereo, key, ate_key, label) in zip(axes.ravel(), specifications):
        for session_id, row in rows.items():
            x = row.get(stereo, {}).get(key, np.nan)
            y = row.get(ate_key, np.nan)
            if np.isfinite(x) and np.isfinite(y):
                axis.scatter(x, y, s=55)
                axis.annotate(session_id, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
        correlation_key = f"{stereo}_{key}_vs_{ate_key.removesuffix('_mm')}"
        correlation = calibration.get("correlations", {}).get(correlation_key, {}).get("pearson_r", np.nan)
        axis.set_xlabel(label)
        axis.set_ylabel("Session mean ATE (mm)")
        axis.set_title(f"Pearson r={correlation:.3f}" if np.isfinite(correlation) else "Pearson r=nan")
        axis.grid(True, alpha=0.3)
    figure.suptitle("Calibration stability versus held-out session ATE")
    figure.tight_layout()
    figure.savefig(output_root / "calibration_quality_vs_ate.png", dpi=190, bbox_inches="tight")
    plt.close(figure)
