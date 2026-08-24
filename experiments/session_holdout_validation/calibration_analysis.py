"""Session calibration quality versus held-out Method-2 performance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _first(payload: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return float(value)
    return float("nan")


def _raw_calibration(root: Path, session_id: str) -> dict[str, Any]:
    suffix = session_id.removeprefix("session_")
    path = root / f"session_{suffix}_diagnostics.json"
    if not path.is_file():
        return {"available": False, "source": str(path)}
    payload = json.loads(path.read_text())
    return {
        "available": True,
        "source": str(path),
        "rotation_p90_deg": _first(
            payload, "rotation_p90_deg", "rotation_variation_p90_deg"
        ),
        "translation_direction_p90_deg": _first(
            payload,
            "translation_direction_p90_deg",
            "translation_direction_variation_p90_deg",
        ),
        "reprojection_error_median_px": _first(
            payload,
            "median_stereo_reprojection_error_px",
            "reprojection_error_median_px",
        ),
        "epipolar_error_median_px": _first(
            payload,
            "median_epipolar_error_px",
            "pooled_inlier_epipolar_sampson_median_approx_px",
        ),
        "valid": payload.get("valid"),
    }


def _analysis_table(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text()).get("session_table", {})


def _normalized_existing(payload: dict[str, Any], source: Path) -> dict[str, Any]:
    return {
        "available": True,
        "source": str(source),
        "rotation_p90_deg": _first(payload, "rotation_p90_deg"),
        "translation_direction_p90_deg": _first(
            payload, "translation_direction_p90_deg"
        ),
        "reprojection_error_median_px": _first(
            payload,
            "reprojection_error_median_px",
            "median_stereo_reprojection_error_px",
        ),
        "epipolar_error_median_px": _first(
            payload, "median_epipolar_error_px"
        ),
        "valid": payload.get("valid"),
        "validity_failures": payload.get("validity_failures", []),
    }


def load_calibration_table(
    *,
    session_ids: list[str],
    session_method_metrics: dict[str, dict[str, Any]],
    e1_root: Path | None,
    e2_root: Path | None,
    analysis_json: Path | None,
    method2a_name: str,
    method2b_name: str,
) -> dict[str, dict[str, Any]]:
    existing = _analysis_table(analysis_json) if analysis_json is not None else {}
    result: dict[str, dict[str, Any]] = {}
    for session_id in session_ids:
        suffix = session_id.removeprefix("session_")
        existing_row = existing.get(suffix, existing.get(session_id, {}))
        if existing_row:
            e1 = _normalized_existing(existing_row.get("e1_stereo", {}), analysis_json)
            e2 = _normalized_existing(existing_row.get("e2_stereo", {}), analysis_json)
        else:
            e1 = _raw_calibration(e1_root, session_id) if e1_root else {"available": False}
            e2 = _raw_calibration(e2_root, session_id) if e2_root else {"available": False}

        method_values = session_method_metrics[session_id]
        result[session_id] = {
            "e1_stereo": e1,
            "e2_stereo": e2,
            "method2a_ate_mm": method_values.get(method2a_name, {}).get("mean_ate", np.nan),
            "method2b_ate_mm": method_values.get(method2b_name, {}).get("mean_ate", np.nan),
        }
    return result


def _pearson(rows: dict[str, dict[str, Any]], path: tuple[str, str], ate_key: str) -> dict[str, Any]:
    sessions: list[str] = []
    x: list[float] = []
    y: list[float] = []
    for session_id, row in rows.items():
        value = row.get(path[0], {}).get(path[1], np.nan)
        ate = row.get(ate_key, np.nan)
        if np.isfinite(value) and np.isfinite(ate):
            sessions.append(session_id)
            x.append(float(value))
            y.append(float(ate))
    correlation = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else np.nan
    return {"num_sessions": len(x), "pearson_r": correlation, "sessions": sessions}


def calibration_correlations(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stereo, method in (("e1_stereo", "method2a"), ("e2_stereo", "method2b")):
        ate_key = f"{method}_ate_mm"
        for key in (
            "translation_direction_p90_deg",
            "rotation_p90_deg",
            "reprojection_error_median_px",
        ):
            result[f"{stereo}_{key}_vs_{method}_ate"] = _pearson(
                rows, (stereo, key), ate_key
            )
    return result
