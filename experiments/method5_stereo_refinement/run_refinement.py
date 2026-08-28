#!/usr/bin/env python3
"""Refine, validate, and export one E1 stereo calibration per session."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import StereoRefinementConfig
from .geometry import prepare_frames, skew
from .load_observations import load_session_observations
from .refine_session import refine_and_validate


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _session_ids(cache_root: Path, requested: list[str] | None) -> list[str]:
    available = sorted(path.stem.removeprefix("session_") for path in cache_root.glob("session_*.npz"))
    if requested:
        wanted = {str(value).removeprefix("session_").zfill(3) for value in requested}
        available = [value for value in available if value in wanted]
    if not available:
        raise RuntimeError("No observation caches selected")
    return available


def _write_calibration(
    source: Path,
    destination: Path,
    R_candidate: np.ndarray,
    t_candidate: np.ndarray,
    valid: bool,
    method5_diagnostics: dict[str, Any],
) -> None:
    with np.load(source, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    if valid:
        R_export = np.asarray(R_candidate, dtype=np.float64)
        t_export = np.asarray(t_candidate, dtype=np.float64).reshape(3)
    else:
        R_export = np.asarray(payload["R_LR"], dtype=np.float64)
        t_export = np.asarray(payload["t_LR_unit"], dtype=np.float64).reshape(3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R_export
    transform[:3, 3] = t_export
    payload["R_LR"] = R_export
    payload["t_LR_unit"] = t_export
    payload["T_R_from_L"] = transform
    payload["E"] = skew(t_export) @ R_export
    old_diagnostics: dict[str, Any] = {}
    if "diagnostic_statistics_json" in payload:
        try:
            old_diagnostics = json.loads(str(payload["diagnostic_statistics_json"].item()))
        except (json.JSONDecodeError, ValueError):
            old_diagnostics = {"unparsed_original_diagnostics": True}
    old_diagnostics["method5_stereo_refinement"] = method5_diagnostics
    payload["diagnostic_statistics_json"] = np.array(
        json.dumps(old_diagnostics, sort_keys=True, default=_json_default)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)


def _original_geometry_diagnostics(calibration_root: Path, session: str) -> dict[str, Any]:
    path = calibration_root / f"session_{session}_diagnostics.json"
    if not path.is_file():
        return {"path": str(path), "available": False}
    payload = json.loads(path.read_text())
    keys = (
        "rotation_variation_p50_deg",
        "rotation_variation_p90_deg",
        "translation_direction_variation_p50_deg",
        "translation_direction_variation_p90_deg",
        "reprojection_error_median_px",
        "reprojection_error_p90_px",
        "fixed_transform_appears_valid",
    )
    return {
        "path": str(path),
        "available": True,
        **{key: payload.get(key) for key in keys},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--session", action="append", default=None)
    parser.add_argument("--rotation-bound-deg", type=float, default=3.0)
    parser.add_argument("--translation-bound-deg", type=float, default=5.0)
    parser.add_argument("--rotation-prior", type=float, default=100.0)
    parser.add_argument("--translation-prior", type=float, default=100.0)
    parser.add_argument("--huber-delta-px", type=float, default=2.0)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StereoRefinementConfig(
        max_rotation_correction_deg=args.rotation_bound_deg,
        max_translation_direction_correction_deg=args.translation_bound_deg,
        rotation_prior_weight=args.rotation_prior,
        translation_prior_weight=args.translation_prior,
        huber_delta_px=args.huber_delta_px,
        optimizer_max_nfev=args.max_nfev,
    )
    calibration_output = args.output_root / "calibration"
    diagnostics_output = args.output_root / "diagnostics"
    diagnostics_output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "method": "method5_session_e1_stereo_refinement",
        "config": config.to_dict(),
        "no_ground_truth_used": True,
        "invalid_session_policy": "export original calibration unchanged",
        "sessions": {},
    }
    started_all = time.perf_counter()
    for session in _session_ids(args.cache_root, args.session):
        observations = load_session_observations(
            args.cache_root / f"session_{session}.npz"
        )
        frames, preparation = prepare_frames(observations)
        if len(frames) < config.minimum_subset_frames * 2:
            raise RuntimeError(
                f"session_{session}: only {len(frames)} prepared frames; cannot "
                "perform split-half validation"
            )
        fit, valid, diagnostics = refine_and_validate(
            frames,
            observations.original_R_LR,
            observations.original_t_LR,
            config,
        )
        diagnostics.update({
            "session": session,
            "cache": str(args.cache_root / f"session_{session}.npz"),
            "num_cached_frames": observations.num_frames,
            "num_prepared_frames": len(frames),
            "num_fixed_pnp_inlier_tracks": int(sum(len(frame.x_e1l) for frame in frames)),
            "preparation": preparation,
            "exported_correction_applied": valid,
            "original_calibration_geometry_diagnostics": (
                _original_geometry_diagnostics(args.calibration_root, session)
            ),
        })
        source = args.calibration_root / f"session_{session}.npz"
        destination = calibration_output / f"session_{session}.npz"
        if not source.is_file():
            raise FileNotFoundError(f"Missing original calibration: {source}")
        _write_calibration(source, destination, fit.R_LR, fit.t_LR, valid, diagnostics)
        diagnostics_path = diagnostics_output / f"session_{session}.json"
        diagnostics_path.write_text(
            json.dumps(diagnostics, indent=2, default=_json_default) + "\n"
        )
        summary["sessions"][f"session_{session}"] = {
            "correction_valid": valid,
            "exported_calibration": str(destination),
            "diagnostics": str(diagnostics_path),
            "delta_rotation_deg": diagnostics["full"]["delta_rotation_deg"],
            "delta_translation_direction_deg": diagnostics["full"]["delta_translation_direction_deg"],
            "split_rotation_disagreement_deg": diagnostics["split_half"]["rotation_disagreement_deg"],
            "split_translation_disagreement_deg": diagnostics["split_half"]["translation_direction_disagreement_deg"],
            "relative_median_reprojection_improvement": diagnostics["relative_median_reprojection_improvement"],
            "validity_failures": diagnostics["validity_failures"],
            "optimizer_runtime_seconds": diagnostics["full"]["runtime_seconds"],
        }
        print(
            f"session_{session}: valid={valid} "
            f"dR={diagnostics['full']['delta_rotation_deg']:.3f}deg "
            f"dt={diagnostics['full']['delta_translation_direction_deg']:.3f}deg "
            f"reproj_gain={100.0 * diagnostics['relative_median_reprojection_improvement']:.2f}%",
            flush=True,
        )
    summary["total_runtime_seconds"] = time.perf_counter() - started_all
    summary_path = args.output_root / "refinement_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    print(f"Saved refinement summary: {summary_path}")
    print(f"Saved corrected/fallback calibration: {calibration_output}")


if __name__ == "__main__":
    main()
