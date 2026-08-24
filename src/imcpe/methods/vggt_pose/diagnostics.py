from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pose_utils import rotation_angle_degrees


@dataclass
class Method3FrameDiagnostic:
    frame_id: int
    valid: bool
    failure_reason: str
    inference_seconds: float
    peak_gpu_memory_bytes: int
    original_resolution: list[int]
    input_resolution: list[int]
    scale_reference_e1_baseline_raw: float
    e1_stereo_baseline_raw: float
    e2_stereo_baseline_raw: float
    scale_factor: float
    cross_translation_magnitude_scaled: float
    e1_stereo_rotation_deg: float
    e2_stereo_rotation_deg: float
    focal_relative_errors: list[float]
    rotation_orthogonality_error_max: float
    rotation_determinant_error_max: float
    used_previous_fallback: bool
    duplicate_t0_cross_rotation_disagreement_deg: float = float("nan")
    duplicate_t0_cross_translation_disagreement: float = float("nan")


def _finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _summary(values: list[float]) -> dict[str, float]:
    array = _finite(values)
    if not array.size:
        return {"median": float("nan"), "p90": float("nan")}
    return {
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }


def _relative_variation(values: list[float]) -> dict[str, float]:
    array = _finite(values)
    if not array.size:
        return {"median": float("nan"), "p90": float("nan")}
    center = float(np.median(array))
    if center <= 0.0:
        return {"median": float("nan"), "p90": float("nan")}
    deviation = np.abs(array / center - 1.0)
    return {
        "median": float(np.median(deviation)),
        "p90": float(np.percentile(deviation, 90)),
    }


def summarize_diagnostics(items: list[Method3FrameDiagnostic]) -> dict[str, object]:
    valid = [item for item in items if item.valid]
    inference = [item.inference_seconds for item in items]
    return {
        "num_frames": len(items),
        "num_valid": len(valid),
        "valid_fraction": len(valid) / max(len(items), 1),
        "num_previous_fallbacks": sum(item.used_previous_fallback for item in items),
        "inference_seconds": _summary(inference),
        "total_camera_inference_seconds": float(np.sum(_finite(inference))),
        "peak_gpu_memory_bytes": max(
            (item.peak_gpu_memory_bytes for item in items), default=0
        ),
        "scale_reference_e1_baseline_raw": _summary(
            [item.scale_reference_e1_baseline_raw for item in valid]
        ),
        "e1_stereo_baseline_raw": _summary(
            [item.e1_stereo_baseline_raw for item in valid]
        ),
        "e2_stereo_baseline_raw": _summary(
            [item.e2_stereo_baseline_raw for item in valid]
        ),
        "e2_over_e1_baseline_ratio": _summary(
            [
                item.e2_stereo_baseline_raw / item.e1_stereo_baseline_raw
                for item in valid
                if item.e1_stereo_baseline_raw > 0
            ]
        ),
        "e1_stereo_baseline_relative_variation": _relative_variation(
            [item.e1_stereo_baseline_raw for item in valid]
        ),
        "e2_stereo_baseline_relative_variation": _relative_variation(
            [item.e2_stereo_baseline_raw for item in valid]
        ),
        "scale_factor": _summary([item.scale_factor for item in valid]),
        "cross_translation_magnitude_scaled": _summary(
            [item.cross_translation_magnitude_scaled for item in valid]
        ),
        "e1_stereo_rotation_deg": _summary(
            [item.e1_stereo_rotation_deg for item in valid]
        ),
        "e2_stereo_rotation_deg": _summary(
            [item.e2_stereo_rotation_deg for item in valid]
        ),
        "focal_relative_error": _summary(
            [error for item in valid for error in item.focal_relative_errors]
        ),
    }


def stereo_rotation_variation(relative_rotations: np.ndarray) -> dict[str, float]:
    if len(relative_rotations) == 0:
        return {"median_deg": float("nan"), "p90_deg": float("nan")}
    reference = relative_rotations[0]
    angles = [rotation_angle_degrees(reference.T @ current) for current in relative_rotations]
    return {
        "median_deg": float(np.median(angles)),
        "p90_deg": float(np.percentile(angles, 90)),
    }
