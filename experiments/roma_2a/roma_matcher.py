"""Exact-index ALIKED-source/RoMa-warp adapter for existing Method 2A."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from experiments.roma_2a.cycle_consistency import compute_cycle_consistency
from experiments.roma_2a.roma_wrapper import OfficialRoMa
from experiments.roma_2a.source_points import ALIKEDSourceDetector
from experiments.roma_2a.warp_points import sample_dense_warp


@dataclass
class RoMaFeature:
    path: Path
    keypoints_px: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    detector_scores: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    image_shape_hw: tuple[int, int] | None = None
    extraction_seconds: float = 0.0


def _stream(path: Path) -> tuple[str | None, str | None]:
    parts = path.parts
    for camera in ("endoscope1", "endoscope2"):
        if camera in parts:
            index = parts.index(camera)
            side = parts[index + 1] if index + 1 < len(parts) else None
            return camera, side
    return None, None


def _image_shape(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return int(image.shape[0]), int(image.shape[1])


def _finite_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _finite_p90(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, 90)) if values.size else float("nan")


class ExactSourceRoMaMatcher:
    """Expose RoMa three-view tracks through Method 2A's index API.

    `run_sequence_method2` calls stereo matching first and cross-camera
    matching second using the same E1-L feature object.  This adapter detects
    E1-L ALIKED points once, samples both dense RoMa fields at those exact
    points, and returns the original ALIKED ID on both branches.  Therefore the
    existing Method-2A intersection is exact and performs no spatial merge.
    """

    def __init__(
        self,
        *,
        roma: OfficialRoMa,
        detector: ALIKEDSourceDetector,
        certainty_threshold: float | None,
        cycle_threshold_px: float | None,
        reuse_symmetric_reverse: bool = True,
    ) -> None:
        if certainty_threshold is not None and not 0.0 <= certainty_threshold <= 1.0:
            raise ValueError("certainty_threshold must be in [0, 1]")
        if cycle_threshold_px is not None and cycle_threshold_px <= 0:
            raise ValueError("cycle_threshold_px must be positive")
        self.roma = roma
        self.detector = detector
        self.certainty_threshold = certainty_threshold
        self.cycle_threshold_px = cycle_threshold_px
        self.reuse_symmetric_reverse = bool(reuse_symmetric_reverse)
        self.history: list[dict[str, Any]] = []
        self._pending_stereo: dict[str, Any] | None = None

    def extract(self, image_path: Path) -> RoMaFeature:
        path = Path(image_path)
        camera, side = _stream(path)
        shape = _image_shape(path)
        feature = RoMaFeature(path=path, image_shape_hw=shape)
        # Method 2A's common source is always E1-L. Target points are produced
        # directly by the dense RoMa warp, so targets need no detector pass.
        if camera == "endoscope1" and side == "L":
            source = self.detector.extract(path)
            if tuple(source.image_shape_hw) != tuple(shape):
                raise RuntimeError(
                    f"ALIKED/OpenCV image shape mismatch for {path}: "
                    f"{source.image_shape_hw} vs {shape}"
                )
            feature.keypoints_px = source.points_px
            feature.detector_scores = source.scores
            feature.extraction_seconds = source.extraction_seconds
        return feature

    @staticmethod
    def _pair_kind(features0: RoMaFeature, features1: RoMaFeature) -> str:
        camera0, side0 = _stream(features0.path)
        camera1, side1 = _stream(features1.path)
        if camera0 == camera1 == "endoscope1" and side0 == "L" and side1 == "R":
            return "stereo"
        if camera0 == "endoscope1" and camera1 == "endoscope2" and side0 == side1 == "L":
            return "cross"
        raise RuntimeError(f"Unexpected Method-2A pair: {features0.path} -> {features1.path}")

    def _warp_pair(
        self, features0: RoMaFeature, features1: RoMaFeature
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if features0.image_shape_hw is None or features1.image_shape_hw is None:
            raise RuntimeError("Missing original image shape")
        if len(features0.keypoints_px) == 0:
            raise RuntimeError(f"No ALIKED source points for {features0.path}")

        forward = self.roma.match(features0.path, features1.path)
        sampled = sample_dense_warp(
            forward.warp,
            forward.certainty,
            features0.keypoints_px,
            features0.image_shape_hw,
            features1.image_shape_hw,
        )
        valid_warp = sampled.valid.copy()
        certainty_mask = valid_warp.copy()
        if self.certainty_threshold is not None:
            certainty_mask &= sampled.certainty >= self.certainty_threshold

        cycle_error = np.full(len(features0.keypoints_px), np.nan, dtype=np.float64)
        cycle_valid = valid_warp.copy()
        reverse_seconds = 0.0
        reverse_peak_gib = 0.0
        reverse_source = "not_requested"
        if self.cycle_threshold_px is not None:
            if (
                self.reuse_symmetric_reverse
                and forward.reverse_warp is not None
                and forward.reverse_certainty is not None
            ):
                reverse_warp = forward.reverse_warp
                reverse_certainty = forward.reverse_certainty
                reverse_source = "reused_from_forward_symmetric_inference"
            else:
                reverse = self.roma.match(features1.path, features0.path)
                reverse_warp = reverse.warp
                reverse_certainty = reverse.certainty
                reverse_seconds = reverse.seconds
                reverse_peak_gib = reverse.peak_gpu_memory_gib
                reverse_source = "separate_explicit_inference"
            cycle = compute_cycle_consistency(
                features0.keypoints_px,
                sampled.target_points_px,
                sampled.valid,
                reverse_warp,
                reverse_certainty,
                source_shape_hw=features0.image_shape_hw,
                target_shape_hw=features1.image_shape_hw,
            )
            cycle_error = cycle.error_px
            cycle_valid = cycle.valid & (cycle.error_px <= self.cycle_threshold_px)

        accepted = certainty_mask & cycle_valid
        features1.keypoints_px = sampled.target_points_px
        features1.detector_scores = sampled.certainty
        indices = np.flatnonzero(accepted).astype(np.int64)
        record = {
            "source": str(features0.path),
            "target": str(features1.path),
            "aliked_source_points": int(len(features0.keypoints_px)),
            "valid_forward_warps": int(np.count_nonzero(valid_warp)),
            "certainty_surviving": int(np.count_nonzero(certainty_mask)),
            "cycle_surviving": int(np.count_nonzero(cycle_valid)),
            "accepted_tracks": int(len(indices)),
            "accepted_source_indices": indices,
            "certainty_median_all_valid": _finite_median(
                sampled.certainty[valid_warp]
            ),
            "certainty_median_accepted": _finite_median(sampled.certainty[accepted]),
            "certainty_p90_accepted": _finite_p90(sampled.certainty[accepted]),
            "cycle_error_median_px": _finite_median(cycle_error[valid_warp]),
            "cycle_error_p90_px": _finite_p90(cycle_error[valid_warp]),
            "forward_seconds": forward.seconds,
            "reverse_seconds": reverse_seconds,
            "reverse_warp_source": reverse_source,
            "peak_gpu_memory_gib": max(
                forward.peak_gpu_memory_gib, reverse_peak_gib
            ),
            "certainty_threshold": self.certainty_threshold,
            "cycle_threshold_px": self.cycle_threshold_px,
            "coordinates_in_original_pixels": True,
        }
        return np.column_stack((indices, indices)), record

    def match_indices(
        self, features0: RoMaFeature, features1: RoMaFeature
    ) -> np.ndarray:
        pair_kind = self._pair_kind(features0, features1)
        matches, record = self._warp_pair(features0, features1)
        if pair_kind == "stereo":
            self._pending_stereo = record
            return matches

        if self._pending_stereo is None:
            raise RuntimeError("Cross-camera RoMa call occurred before stereo call")
        stereo = self._pending_stereo
        stereo_ids = set(int(value) for value in np.asarray(
            self._last_source_indices(stereo, features0, "stereo")
        ))
        cross_ids = set(int(value) for value in matches[:, 0])
        joint_ids = stereo_ids & cross_ids
        self.history.append(
            {
                "e1_left": str(features0.path),
                "e1_right": stereo["target"],
                "e2_left": record["target"],
                "aliked_source_points": record["aliked_source_points"],
                "valid_stereo_warps": stereo["valid_forward_warps"],
                "valid_cross_warps": record["valid_forward_warps"],
                "stereo_certainty_surviving": stereo["certainty_surviving"],
                "cross_certainty_surviving": record["certainty_surviving"],
                "stereo_cycle_surviving": stereo["cycle_surviving"],
                "cross_cycle_surviving": record["cycle_surviving"],
                "stereo_accepted_tracks": stereo["accepted_tracks"],
                "cross_accepted_tracks": record["accepted_tracks"],
                "joint_three_view_tracks": int(len(joint_ids)),
                "stereo_certainty_median": stereo["certainty_median_accepted"],
                "cross_certainty_median": record["certainty_median_accepted"],
                "stereo_cycle_error_median_px": stereo["cycle_error_median_px"],
                "cross_cycle_error_median_px": record["cycle_error_median_px"],
                "stereo_cycle_error_p90_px": stereo["cycle_error_p90_px"],
                "cross_cycle_error_p90_px": record["cycle_error_p90_px"],
                "aliked_extraction_seconds": features0.extraction_seconds,
                "stereo_forward_seconds": stereo["forward_seconds"],
                "stereo_reverse_seconds": stereo["reverse_seconds"],
                "cross_forward_seconds": record["forward_seconds"],
                "cross_reverse_seconds": record["reverse_seconds"],
                "stereo_reverse_warp_source": stereo["reverse_warp_source"],
                "cross_reverse_warp_source": record["reverse_warp_source"],
                "peak_gpu_memory_gib": max(
                    stereo["peak_gpu_memory_gib"], record["peak_gpu_memory_gib"]
                ),
                "certainty_threshold": self.certainty_threshold,
                "cycle_threshold_px": self.cycle_threshold_px,
                "exact_source_identity": True,
                "nearest_neighbour_association_used": False,
                "symmetric_reverse_reuse_enabled": self.reuse_symmetric_reverse,
            }
        )
        self._pending_stereo = None
        return matches

    @staticmethod
    def _last_source_indices(
        record: dict[str, Any], features0: RoMaFeature, label: str
    ) -> np.ndarray:
        # Accepted target arrays retain the original ALIKED indexing. Recreate
        # the accepted source IDs from the masks by caching them in the record.
        indices = record.get("accepted_source_indices")
        if indices is None:
            raise RuntimeError(f"Missing cached {label} accepted source indices")
        return np.asarray(indices, dtype=np.int64)

    @staticmethod
    def keypoints(features: RoMaFeature) -> np.ndarray:
        return features.keypoints_px
