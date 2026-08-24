"""Method-2 matcher adapter backed exclusively by pairwise EfficientLoFTR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from config import MultiViewConfig
from loftr_wrapper import EfficientLoFTRMatcher
from source_association import associate_source_coordinates


@dataclass
class MultiViewFeature:
    path: Path
    keypoints_px: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    confidence: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )


def _stream(path: Path) -> tuple[str | None, str | None]:
    parts = path.parts
    for name in ("endoscope1", "endoscope2"):
        if name in parts:
            index = parts.index(name)
            side = parts[index + 1] if index + 1 < len(parts) else None
            return name, side
    return None, None


class PureLoFTRMultiViewMatcher:
    """Expose LoFTR three-view tracks through Method 2's exact-index API.

    Method 2 always calls stereo matching first. That call establishes the
    shared-left keypoint array using the actual stereo source coordinates. The
    subsequent cross-camera call associates its independently predicted source
    coordinates to that stored array and returns the stereo-source indices.
    """

    def __init__(self, *, loftr: EfficientLoFTRMatcher, config: MultiViewConfig) -> None:
        self.loftr = loftr
        self.config = config
        self.history: list[dict[str, Any]] = []
        self._pending_stereo: dict[str, Any] | None = None

    def extract(self, image_path: Path) -> MultiViewFeature:
        return MultiViewFeature(path=Path(image_path))

    @staticmethod
    def _pair_kind(features0: MultiViewFeature, features1: MultiViewFeature) -> str:
        camera0, side0 = _stream(features0.path)
        camera1, side1 = _stream(features1.path)
        if camera0 == camera1 and side0 == "L" and side1 == "R":
            return "stereo"
        if camera0 != camera1 and side0 == "L" and side1 == "L":
            return "cross"
        raise RuntimeError(f"Unexpected pair ordering: {features0.path} -> {features1.path}")

    def match_indices(
        self, features0: MultiViewFeature, features1: MultiViewFeature
    ) -> np.ndarray:
        pair_kind = self._pair_kind(features0, features1)
        raw = self.loftr.match(features0.path, features1.path)
        filtered = raw.filtered(self.config.confidence_threshold)

        if pair_kind == "stereo":
            features0.keypoints_px = np.asarray(filtered.points0, dtype=np.float64)
            features0.confidence = np.asarray(filtered.confidence, dtype=np.float64)
            features1.keypoints_px = np.asarray(filtered.points1, dtype=np.float64)
            features1.confidence = np.asarray(filtered.confidence, dtype=np.float64)
            count = len(filtered.confidence)
            self._pending_stereo = {
                "source": str(features0.path),
                "target": str(features1.path),
                "raw_matches": int(len(raw.confidence)),
                "filtered_matches": int(count),
                "confidence_median": (
                    float(np.median(filtered.confidence)) if count else float("nan")
                ),
                "inference_seconds": raw.inference_seconds,
                "original_shape0": raw.original_shape0,
                "inference_shape0": raw.inference_shape0,
            }
            indices = np.arange(count, dtype=np.int64)
            return np.column_stack((indices, indices))

        if self._pending_stereo is None or not len(features0.keypoints_px):
            raise RuntimeError("Cross-camera LoFTR call occurred before stereo source initialization")
        association = associate_source_coordinates(
            features0.keypoints_px,
            filtered.points0,
            features0.confidence,
            filtered.confidence,
            self.config.association_tolerance_px,
        )
        # The target array is indexed only by accepted cross-camera matches.
        features1.keypoints_px = np.asarray(
            filtered.points1[association.cross_indices], dtype=np.float64
        )
        features1.confidence = np.asarray(
            filtered.confidence[association.cross_indices], dtype=np.float64
        )
        target_indices = np.arange(len(association.cross_indices), dtype=np.int64)
        distances = association.distances_px
        combined = association.combined_confidence
        stereo = self._pending_stereo
        self.history.append(
            {
                "stereo_source": stereo["source"],
                "stereo_target": stereo["target"],
                "cross_source": str(features0.path),
                "cross_target": str(features1.path),
                "stereo_raw_matches": stereo["raw_matches"],
                "stereo_filtered_matches": stereo["filtered_matches"],
                "cross_raw_matches": int(len(raw.confidence)),
                "cross_filtered_matches": int(len(filtered.confidence)),
                "three_view_tracks": int(len(association.stereo_indices)),
                "association_rate_vs_stereo": float(
                    len(association.stereo_indices) / max(stereo["filtered_matches"], 1)
                ),
                "association_rate_vs_cross": float(
                    len(association.stereo_indices) / max(len(filtered.confidence), 1)
                ),
                "association_distance_median_px": (
                    float(np.median(distances)) if len(distances) else float("nan")
                ),
                "association_distance_p90_px": (
                    float(np.percentile(distances, 90)) if len(distances) else float("nan")
                ),
                "stereo_confidence_median": stereo["confidence_median"],
                "cross_confidence_median": (
                    float(np.median(filtered.confidence))
                    if len(filtered.confidence)
                    else float("nan")
                ),
                "combined_confidence_median": (
                    float(np.median(combined)) if len(combined) else float("nan")
                ),
                "confidence_threshold": self.config.confidence_threshold,
                "association_tolerance_px": self.config.association_tolerance_px,
                "stereo_inference_seconds": stereo["inference_seconds"],
                "cross_inference_seconds": raw.inference_seconds,
                "coordinates_in_original_pixels": True,
                "original_shape": raw.original_shape0,
                "inference_shape": raw.inference_shape0,
            }
        )
        self._pending_stereo = None
        return np.column_stack((association.stereo_indices, target_indices)).astype(
            np.int64, copy=False
        )

    @staticmethod
    def keypoints(features: MultiViewFeature) -> np.ndarray:
        return features.keypoints_px

