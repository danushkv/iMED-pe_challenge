"""ALIKED stereo plus EfficientLoFTR cross-camera matching for L1."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from config import AssociationConfig
from loftr_wrapper import EfficientLoFTRMatcher
from track_association import mutual_nearest_coordinate_association
from imcpe.methods.stereo_vo.features import SharedFeatureMatcher


@dataclass
class HybridFeature:
    path: Path
    aliked: dict[str, Any]
    extra_keypoints: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )


def _stream(path: Path) -> tuple[str | None, str | None]:
    parts = path.parts
    for name in ("endoscope1", "endoscope2"):
        if name in parts:
            index = parts.index(name)
            side = parts[index + 1] if index + 1 < len(parts) else None
            return name, side
    return None, None


class L1HybridMatcher:
    """Conforms to Method 2's matcher interface without changing its geometry."""

    def __init__(
        self,
        *,
        loftr: EfficientLoFTRMatcher,
        device: str,
        max_num_keypoints: int,
        association: AssociationConfig,
    ) -> None:
        self.aliked = SharedFeatureMatcher(
            device=device, max_num_keypoints=max_num_keypoints
        )
        self.loftr = loftr
        self.association = association
        self.cross_history: list[dict[str, Any]] = []

    def extract(self, image_path: Path) -> HybridFeature:
        return HybridFeature(Path(image_path), self.aliked.extract(Path(image_path)))

    @staticmethod
    def _is_cross(left: HybridFeature, right: HybridFeature) -> bool:
        camera0, side0 = _stream(left.path)
        camera1, side1 = _stream(right.path)
        return (
            camera0 is not None
            and camera1 is not None
            and camera0 != camera1
            and side0 == "L"
            and side1 == "L"
        )

    def match_indices(self, features0: HybridFeature, features1: HybridFeature) -> np.ndarray:
        if not self._is_cross(features0, features1):
            return self.aliked.match_indices(features0.aliked, features1.aliked)

        raw = self.loftr.match(features0.path, features1.path)
        filtered = raw.filtered(self.association.confidence_threshold)
        sparse_source = self.aliked.keypoints(features0.aliked)
        association = mutual_nearest_coordinate_association(
            filtered.points0,
            sparse_source,
            self.association.tolerance_px,
        )
        accepted_target = filtered.points1[association.loftr_indices]
        base_target = self.aliked.keypoints(features1.aliked)
        features1.extra_keypoints = np.asarray(accepted_target, dtype=np.float64)
        target_indices = base_target.shape[0] + np.arange(
            association.loftr_indices.size, dtype=np.int64
        )
        confidence = filtered.confidence[association.loftr_indices]
        distances = association.distances_px
        self.cross_history.append(
            {
                "source": str(features0.path),
                "target": str(features1.path),
                "raw_loftr_matches": int(raw.confidence.size),
                "filtered_loftr_matches": int(filtered.confidence.size),
                "associated_matches": int(association.loftr_indices.size),
                "association_rate": float(
                    association.loftr_indices.size / max(filtered.confidence.size, 1)
                ),
                "association_distance_median_px": (
                    float(np.median(distances)) if distances.size else float("nan")
                ),
                "association_distance_p90_px": (
                    float(np.percentile(distances, 90)) if distances.size else float("nan")
                ),
                "associated_confidence_median": (
                    float(np.median(confidence)) if confidence.size else float("nan")
                ),
                "confidence_threshold": self.association.confidence_threshold,
                "association_tolerance_px": self.association.tolerance_px,
                "loftr_inference_seconds": raw.inference_seconds,
                "original_shape0": raw.original_shape0,
                "original_shape1": raw.original_shape1,
                "inference_shape0": raw.inference_shape0,
                "inference_shape1": raw.inference_shape1,
            }
        )
        if association.sparse_indices.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        return np.column_stack(
            [association.sparse_indices, target_indices]
        ).astype(np.int64, copy=False)

    def keypoints(self, features: HybridFeature) -> np.ndarray:
        base = self.aliked.keypoints(features.aliked)
        if features.extra_keypoints.size == 0:
            return base
        return np.concatenate([base, features.extra_keypoints], axis=0)

