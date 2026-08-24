"""Mutual-nearest one-to-one association in original source-image pixels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class MultiViewAssociation:
    stereo_indices: np.ndarray
    cross_indices: np.ndarray
    distances_px: np.ndarray
    combined_confidence: np.ndarray


def associate_source_coordinates(
    stereo_points: np.ndarray,
    cross_points: np.ndarray,
    stereo_confidence: np.ndarray,
    cross_confidence: np.ndarray,
    tolerance_px: float,
) -> MultiViewAssociation:
    """Return strict mutual-nearest associations with deterministic tie breaks.

    Candidate pairs are ordered first by source-coordinate distance and then by
    descending geometric-mean confidence. Each source point in either pairwise
    match set can appear in at most one returned three-view track.
    """

    stereo = np.asarray(stereo_points, dtype=np.float64).reshape(-1, 2)
    cross = np.asarray(cross_points, dtype=np.float64).reshape(-1, 2)
    stereo_conf = np.asarray(stereo_confidence, dtype=np.float64).reshape(-1)
    cross_conf = np.asarray(cross_confidence, dtype=np.float64).reshape(-1)
    if len(stereo) != len(stereo_conf) or len(cross) != len(cross_conf):
        raise ValueError("Point and confidence arrays have inconsistent lengths")
    if not len(stereo) or not len(cross):
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return MultiViewAssociation(empty_i, empty_i.copy(), empty_f, empty_f.copy())

    neighborhoods = cKDTree(cross).query_ball_point(stereo, r=float(tolerance_px))
    candidates: list[tuple[int, int, float, float]] = []
    for stereo_index, cross_indices in enumerate(neighborhoods):
        for cross_index in cross_indices:
            distance = float(np.linalg.norm(stereo[stereo_index] - cross[cross_index]))
            confidence = float(
                np.sqrt(max(stereo_conf[stereo_index], 0.0) * max(cross_conf[cross_index], 0.0))
            )
            candidates.append((stereo_index, int(cross_index), distance, confidence))
    if not candidates:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return MultiViewAssociation(empty_i, empty_i.copy(), empty_f, empty_f.copy())

    # The first candidate for either endpoint is its nearest; exact-distance
    # ties prefer the larger combined confidence.
    ordered = sorted(candidates, key=lambda item: (item[2], -item[3], item[0], item[1]))
    best_for_stereo: dict[int, tuple[int, float, float]] = {}
    best_for_cross: dict[int, tuple[int, float, float]] = {}
    for stereo_index, cross_index, distance, confidence in ordered:
        best_for_stereo.setdefault(stereo_index, (cross_index, distance, confidence))
        best_for_cross.setdefault(cross_index, (stereo_index, distance, confidence))

    accepted: list[tuple[int, int, float, float]] = []
    for stereo_index, (cross_index, distance, confidence) in best_for_stereo.items():
        reverse = best_for_cross[cross_index]
        if reverse[0] == stereo_index:
            accepted.append((stereo_index, cross_index, distance, confidence))
    accepted.sort(key=lambda item: item[0])
    return MultiViewAssociation(
        stereo_indices=np.asarray([item[0] for item in accepted], dtype=np.int64),
        cross_indices=np.asarray([item[1] for item in accepted], dtype=np.int64),
        distances_px=np.asarray([item[2] for item in accepted], dtype=np.float64),
        combined_confidence=np.asarray([item[3] for item in accepted], dtype=np.float64),
    )

