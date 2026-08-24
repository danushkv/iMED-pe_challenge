"""Strict one-to-one spatial association for pairwise LoFTR coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class AssociationResult:
    loftr_indices: np.ndarray
    sparse_indices: np.ndarray
    distances_px: np.ndarray


def mutual_nearest_coordinate_association(
    loftr_source_points: np.ndarray,
    sparse_source_points: np.ndarray,
    tolerance_px: float,
) -> AssociationResult:
    """Associate LoFTR source coordinates to sparse keypoints one-to-one.

    A pair is accepted only when each point is the other's nearest neighbor and
    their distance does not exceed the fixed TRAIN-selected tolerance.
    """

    loftr = np.asarray(loftr_source_points, dtype=np.float64).reshape(-1, 2)
    sparse = np.asarray(sparse_source_points, dtype=np.float64).reshape(-1, 2)
    if loftr.size == 0 or sparse.size == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return AssociationResult(empty_i, empty_i.copy(), np.empty(0))
    sparse_distance, sparse_index = cKDTree(sparse).query(loftr, k=1)
    _, reverse_loftr_index = cKDTree(loftr).query(sparse, k=1)
    loftr_index = np.arange(loftr.shape[0], dtype=np.int64)
    mutual = reverse_loftr_index[sparse_index] == loftr_index
    accepted = mutual & (sparse_distance <= float(tolerance_px))
    return AssociationResult(
        loftr_indices=loftr_index[accepted],
        sparse_indices=np.asarray(sparse_index[accepted], dtype=np.int64),
        distances_px=np.asarray(sparse_distance[accepted], dtype=np.float64),
    )

