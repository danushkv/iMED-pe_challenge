from __future__ import annotations

import numpy as np

from imcpe.methods.stereo_vo.features import FeatureDict, SharedFeatureMatcher


Method2BFeatureMatcher = SharedFeatureMatcher


def intersect_e2_left_tracks(
    stereo_matches: np.ndarray,
    cross_matches: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect exact E2-left indices shared by stereo and cross matches.

    Returns E2-left indices, E2-right indices, and E1-left indices.
    """

    cross_by_e2_left = {
        int(e2_left): int(e1_left)
        for e2_left, e1_left in np.asarray(cross_matches).reshape(-1, 2)
    }
    shared_e2_left: list[int] = []
    shared_e2_right: list[int] = []
    shared_e1_left: list[int] = []
    for e2_left, e2_right in np.asarray(stereo_matches).reshape(-1, 2):
        e1_left = cross_by_e2_left.get(int(e2_left))
        if e1_left is None:
            continue
        shared_e2_left.append(int(e2_left))
        shared_e2_right.append(int(e2_right))
        shared_e1_left.append(e1_left)
    return (
        np.asarray(shared_e2_left, dtype=np.int64),
        np.asarray(shared_e2_right, dtype=np.int64),
        np.asarray(shared_e1_left, dtype=np.int64),
    )


__all__ = [
    "FeatureDict",
    "Method2BFeatureMatcher",
    "intersect_e2_left_tracks",
]
