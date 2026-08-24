from __future__ import annotations

import numpy as np

from imcpe.methods.stereo_vo.features import FeatureDict, SharedFeatureMatcher


Method2FeatureMatcher = SharedFeatureMatcher


def intersect_e1_left_tracks(
    stereo_matches: np.ndarray,
    cross_matches: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect exact E1-left feature indices in stereo and cross matches.

    Returns E1-left indices, E1-right indices, and E2-left indices.
    """

    cross_by_e1_left = {
        int(e1_left): int(e2_left)
        for e1_left, e2_left in np.asarray(cross_matches).reshape(-1, 2)
    }
    shared_e1_left: list[int] = []
    shared_e1_right: list[int] = []
    shared_e2_left: list[int] = []
    for e1_left, e1_right in np.asarray(stereo_matches).reshape(-1, 2):
        e2_left = cross_by_e1_left.get(int(e1_left))
        if e2_left is None:
            continue
        shared_e1_left.append(int(e1_left))
        shared_e1_right.append(int(e1_right))
        shared_e2_left.append(e2_left)
    return (
        np.asarray(shared_e1_left, dtype=np.int64),
        np.asarray(shared_e1_right, dtype=np.int64),
        np.asarray(shared_e2_left, dtype=np.int64),
    )


__all__ = [
    "FeatureDict",
    "Method2FeatureMatcher",
    "intersect_e1_left_tracks",
]

