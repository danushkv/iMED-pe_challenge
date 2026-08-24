from __future__ import annotations

import numpy as np

from experiments.efficient_loftr_multiview.source_association import (
    associate_source_coordinates,
)


def test_mutual_nearest_association_is_one_to_one() -> None:
    stereo = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    cross = np.array([[0.2, 0.1], [10.5, 10.0], [10.7, 10.0], [30.0, 30.0]])
    result = associate_source_coordinates(
        stereo,
        cross,
        np.array([0.9, 0.8, 0.7]),
        np.array([0.9, 0.8, 0.7, 1.0]),
        tolerance_px=1.0,
    )
    np.testing.assert_array_equal(result.stereo_indices, [0, 1])
    np.testing.assert_array_equal(result.cross_indices, [0, 1])
    assert len(set(result.stereo_indices.tolist())) == len(result.stereo_indices)
    assert len(set(result.cross_indices.tolist())) == len(result.cross_indices)


def test_association_respects_tolerance() -> None:
    result = associate_source_coordinates(
        np.array([[0.0, 0.0]]),
        np.array([[3.1, 0.0]]),
        np.ones(1),
        np.ones(1),
        tolerance_px=3.0,
    )
    assert result.stereo_indices.size == 0

