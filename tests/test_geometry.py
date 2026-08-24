from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from imcpe.alignment import horn_align_sim3
from imcpe.geometry import from_homogeneous, to_homogeneous


def test_homogeneous_round_trip() -> None:
    rotation = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_matrix()
    translation = np.array([1.0, 2.0, -3.0])
    recovered_rotation, recovered_translation = from_homogeneous(
        to_homogeneous(rotation, translation)
    )
    np.testing.assert_allclose(recovered_rotation, rotation)
    np.testing.assert_allclose(recovered_translation, translation)


def test_horn_recovers_sim3() -> None:
    model = np.array(
        [[0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, -1.0], [0.0, 0.2, 0.4, 1.0]]
    )
    rotation = Rotation.from_euler("z", 23.0, degrees=True).as_matrix()
    scale = 2.5
    translation = np.array([[3.0], [-2.0], [0.5]])
    data = scale * rotation @ model + translation
    found_rotation, found_translation, errors, found_scale = horn_align_sim3(model, data)
    np.testing.assert_allclose(found_rotation, rotation, atol=1e-10)
    np.testing.assert_allclose(found_translation, translation, atol=1e-10)
    np.testing.assert_allclose(found_scale, scale, atol=1e-10)
    np.testing.assert_allclose(errors, 0.0, atol=1e-10)

