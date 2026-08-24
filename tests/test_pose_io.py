from __future__ import annotations

import numpy as np

from imcpe.io_pose import PoseRow, read_pose_txt, write_pose_txt


def test_pose_txt_xyzw_round_trip(tmp_path) -> None:
    path = tmp_path / "pose.txt"
    rows = [
        PoseRow(2, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])),
        PoseRow(8, np.array([1.2, -3.4, 5.6]), np.array([0.1, 0.2, 0.3, 0.9])),
    ]
    write_pose_txt(path, rows)
    recovered = read_pose_txt(path)
    assert [row.frame_idx for row in recovered] == [2, 8]
    np.testing.assert_allclose(recovered[1].t, rows[1].t)
    np.testing.assert_allclose(recovered[1].q_xyzw, rows[1].q_xyzw)

