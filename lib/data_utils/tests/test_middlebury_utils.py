from pathlib import Path

import numpy as np

from lib.data_utils.middlebury_utils import load_camera_intrinsics


def test_load_camera_intrinsics():
    intrinsics = load_camera_intrinsics(Path(__file__).parent / "test_par.txt", 1)
    expected_intrinsics = np.array(range(1, 10)).reshape((3, 3))
    np.testing.assert_almost_equal(expected_intrinsics, intrinsics)
