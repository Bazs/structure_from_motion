from pathlib import Path

import numpy as np

from lib.data_utils.middlebury_utils import load_camera_k_r_t

from ...transforms.transforms import Transform3D


def test_load_camera_intrinsics():
    intrinsics, transform = load_camera_k_r_t(Path(__file__).parent / "test_par.txt", 1)
    expected_intrinsics = np.array(range(1, 10)).reshape((3, 3))
    expected_transform = Transform3D.from_rmat_t(
        np.array(range(1, 10)).reshape(3, 3), np.array(range(1, 4)).reshape((3, 1))
    )
    np.testing.assert_almost_equal(expected_intrinsics, intrinsics)
    np.testing.assert_almost_equal(expected_transform.Tmat, transform.Tmat)
