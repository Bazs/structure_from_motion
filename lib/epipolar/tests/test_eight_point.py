import unittest

from scipy.spatial.transform import Rotation
import numpy as np

from lib.common.feature import Feature
from lib.epipolar import eight_point
from lib.feature_matching.matching import Match


class EightPointTest(unittest.TestCase):
    def test_get_normalized_match_coordinates(self):
        image_size = (256, 512)
        features_a = [Feature(256, 128), Feature(128, 64)]
        features_b = [Feature(32, 64), Feature(16, 8)]
        matches = [Match(0, 1), Match(1, 0)]

        coords_a, coords_b = eight_point._get_normalized_match_coordinates(
            features_a, features_b, matches, image_size
        )

        expected_coords_a = np.array([[0.0, 0.0], [-0.5, -0.5]])
        np.testing.assert_allclose(expected_coords_a, coords_a)
        expected_coords_b = np.array(
            [
                [1.0 / 32.0 * 2.0 - 1.0, 1.0 / 32.0 * 2.0 - 1.0],
                [1.0 / 16.0 * 2.0 - 1.0, 1.0 / 4.0 * 2.0 - 1.0],
            ]
        )
        np.testing.assert_allclose(expected_coords_b, coords_b)

    def test_get_y_col(self):
        coord_a = np.array([2.0, 3.0])
        coord_b = np.array([7.0, 6.0])

        y_col = eight_point._get_y_col(coord_a, coord_b)
        self.assertEqual((9, 1), y_col.shape)

        expected_y_col = np.array(
            [
                [7.0 * 2.0],
                [7.0 * 3.0],
                [7.0],
                [6.0 * 2.0],
                [6.0 * 3.0],
                [6.0],
                [2.0],
                [3.0],
                [1.0],
            ]
        )

        np.testing.assert_allclose(expected_y_col, y_col)

    def test_estimate_essential_matrix(self):
        rectangle_width = np.array([1.0, 0.0, 0.0])
        rectangle_height = np.array([0.0, 0.5, 0.0])
        rectangle_origin = np.zeros((3,), dtype=float)

        rectangle_a = np.array(
            [
                rectangle_origin,
                rectangle_origin + rectangle_width,
                rectangle_origin + rectangle_width + rectangle_height,
                rectangle_origin + rectangle_height,
            ]
        )

        rectangle_b = rectangle_a.copy()
        rectangle_b_offset = np.array([2.0, 0.0, 0.0])
        rectangle_b += rectangle_b_offset

        rect_a_rot = Rotation.from_euler("y", 45.0, degrees=True)
        rectangle_a = rect_a_rot.apply(rectangle_a)

        rect_b_rot = Rotation.from_euler("y", -45.0, degrees=True)
        rectangle_b = rect_b_rot.apply(rectangle_b)

        # TODO define camera poses and do projection


if __name__ == "__main__":
    unittest.main()
