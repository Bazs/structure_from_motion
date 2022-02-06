import unittest

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


if __name__ == "__main__":
    unittest.main()
