from functools import partial
import unittest

import numpy as np

from lib.common import feature
from lib.feature_matching import ssd


class SsdTest(unittest.TestCase):
    def test_ssd(self):
        image_a = np.array(
            [
                [1, 2, 3, 0, 0, 0],
                [4, 5, 6, 0, 0, 0],
                [7, 8, 9, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        image_b = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 9, 8, 7],
                [0, 0, 0, 6, 5, 4],
                [0, 0, 0, 3, 2, 1],
            ]
        )

        window_size = 3
        calculate_ssd = lambda feature_a, feature_b: ssd.calculate_ssd(
            image_a, image_b, feature_a, feature_b, window_size=window_size
        )

        output = calculate_ssd(feature.Feature(1, 1), feature.Feature(4, 4))
        sq = lambda x: x ** 2

        self.assertEqual(
            (sq(8) + sq(6) + sq(4) + sq(2) + sq(0) + sq(2) + sq(4) + sq(6) + sq(8))
            / window_size
            / window_size,
            output,
        )

        # Check out-of-bounds features
        self.assertEqual(
            np.Infinity, calculate_ssd(feature.Feature(0, 0), feature.Feature(4, 4))
        )
        self.assertEqual(
            np.Infinity, calculate_ssd(feature.Feature(1, 1), feature.Feature(5, 5))
        )
