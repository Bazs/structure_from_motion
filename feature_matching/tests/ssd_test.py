from functools import partial
import unittest

import numpy as np

from common import feature
import feature_matching.ssd as ssd


class SsdTest(unittest.TestCase):
    def test_is_within_bounds(self):
        image_shape = (100, 200)
        window_size = 5

        is_within_bounds = partial(
            ssd._is_within_bounds, image_shape=image_shape, window_size=window_size
        )

        self.assertTrue(is_within_bounds(feature.Feature(2, 2)))
        self.assertFalse(is_within_bounds(feature.Feature(1, 2)))
        self.assertFalse(is_within_bounds(feature.Feature(2, 1)))
        self.assertTrue(is_within_bounds(feature.Feature(y=97, x=197)))
        self.assertFalse(is_within_bounds(feature.Feature(y=98, x=197)))
        self.assertFalse(is_within_bounds(feature.Feature(y=97, x=198)))

    def test_ssd(self):
        image = np.array(
            [
                [1, 2, 3, 0, 0, 0],
                [4, 5, 6, 0, 0, 0],
                [7, 8, 9, 0, 0, 0],
                [0, 0, 0, 9, 8, 7],
                [0, 0, 0, 6, 5, 4],
                [0, 0, 0, 3, 2, 1],
            ]
        )

        calculate_ssd = lambda feature_a, feature_b: ssd.calculate_ssd(
            image, feature_a, feature_b, window_size=3
        )

        output = calculate_ssd(feature.Feature(1, 1), feature.Feature(4, 4))
        sq = lambda x: x ** 2

        self.assertEqual(
            sq(8) + sq(6) + sq(4) + sq(2) + sq(0) + sq(2) + sq(4) + sq(6) + sq(8),
            output,
        )

        # Check out-of-bounds features
        self.assertEqual(
            np.Infinity, calculate_ssd(feature.Feature(0, 0), feature.Feature(4, 4))
        )
        self.assertEqual(
            np.Infinity, calculate_ssd(feature.Feature(1, 1), feature.Feature(5, 5))
        )
