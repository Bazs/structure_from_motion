import unittest
from functools import partial

from lib.common import feature
from lib.feature_matching import util


class UtilTest(unittest.TestCase):
    def test_is_within_bounds(self):
        image_shape = (100, 200)
        window_size = 5

        is_within_bounds = partial(
            util.is_within_bounds, image_shape=image_shape, window_size=window_size
        )

        self.assertTrue(is_within_bounds(feature.Feature(2, 2)))
        self.assertFalse(is_within_bounds(feature.Feature(1, 2)))
        self.assertFalse(is_within_bounds(feature.Feature(2, 1)))
        self.assertTrue(is_within_bounds(feature.Feature(y=97, x=197)))
        self.assertFalse(is_within_bounds(feature.Feature(y=98, x=197)))
        self.assertFalse(is_within_bounds(feature.Feature(y=97, x=198)))
