import unittest

import numpy as np

import harris.harris_detector as harris


class HarrisDetectorTest(unittest.TestCase):
    def test_cross_correlate_basic(self):
        input_image = np.ones((5, 10), dtype=float)
        kernel = np.ones((3, 3), dtype=float)

        output = harris._cross_correlate(input_image, kernel)
        self.assertTrue(np.allclose(output[-1:2:, -1:2], 0))
        self.assertTrue(np.allclose(output[1:-1, 1:-1], 9))

        kernel = np.ones((5, 5), dtype=float)
        output = harris._cross_correlate(input_image, kernel)
        self.assertTrue(np.allclose(output[-2:3, -2:3], 0))
        self.assertTrue(np.allclose(output[2:-2, 2:-2], 25))
