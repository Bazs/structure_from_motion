import unittest

import cv2 as cv
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

    def test_cross_correlate_complex(self):
        input_image = np.array(
            [[1, 5, 4, 3, 7], [2, 5, 7, 4, -10], [9, -5, 4, 3, 2]], dtype=float
        )
        kernel = np.array([[1, -2, 3], [2, 1, 0], [7, -5, 1]], dtype=float)
        output = harris._cross_correlate(input_image, kernel)
        self.assertTrue(np.allclose(output[-1:2, -1:2], 0))
        self.assertEqual(
            1 * 1 + 5 * -2 + 4 * 3 + 2 * 2 + 5 * 1 + 7 * 0 + 9 * 7 + -5 * -5 + 4 * 1,
            output[1, 1],
        )
        self.assertEqual(
            5 * 1 + 4 * -2 + 3 * 3 + 5 * 2 + 7 * 1 + 4 * 0 + -5 * 7 + 4 * -5 + 3 * 1,
            output[1, 2],
        )
        self.assertEqual(
            4 * 1 + 3 * -2 + 7 * 3 + 7 * 2 + 4 * 1 + -10 * 0 + 4 * 7 + 3 * -5 + 2 * 1,
            output[1, 3],
        )

    def test_calculate_cornerness_image(self):
        background = np.zeros((100, 200), dtype=float)
        image = cv.rectangle(background, (50, 25), (150, 75), 255, -1)
        window_name = "window"
        sobel_x = harris._apply_sobel_x(image)
        cornerness_image = harris._calculate_cornerness_image(image)
        # TODO check that maxima occur around the corners
