from pathlib import Path
import unittest

import cv2.cv2
import cv2.cv2 as cv
import numpy as np

import sys

sys.path.append("/Users/bopra/code/structure_from_motion")

import blur.gaussian as gaussian
import common.correlate as correlate
import harris.harris_detector as harris


class TestHarrisDetector(unittest.TestCase):
    def test_detect_harris_corners(self) -> None:
        background = np.zeros((100, 200), dtype=float)
        top_y = 25
        bottom_y = 75
        left_x = 50
        right_x = 150

        expected_corners = [
            np.array([top_y, left_x]),
            np.array([top_y, right_x]),
            np.array([bottom_y, left_x]),
            np.array([bottom_y, right_x]),
        ]

        image = cv.rectangle(background, (left_x, top_y), (right_x, bottom_y), 255, -1)
        corner_coordinates = harris.detect_harris_corners(image)
        for expected_corner, corner in zip(expected_corners, corner_coordinates):
            self.assertTrue(np.allclose(expected_corner, corner, atol=1.0))

    def test_harris_visual(self):
        test_image_path = Path(
            "/Users/bopra/code/structure_from_motion/data/barcelona/DSCN8238.JPG"
        )
        self.assertTrue(test_image_path.is_file())

        test_image = cv.imread(str(test_image_path))
        downscale_factor = 8
        smaller_size = (
            int(test_image.shape[1] / downscale_factor),
            int(test_image.shape[0] / downscale_factor),
        )
        test_image = cv.resize(
            test_image, smaller_size, interpolation=cv2.INTER_LANCZOS4
        )
        gray_test_image = cv.cvtColor(test_image, cv.COLOR_RGB2GRAY)

        gaussian_kernel = gaussian.create_gaussian_kernel(5, 1.0)
        gray_test_image = correlate.cross_correlate(gray_test_image, gaussian_kernel)

        corner_coordinates = harris.detect_harris_corners(gray_test_image)

        for coordinate in corner_coordinates:
            center = [int(coordinate[1]), int(coordinate[0])]
            test_image = cv.circle(
                test_image, center, radius=1, color=(255, 0, 0), thickness=-1
            )

        window_name = "window"
        cv.namedWindow(
            window_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED
        )
        cv.imshow(window_name, gray_test_image)
        cv.waitKey()
        cv.imshow(window_name, test_image)
        cv.waitKey()


if __name__ == "__main__":
    unittest.main()
