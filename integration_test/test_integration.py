from pathlib import Path
import unittest

import cv2 as cv


class IntegrationTest(unittest.TestCase):
    def disabled_test_integration(self):
        dataset_folder = Path("data/barcelona")
        self.assertTrue(dataset_folder.is_dir())

        test_image_1_filename = "DSCN8238.JPG"
        test_image_2_filename = "DSCN8238.JPG"

        window_name = "window"
        cv.namedWindow(
            window_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED
        )
