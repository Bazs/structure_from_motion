import unittest

import numpy as np

from lib.common.feature import Feature
from lib.feature_matching import ncc


class TestNcc(unittest.TestCase):
    def setUp(self) -> None:
        self.image_a = np.array(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [9, 8, 7, 6, 5],
                [4, 3, 2, 1, 0],
                [1, 2, 3, 4, 5],
            ]
        )

    def test_nnc_pefect_match(self):
        image_b = np.copy(self.image_a)
        score = ncc.calculate_ncc(
            self.image_a, image_b, Feature(2, 2), Feature(2, 2), 5
        )
        np.testing.assert_allclose(0.0, score, atol=1e-10)

    def test_nnc_worst_match(self):
        image_b = -np.copy(self.image_a)
        score = ncc.calculate_ncc(
            self.image_a, image_b, Feature(2, 2), Feature(2, 2), 5
        )
        np.testing.assert_allclose(2.0, score)

    def test_random_inputs(self):
        num_tests = 100
        window_size = 5
        np.random.seed(55)

        for _ in range(num_tests):
            image_a = np.random.rand(window_size, window_size) + np.random.randint(
                -100, 100
            )
            image_b = np.random.rand(window_size, window_size) + np.random.randint(
                -100, 100
            )
            score = ncc.calculate_ncc(
                image_a, image_b, Feature(2, 2), Feature(2, 2), window_size
            )
            tolerance = 1e-8
            self.assertGreaterEqual(score, 0.0 - tolerance)
            self.assertLess(score, 2.0 + tolerance)


if __name__ == "__main__":
    unittest.main()
