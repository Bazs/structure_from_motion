import unittest

import numpy as np

from lib.blur import gaussian


class TestGaussian(unittest.TestCase):
    def test_create_gaussian_kernel(self):
        kernel_size = 5
        kernel = gaussian.create_gaussian_kernel(kernel_size, 1.0)

        self.assertEqual((kernel_size, kernel_size), kernel.shape)
        self.assertAlmostEqual(1.0, np.sum(kernel))

        half_kernel_size = int(kernel_size / 2)

        for i in range(half_kernel_size + 1):
            for j in range(half_kernel_size + 1):
                # Check that elements increase towards the center
                if i < half_kernel_size:
                    self.assertLess(kernel[i, j], kernel[i + 1, j])
                if j < half_kernel_size:
                    self.assertLess(kernel[i, j], kernel[i, j + 1])
                # Check that the kernel is symmetric
                self.assertEqual(kernel[i, j], kernel[kernel_size - i - 1, j])
                self.assertEqual(kernel[i, j], kernel[i, kernel_size - j - 1])

    def test_create_gaussian_kernel_invalid_size(self):
        with self.assertRaises(ValueError):
            gaussian.create_gaussian_kernel(4, 1.0)


if __name__ == "__main__":
    unittest.main()
