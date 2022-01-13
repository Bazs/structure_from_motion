import numpy as np


def create_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    """
    Creates a kernel_size x kernel_size Gaussian smoothing kernel.

    The output kernel will be normalized.

    :param kernel_size: The size of the kernel to create.
    :return: The normalized kernel as a numpy array.
    """
    if kernel_size <= 2:
        raise ValueError("kernel_size must be at least 3")
    if kernel_size % 2 == 0:
        raise ValueError("Only odd-sized kernels are accepted")
    half_kernel_size = int(kernel_size / 2)
    x_coords = np.arange(-half_kernel_size, half_kernel_size + 1)
    y_coords = np.arange(-half_kernel_size, half_kernel_size + 1)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    kernel = np.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))
    kernel /= 2 * np.pi * sigma ** 2
    kernel /= np.sum(kernel)

    return kernel
