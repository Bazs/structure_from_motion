import numpy as np


def cross_correlate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Slides the kernel over the image and calculates the per-pixel cross correlation.

    Uses zero "same" padding. Only odd-sized square kernels are supported.
    :param image The image to cross correlate. Must be grayscale.
    :param kernel The kernel to cross correlate with.
    :return The result of the cross correlation, same size as image due to the zero padding.
    """
    if len(image.shape) != 2 or len(kernel.shape) != 2:
        raise ValueError("Only 2D single channel images are supported")
    height = image.shape[0]
    width = image.shape[1]

    if kernel.shape[0] != kernel.shape[1] or (kernel.shape[0] % 2) == 0:
        raise ValueError("Only odd-sized square kernels are supported")
    kernel_size = kernel.shape[0]
    if height < kernel_size or width < kernel_size:
        raise ValueError("Kernel cannot be larger than image")

    output_width = width - kernel_size + 1
    output_height = height - kernel_size + 1

    kernel_half_size = int(kernel_size / 2)
    output_image = np.zeros(image.shape, dtype=float)

    for row_idx in range(kernel_half_size, kernel_half_size + output_height):
        for column_idx in range(kernel_half_size, kernel_half_size + output_width):
            output_image[row_idx, column_idx] = np.dot(
                image[
                    row_idx - kernel_half_size : row_idx + kernel_half_size + 1,
                    column_idx - kernel_half_size : column_idx + kernel_half_size + 1,
                ].flatten(),
                kernel.flatten(),
            )

    return output_image
