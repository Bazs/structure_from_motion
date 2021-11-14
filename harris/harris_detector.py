import numpy as np

_sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)


def detect_harris_corners(image: np.ndarray, block_size: int = 2, k: float = 0.04):
    pass


def _calculate_cornerness_image(
    image: np.ndarray, block_size: int = 2, k: float = 0.04
):
    sobel_y = _apply_sobel_y(image)
    sobel_x = _apply_sobel_x(image)
    # TODO shift intensities back into the positive domain
    width = image.shape[1]
    height = image.shape[0]

    cornerness_image = np.zeros(
        (
            height - int(np.around(block_size / 2)),
            width - int(np.around(block_size / 2)),
        ),
        dtype=float,
    )

    for row_idx in range(height - block_size):
        for col_idx in range(width - block_size):
            x_block = sobel_x[
                row_idx : row_idx + block_size, col_idx : col_idx + block_size
            ]
            y_block = sobel_y[
                row_idx : row_idx + block_size, col_idx : col_idx + block_size
            ]
            Ix2 = np.sum(x_block ** 2)
            Ix_Iy = np.dot(x_block.flatten(), y_block.flatten())
            Iy2 = np.sum(y_block ** 2)
            M = np.array([[Ix2, Ix_Iy], [Ix_Iy, Iy2]])
            cornerness_value = np.linalg.det(M) - k * np.trace(M) ** 2
            if cornerness_value:
                break
            cornerness_image[row_idx, col_idx] = cornerness_value

    return cornerness_image


def _apply_sobel_x(image: np.ndarray) -> np.ndarray:
    return _cross_correlate(image, _sobel_x_kernel)


def _apply_sobel_y(image: np.ndarray) -> np.ndarray:
    return _cross_correlate(image, _sobel_x_kernel.transpose())


def _cross_correlate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Slides the kernel over the image and calculates the per-pixel cross correlation.

    Uses zero padding. Only odd-sized square kernels are supported.
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
