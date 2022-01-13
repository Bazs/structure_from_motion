import cv2.cv2 as cv
import numpy as np

from common import correlate

_sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)


def detect_harris_corners(image: np.ndarray, block_size: int = 2, k: float = 0.04):
    """Detects corners in the image using the Harris corner detector algorithm.
    See https://en.wikipedia.org/wiki/Harris_corner_detector.

    @param image MxN matrix representing a grayscale image.
    @param block_size The block size to examine for cornerness. A block of block_size x block_size dimensions will be
    used to calculate the second moment matrix for each pixel.
    @param k The constant to use in the cornerness-function calculation.
    """
    cornerness_image = _calculate_cornerness_image(image, block_size, k)
    cornerness_image[cornerness_image < 0] = 0.0
    _non_max_suppress(cornerness_image)
    y_indices, x_indices = np.nonzero(cornerness_image)
    y_indices = y_indices.astype(float) + float(block_size) / 2.0
    x_indices = x_indices.astype(float) + float(block_size) / 2.0
    corner_coordinates = [
        np.array([y_index, x_index]) for y_index, x_index in zip(y_indices, x_indices)
    ]
    return corner_coordinates


def _calculate_cornerness_image(
    image: np.ndarray, block_size: int = 2, k: float = 0.04
):
    sobel_y = _apply_sobel_y(image)
    sobel_x = _apply_sobel_x(image)
    Ix2_mat = sobel_x ** 2
    Iy2_mat = sobel_y ** 2
    Ix_Iy_mat = sobel_x * sobel_y
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
            Ix2 = np.sum(_select_window(Ix2_mat, row_idx, col_idx, block_size))
            Ix_Iy = np.sum(_select_window(Ix_Iy_mat, row_idx, col_idx, block_size))
            Iy2 = np.sum(_select_window(Iy2_mat, row_idx, col_idx, block_size))
            M = np.array([[Ix2, Ix_Iy], [Ix_Iy, Iy2]])
            cornerness_value = np.linalg.det(M) - k * (np.trace(M) ** 2)
            cornerness_image[row_idx, col_idx] = cornerness_value

    return cornerness_image


def _select_window(
    matrix: np.ndarray, row: int, col: int, window_size: int
) -> np.ndarray:
    return matrix[row : row + window_size, col : col + window_size]


def _non_max_suppress(image: np.ndarray):
    for row_idx in range(image.shape[0]):
        for col_idx in range(image.shape[1]):
            left_idx = max(0, col_idx - 1)
            right_idx = min(image.shape[1], col_idx + 2)
            top_idx = max(0, row_idx - 1)
            bottom_idx = min(image.shape[0], row_idx + 2)
            window = image[top_idx:bottom_idx, left_idx:right_idx]
            if image[row_idx, col_idx] < np.amax(window):
                image[row_idx, col_idx] = 0.0


def _apply_sobel_x(image: np.ndarray) -> np.ndarray:
    return correlate.cross_correlate(image, _sobel_x_kernel)


def _apply_sobel_y(image: np.ndarray) -> np.ndarray:
    return correlate.cross_correlate(image, _sobel_x_kernel.transpose())
