from typing import Tuple

import numpy as np

from lib.common import feature as feat


def calculate_ssd(
    image_a: np.ndarray,
    image_b: np.ndarray,
    feature_a: feat.Feature,
    feature_b: feat.Feature,
    window_size: int = 5,
) -> float:
    """Calculate the Squared Sum of Differences between two patches of two images.

    :param image_a: The image where feature_a belongs to.
    :param image_b: The image where feature_b belongs to.
    :param feature_a: One feature identified by an image coordinate, center of one of the patches
    :param feature_b: Center of the other patch
    :param window_size: The size of a patch. If the patch would extend beyond the image size for either feature_a or
    feature_b, np.Infinity is returned.
    :return: The SSD between the two patches.
    """
    if image_a.shape != image_b.shape:
        raise ValueError("the images must have the same shape")

    if not _is_within_bounds(
        feature_a, image_a.shape, window_size
    ) or not _is_within_bounds(feature_b, image_a.shape, window_size):
        return np.Infinity

    diff = _select_window(image_a, feature_a, window_size) - _select_window(
        image_b, feature_b, window_size
    )
    square_diff = np.square(diff)
    return np.sum(square_diff)


def _is_within_bounds(
    feature: feat.Feature, image_shape: Tuple[int, int], window_size: int
) -> bool:
    half_window_size = int(window_size / 2)
    if not half_window_size <= feature.y < (image_shape[0] - half_window_size):
        return False
    if not half_window_size <= feature.x < (image_shape[1] - half_window_size):
        return False

    return True


def _select_window(
    image: np.ndarray, feature: feat.Feature, window_size: int
) -> np.ndarray:
    half_window_size = int(window_size / 2)
    return image[
        int(feature.y) - half_window_size : int(feature.y) + half_window_size + 1,
        int(feature.x) - half_window_size : int(feature.x) + half_window_size + 1,
    ]
