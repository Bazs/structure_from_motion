from typing import Tuple

import numpy as np

from lib.common import feature as feat


def is_within_bounds(
    feature: feat.Feature, image_shape: Tuple[int, int], window_size: int
) -> bool:
    half_window_size = int(window_size / 2)
    if not half_window_size <= feature.y < (image_shape[0] - half_window_size):
        return False
    if not half_window_size <= feature.x < (image_shape[1] - half_window_size):
        return False

    return True


def select_window(
    image: np.ndarray, feature: feat.Feature, window_size: int
) -> np.ndarray:
    half_window_size = int(window_size / 2)
    return image[
        int(feature.y) - half_window_size : int(feature.y) + half_window_size + 1,
        int(feature.x) - half_window_size : int(feature.x) + half_window_size + 1,
    ]
