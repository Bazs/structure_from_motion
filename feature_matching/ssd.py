from typing import Sequence, Tuple

import numpy as np

from common import feature as feat


def calculate_ssd(
    image: np.ndarray,
    feature_a: feat.Feature,
    feature_b: feat.Feature,
    window_size: int = 5,
) -> float:
    pass


def _is_within_bounds(
    feature: feat.Feature, image_shape: Tuple[int, int], window_size: int
) -> bool:
    half_window_size = int(window_size / 2)
    if not half_window_size <= feature.y < (image_shape[0] - half_window_size):
        return False
    if not half_window_size <= feature.x < (image_shape[1] - half_window_size):
        return False

    return True
