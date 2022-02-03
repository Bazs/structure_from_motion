import numpy as np

from lib.common import feature as feat
from lib.feature_matching import util


def calculate_nnc(
    image_a: np.ndarray,
    image_b: np.ndarray,
    feature_a: feat.Feature,
    feature_b: feat.Feature,
    window_size: int = 5,
) -> float:
    """Calculate the Normalized Cross-Correlation between two features in two images."""
    if image_a.shape != image_b.shape:
        raise ValueError("the images must have the same shape")

    if not util.is_within_bounds(
        feature_a, image_a.shape, window_size
    ) or not util.is_within_bounds(feature_b, image_a.shape, window_size):
        return np.NINF

    window_a = util.select_window(image_a, feature_a, window_size)
    window_b = util.select_window(image_b, feature_b, window_size)

    mu_a = np.mean(window_a)
    mu_b = np.mean(window_b)

    window_a_shifted = window_a - mu_a
    window_b_shifted = window_b - mu_b

    numerator = np.dot(window_a_shifted.flatten(), window_b_shifted.flatten())
    denominator = np.sqrt(
        np.sum(np.square(window_a_shifted)) * np.sum(np.square(window_b_shifted))
    )

    if denominator == 0:
        return -1.0

    nnc = numerator / denominator
    if not -1.0 <= nnc <= 1.0:
        print(f"Out of bounds: {nnc}")
        assert False
    # assert -1.0 <= nnc <= 1.0
    return nnc
