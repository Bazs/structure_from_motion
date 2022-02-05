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
    """Calculate the Normalized Cross-Correlation between two features in two images.

    The output score is scaled to be in the interval [0, 2], 0 meaning a perfect match.

    Note that this is different than the raw result of the NNC formula, where the output interval is [-1, 1],
    1 representing a perfect match. The change is to respect the interface of Match.match_score.
    """
    if image_a.shape != image_b.shape:
        raise ValueError("the images must have the same shape")

    if not util.is_within_bounds(
        feature_a, image_a.shape, window_size
    ) or not util.is_within_bounds(feature_b, image_a.shape, window_size):
        return 2.0

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
        return 2.0

    nnc = numerator / denominator
    tolerance = 1e-8
    assert -1.0 - tolerance <= nnc <= 1.0 + tolerance

    # Flip the result and shift to the [0, 2] range
    nnc *= -1.0
    nnc += 1.0

    return nnc
