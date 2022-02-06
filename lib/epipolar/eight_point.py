from typing import List, Tuple

import numpy as np

from lib.common.feature import Feature
from lib.feature_matching.matching import Match


def estimate_essential_mat(
    features_a: List[Feature],
    features_b: List[Feature],
    matches: List[Match],
    image_size: Tuple[int, int],
) -> np.ndarray:
    """
    Estimates the Essential Matrix from eight point correspondences.

    @param features_a List of features from the first image.
    @param features_b List of features from the second image.
    @param matches Exactly eight matches between features_a and features_b.
    @param image_size The size of the original images A and B from which the matching features
    are from. The size is [image_height, image_width].
    @return The 3x3 Essential Matrix (https://en.wikipedia.org/wiki/Essential_matrix).
    """
    if 8 != len(matches):
        raise ValueError("Exactly eight matches are needed")
    coords_a, coords_b = _get_normalized_match_coordinates(
        features_a, features_b, matches, image_size
    )


def _get_normalized_match_coordinates(
    features_a: List[Feature],
    features_b: List[Feature],
    matches: List[Match],
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the matching feature coordinates normalized to [-1 1].

    @param features_a List of features from the first image.
    @param features_b List of features from the second image.
    @param matches N matches between features_a and features_b.
    @param image_size The size of the original images A and B from which the matching features
    are from. The size is [image_height, image_width].
    @return A pair of Nx2 matrices of normalized matching coordinates, one for features_b
    and one for features_a.
    """
    coords_a = np.empty((len(matches), 2), dtype=float)
    coords_b = np.empty((len(matches), 2), dtype=float)

    def get_normalized_coord(feature: Feature, imsize: Tuple[int, int]) -> np.ndarray:
        """Returns the feature's coordinates normalized to the [-1 1] range."""
        assert 0 <= feature.x <= imsize[1]
        assert 0 <= feature.y <= imsize[0]
        coord = np.array([feature.x / imsize[1], feature.y / imsize[0]])
        coord *= 2.0
        coord -= 1.0
        return coord

    for index, match in enumerate(matches):
        feature_a = features_a[match.a_index]
        coord_a = get_normalized_coord(feature_a, image_size)
        coords_a[index, :] = coord_a
        feature_b = features_b[match.b_index]
        coord_b = get_normalized_coord(feature_b, image_size)
        coords_b[index, :] = coord_b

    return coords_a, coords_b
