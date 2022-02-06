from typing import List, Tuple
import logging

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

    y = _get_y_mat(coords_a, coords_b)
    e_est = _compute_e_est(y)
    e = _enforce_essential_mat_constraints(e_est)

    return e


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


def _get_y_mat(coords_a: np.ndarray, coords_b: np.ndarray):
    assert len(coords_a) == len(coords_b) == 8

    y = np.empty((9, 8), dtype=float)

    for col_idx in range(len(coords_a)):
        y[:, col_idx] = _get_y_col(coords_a[col_idx, :], coords_b[col_idx, :])

    return y


def _get_y_col(coord_a: np.ndarray, coord_b: np.ndarray):
    assert 2 == len(coord_a)
    assert 2 == len(coord_b)

    y_col = np.empty((9, 1), dtype=float)
    y_col[0, 0] = coord_b[0] * coord_a[0]
    y_col[1, 0] = coord_b[0] * coord_a[1]
    y_col[2, 0] = coord_b[0]
    y_col[3, 0] = coord_b[1] * coord_a[0]
    y_col[4, 0] = coord_b[1] * coord_a[1]
    y_col[5, 0] = coord_b[1]
    y_col[6, 0] = coord_a[0]
    y_col[7, 0] = coord_a[1]
    y_col[8, 0] = 1.0

    return y_col


def _compute_e_est(y: np.ndarray) -> np.ndarray:
    """
    Estimates the essential matrix from the Y matrix.
    (https://en.wikipedia.org/wiki/Eight-point_algorithm#Step_2:_Solving_the_equation)

    @param y The Y matrix, constructed from eight matching normalized coordinate pairs.
    @return An estimate of the Essential matrix, not necessarily of rank 2.
    """
    assert (9, 8) == y.shape

    u, s, vh = np.linalg.svd(y)

    # The solution is the left-singular vector of Y corresponding to the smallest singular value
    e_est_vec = u[:, -1]
    assert (9, 1) == e_est_vec.shape
    e_est = e_est_vec.reshape((3, 3))

    return e_est


def _enforce_essential_mat_constraints(e_est: np.ndarray) -> np.ndarray:
    """
    Creates E' from the estimated E matrix, enforcing that rank(E') == 2.
    (https://en.wikipedia.org/wiki/Eight-point_algorithm#Step_3:_Enforcing_the_internal_constraint)

    @param e_est Estimated Essential matrix.
    @return Essential matrix fulfilling the internal constraints.
    """
    u, s, vh = np.linalg.svd(e_est)
    logging.debug(f"Singular values of E_est: {s}")
    # Set the smallest singular value of E_est to 0
    s_prime = np.eye(3, dtype=float) * np.array([s[0], s[1], 0.0])
    e = u @ s_prime @ vh
    return e
