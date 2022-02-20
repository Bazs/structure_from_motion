from typing import List, Tuple
import logging

from scipy.spatial.transform import Rotation
import numpy as np

from lib.common.feature import Feature
from lib.feature_matching.matching import Match


def estimate_r_t(
    features_a: List[Feature],
    features_b: List[Feature],
    matches: List[Match],
    image_size: Tuple[int, int],
):
    """Estimates rotation and translation up to a scale based on eight feature matches between two images.

    Uses the Eight-Point algorithm to estimate the Essential matrix, then decomposes the Essential matrix
    into rotation and translation up to a scale.

    @param features_a List of features from the first image.
    @param features_b List of features from the second image.
    @param matches Exactly eight matches between features_a and features_b.
    @param image_size The size of the original images A and B from which the matching features
    are from. The size is [image_height, image_width].
    @return: Tuple of cam2_R_cam1, cam2_t_cam2_cam1.
    """
    if not features_a or not features_b:
        raise ValueError("Need some matching features")

    e = estimate_essential_mat(features_a, features_b, matches, image_size)

    feature_a = features_a[0]
    match = next(match for match in matches if match.a_index == 0)
    feature_b = features_b[match.b_index]

    r, t = recover_r_t(feature_a, feature_b, e)

    return r, t


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


def recover_r_t(
    feature_a: Feature, feature_b: Feature, e: np.ndarray
) -> Tuple[Rotation, np.ndarray]:
    """Recover the rotation, and the translation up to a scale from an Essential matrix.

    @param feature_a A feature point from the first image.
    @param feature_b Corresponding feature point from the second image.
    @param e essential matrix.
    @return Tuple of rotation, translation.
    """
    u, s, vh = np.linalg.svd(e)

    if not np.isclose(0.0, s[-1]):
        raise ValueError(
            "The smallest singluar value of the Essential matrix is expected to be ~0"
        )

    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    t_1 = u[:, 2]
    t_2 = -t_1
    R_1 = u @ w.T @ vh
    R_2 = u @ w @ vh

    for R in [R_1, R_2]:
        for t in [t_1, t_2]:
            x = _triangulate(feature_a, feature_b, R, t)
            (x)
            # TODO check if x is in front of both cameras

    return None, None


def _get_normalized_match_coordinates(
    features_a: List[Feature],
    features_b: List[Feature],
    matches: List[Match],
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the matching feature coordinates normalized to [-1 1].

    The aspect ratio of coordinates is kept, the larger extent being normalized to [-1, 1].

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

    normalizer = np.max(image_size)

    def get_normalized_coord(feature: Feature) -> np.ndarray:
        """Returns the feature's coordinates normalized to the [-1 1] range."""
        assert 0 <= feature.x <= image_size[1]
        assert 0 <= feature.y <= image_size[0]
        coord = np.array([feature.x / normalizer, feature.y / normalizer])
        coord *= 2.0
        coord -= 1.0
        return coord

    for index, match in enumerate(matches):
        feature_a = features_a[match.a_index]
        coord_a = get_normalized_coord(feature_a)
        coords_a[index, :] = coord_a
        feature_b = features_b[match.b_index]
        coord_b = get_normalized_coord(feature_b)
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

    y_col = np.empty((9,), dtype=float)
    y_col[0] = coord_b[0] * coord_a[0]
    y_col[1] = coord_b[0] * coord_a[1]
    y_col[2] = coord_b[0]
    y_col[3] = coord_b[1] * coord_a[0]
    y_col[4] = coord_b[1] * coord_a[1]
    y_col[5] = coord_b[1]
    y_col[6] = coord_a[0]
    y_col[7] = coord_a[1]
    y_col[8] = 1.0

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
    assert (9,) == e_est_vec.shape
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


def _triangulate(
    feature_a: Feature, feature_b: Feature, Rmat: np.ndarray, t: np.ndarray
) -> np.ndarray:
    u1 = feature_a.x
    v1 = feature_a.y

    u2 = feature_b.x
    v2 = feature_b.y

    r1 = Rmat[0, :]
    r2 = Rmat[1, :]
    r3 = Rmat[2, :]

    y = np.array([u1, v1, 1.0], dtype=float)

    x3_a = np.dot(r1 - u2 * r3, t) / np.dot(r1 - u2 * r3, y)
    x3_b = np.dot(r2 - v2 * r3, t) / np.dot(r2 - v2 * r3, y)
    x3 = np.mean([x3_a, x3_b])
    x1 = x3 * u1
    x2 = x3 * v1

    return np.array([x1, x2, x3], dtype=float)
