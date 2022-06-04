import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from lib.common.feature import Feature
from lib.feature_matching.matching import Match

_logger = logging.getLogger(Path(__file__).stem)


class EightPointCalculationError(Exception):
    """Raised if the computation cannot proceed due to ill-conditioned input data."""

    pass


def estimate_r_t(
    features_a: List[Feature],
    features_b: List[Feature],
    matches: List[Match],
):
    """Estimates rotation and translation up to a scale based on eight feature matches between two images.

    Uses the Eight-Point algorithm to estimate the Essential matrix, then decomposes the Essential matrix
    into rotation and translation up to a scale.

    @param features_a List of features from the first image.
    @param features_b List of features from the second image.
    @param matches Exactly eight matches between features_a and features_b.
    @return: Tuple of cam2_R_cam1, cam2_t_cam2_cam1.
    """
    if not features_a or not features_b:
        raise ValueError("Need some matching features")

    e = estimate_essential_mat(features_a, features_b, matches)

    feature_a = features_a[0]
    match = next(match for match in matches if match.a_index == 0)
    feature_b = features_b[match.b_index]

    r, t = recover_r_t(feature_a, feature_b, e)

    return r, t


def estimate_essential_mat(
    features_a: List[Feature],
    features_b: List[Feature],
    matches: List[Match],
) -> np.ndarray:
    """
    Estimates the Essential Matrix from eight point correspondences.

    @param features_a List of features from the first image.
    @param features_b List of features from the second image.
    @param matches Exactly eight matches between features_a and features_b.
    @return The 3x3 Essential Matrix (https://en.wikipedia.org/wiki/Essential_matrix).
    """
    if 8 != len(matches):
        raise ValueError("Exactly eight matches are needed")

    coords_a, T1, coords_b, T2 = _get_normalized_match_coordinates(
        features_a, features_b, matches
    )

    yT_y = _get_yT_y(coords_a, coords_b)
    e_est = _compute_e_est(yT_y)
    e = _enforce_essential_mat_constraints(e_est)

    # Apply the inverse transformation to un-normalize the essential matrix
    e = T2.T @ e @ T1

    # Enforce that the [2, 2] element == 1.0
    e /= e[2, 2]

    print(f"Final essential mat: {e}")

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
        raise EightPointCalculationError(
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
            cam2_t_cam2_x = x - t
            cam2_t_cam2_x = R.T @ cam2_t_cam2_x
            # TODO check if x is in front of both cameras

    return None, None


def _get_matching_coordinates(
    features_a: List[Feature], features_b: List[Feature], matches: List[Match]
) -> Tuple[np.ndarray, np.ndarray]:
    """Get two lists of matching image coordinates based on lists of features from
    both images and a list of matches.

    :param features_a: Features from image A.
    :param features_b: Features from image B.
    :param matches: List of matches between image A and B.
    :return: Two Nx2 matrices of corresponding image coordinates.
    """
    coords_a = np.empty((len(matches), 2), dtype=np.float64)
    coords_b = np.empty((len(matches), 2), dtype=np.float64)

    for index, match in enumerate(matches):
        feature_a = features_a[match.a_index]
        coords_a[index, :] = np.array([feature_a.x, feature_a.y], dtype=np.float64)
        feature_b = features_b[match.b_index]
        coords_b[index, :] = np.array([feature_b.x, feature_b.y], dtype=np.float64)

    return coords_a, coords_b


def _normalize_coords(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize two lists of coordinates so that their mass centers are at the
    coordinate origin, and the average distance from the origin is sqrt(2).

    Return the inverse normalization transformation.

    :param coords: Coordinates from an image.
    :return: Tuple of [normalized coordinates, inverse transformation].
    """

    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid

    centered_norms = np.linalg.norm(centered_coords, axis=1)
    scale = np.sqrt(2.0) / np.mean(centered_norms)

    normalized_coords = centered_coords * scale

    # Compute the inverse transformation as well.
    t = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return normalized_coords, t


def _get_normalized_match_coordinates(
    features_a: List[Feature],
    features_b: List[Feature],
    matches: List[Match],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get two sets of corresponding coordinates, normalized so that their mass centers are at the
    coordinate origin, and the average distance from the origin is sqrt(2).

    Also compute the inverse of the normalization transformations.

    @param features_a List of features from the first image.
    @param features_b List of features from the second image.
    @param matches N matches between features_a and features_b.
    @return Tuple of [normalized coords a (Nx2 matrix), inverse transformation T1,
        corresponding normalized coords b (Nx2) matrix, inverse transformation T2]
    """
    coords_a, coords_b = _get_matching_coordinates(features_a, features_b, matches)
    return (*_normalize_coords(coords_a), *_normalize_coords(coords_b))


def _get_yT_y(coords_a: np.ndarray, coords_b: np.ndarray):
    """Compute y.T @ y, from which the essential matrix can be recovered."""
    assert len(coords_a) == len(coords_b) == 8

    yT_y = np.zeros((9, 9), dtype=np.float64)

    for col_idx in range(len(coords_a)):
        col = _get_y_col(coords_a[col_idx, :], coords_b[col_idx, :])
        yT_y += np.outer(col, col)

    _logger.debug(f"yT_y:\n{yT_y}")

    return yT_y


def _get_y_col(coord_a: np.ndarray, coord_b: np.ndarray):
    assert 2 == len(coord_a)
    assert 2 == len(coord_b)

    y_col = np.empty((9,), dtype=np.float64)
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


def _compute_e_est(yT_y: np.ndarray) -> np.ndarray:
    """
    Estimates the essential matrix from the Y matrix.
    (https://en.wikipedia.org/wiki/Eight-point_algorithm#Step_2:_Solving_the_equation)

    @param yT_y The Y matrix, constructed from eight matching normalized coordinate pairs.
    @return An estimate of the Essential matrix, not necessarily of rank 2.
    @raise EightPointCalculationError If the Essential matrix cannot be estimated.
    """
    assert (9, 9) == yT_y.shape

    w, v = np.linalg.eig(yT_y)
    _logger.debug(f"Eigenvalues of Y.T @ Y:\n{w}")
    _logger.debug(f"Eigenvectors of Y.T @ Y:\n{v}")

    # Check that only the last eigenvalue is very small.
    VERY_SMALL = 1e-10
    sorted_w = np.sort(w)
    if np.any(sorted_w[1:] <= VERY_SMALL):
        raise EightPointCalculationError(
            "More than one eigenvalue of Y.T @ Y is small. Cannot confidently estimate"
            " essential matrix."
        )

    min_index = np.argmin(np.abs(w))
    v_min = v[:, min_index]
    e_est = v_min.reshape((3, 3))

    return e_est


def _enforce_essential_mat_constraints(e_est: np.ndarray) -> np.ndarray:
    """
    Creates E' from the estimated E matrix, enforcing that rank(E') == 2.
    (https://en.wikipedia.org/wiki/Eight-point_algorithm#Step_3:_Enforcing_the_internal_constraint)

    @param e_est Estimated Essential matrix.
    @return Essential matrix fulfilling the internal constraints.
    """
    u, s, vh = np.linalg.svd(e_est)
    _logger.debug(f"Singular values of E_est: {s}")
    # Set the smallest singular value of E_est to 0
    s[2] = 0.0
    s_prime = np.diag(s)
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

    y = np.array([u1, v1, 1.0], dtype=np.float64)

    x3_a = np.dot(r1 - u2 * r3, t) / np.dot(r1 - u2 * r3, y)
    x3_b = np.dot(r2 - v2 * r3, t) / np.dot(r2 - v2 * r3, y)
    x3 = np.mean([x3_a, x3_b])
    x1 = x3 * u1
    x2 = x3 * v1

    return np.array([x1, x2, x3], dtype=np.float64)
