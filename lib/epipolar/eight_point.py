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

    coords_a, coords_b, T1, T2 = _get_normalized_match_coordinates(
        features_a, features_b, matches
    )

    y = _get_y_mat(coords_a, coords_b)
    e_est = _compute_e_est(y)
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
    coords_a = np.empty((len(matches), 2), dtype=float)
    coords_b = np.empty((len(matches), 2), dtype=float)

    for index, match in enumerate(matches):
        feature_a = features_a[match.a_index]
        coords_a[index, :] = np.array([feature_a.x, feature_a.y])
        feature_b = features_b[match.b_index]
        coords_b[index, :] = np.array([feature_b.x, feature_b.y])

    return coords_a, coords_b


def _normalize_coords(
    coords_a: np.ndarray, coords_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize two lists of coordinates so that their mass centers are at the
    coordinate origin, and the average distance from the origin is sqrt(2).

    Return the inverse normalization transformation.

    :param coords_a: Coordnates from image A.
    :param coords_b: Corresponding coordinates from image B.
    :return: Tuple of [normalized coordinates A, normalized coordintes B, inverse transformation
    pt1, inverse transformation pt2.]
    """
    assert len(coords_a) == len(coords_b)

    centroid_a = np.mean(coords_a, axis=0)
    centroid_b = np.mean(coords_b, axis=0)

    centered_coords_a = coords_a - centroid_a
    centered_coords_b = coords_b - centroid_b

    centered_norms_a = np.linalg.norm(centered_coords_a, axis=1)
    scale_a = np.sum(centered_norms_a) / len(coords_a)
    scale_a = np.sqrt(2.0) / scale_a
    centered_norms_b = np.linalg.norm(centered_coords_b, axis=1)
    scale_b = np.sum(centered_norms_b) / len(coords_b)
    scale_b = np.sqrt(2.0) / scale_b

    normalized_coords_a = centered_coords_a * scale_a
    normalized_coords_b = centered_coords_b * scale_b

    # Compute the inverse transformations as well
    T1 = np.array(
        [
            [scale_a, 0.0, -scale_a * centroid_a[0]],
            [0.0, scale_a, -scale_a * centroid_a[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    T2 = np.array(
        [
            [scale_b, 0.0, -scale_b * centroid_b[0]],
            [0.0, scale_b, -scale_b * centroid_b[1]],
            [0.0, 0.0, 1.0],
        ]
    )

    return normalized_coords_a, normalized_coords_b, T1, T2


def _get_normalized_match_coordinates(
    features_a: List[Feature],
    features_b: List[Feature],
    matches: List[Match],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TODO update doc

    @param features_a List of features from the first image.
    @param features_b List of features from the second image.
    @param matches N matches between features_a and features_b.
    @return A pair of Nx2 matrices of normalized matching coordinates, one for features_b
    and one for features_a.
    """
    coords_a, coords_b = _get_matching_coordinates(features_a, features_b, matches)
    return _normalize_coords(coords_a, coords_b)


def _get_y_mat(coords_a: np.ndarray, coords_b: np.ndarray):
    assert len(coords_a) == len(coords_b) == 8

    y = np.empty((8, 9), dtype=float)

    for row_idx in range(len(coords_a)):
        y[row_idx, :] = _get_y_row(coords_a[row_idx, :], coords_b[row_idx, :])

    return y


def _get_y_row(coord_a: np.ndarray, coord_b: np.ndarray):
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
    assert (8, 9) == y.shape

    yTy = y.T @ y

    w, v = np.linalg.eig(yTy)
    min_index = np.argmin(np.abs(w))
    v_min = v[min_index]
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
    logging.info(f"Singular values of E_est: {s}")
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

    y = np.array([u1, v1, 1.0], dtype=float)

    x3_a = np.dot(r1 - u2 * r3, t) / np.dot(r1 - u2 * r3, y)
    x3_b = np.dot(r2 - v2 * r3, t) / np.dot(r2 - v2 * r3, y)
    x3 = np.mean([x3_a, x3_b])
    x1 = x3 * u1
    x2 = x3 * v1

    return np.array([x1, x2, x3], dtype=float)
