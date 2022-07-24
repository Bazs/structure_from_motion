import itertools
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from lib.common.feature import Feature
from lib.feature_matching.matching import Match
from lib.transforms.transforms import Transform3D

_logger = logging.getLogger(Path(__file__).stem)


class EightPointCalculationError(Exception):
    """Raised if the computation cannot proceed due to ill-conditioned input data."""

    pass


def estimate_r_t(
    features_a: List[Feature],
    features_b: List[Feature],
    matches: List[Match],
):
    """Estimate rotation and translation up to a scale based on eight feature matches between two images.

    Uses the Eight-Point algorithm to estimate the Essential matrix, then decomposes the Essential matrix
    into rotation and translation up to a scale.

    Args:
        features_a: List of features from the first image.
        features_b: List of features from the second image.
        matches: Exactly eight matches between features_a and features_b.
    Returns:
        Tuple of cam2_R_cam1, cam2_t_cam2_cam1.
    """
    if not features_a or not features_b:
        raise ValueError("Need some matching features")

    e = estimate_essential_mat(features_a, features_b, matches)

    feature_a = features_a[0]
    match = next(match for match in matches if match.a_index == 0)
    feature_b = features_b[match.b_index]

    r, t = _recover_all_r_t(feature_a, feature_b, e)

    return r, t


def estimate_essential_mat(
    features_a: List[Feature],
    features_b: List[Feature],
    matches: List[Match],
) -> np.ndarray:
    """
    Estimate the Essential Matrix from eight point correspondences.

    Args:
        features_a: List of features from the first image.
        features_b: List of features from the second image.
        matches: Exactly eight matches between features_a and features_b.
    Returns:
        The 3x3 Essential Matrix (https://en.wikipedia.org/wiki/Essential_matrix).
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

    _logger.info(f"Final essential mat: {e}")

    return e


def _recover_r_t(feature_a: Feature, feature_b: Feature, e: np.ndarray):
    """Recover the rotation and translation up to a scale from an Essential matrix.

    Args:
        feature_a: A feature point from the first image.
        feature_b: Corresponding feature point from the second image.
        e: essential matrix.
    Returns:
        Tuple of (cam2_R_cam1, cam2_t_cam2_cam1)
    """
    cam2_R_cam1_1, cam2_R_cam1_2, cam2_t_cam2_cam1_1 = _recover_all_r_t(e)
    for cam2_R_cam1, cam2_t_cam2_cam1 in itertools.product(
        [cam2_R_cam1_1, cam2_R_cam1_2], [cam2_t_cam2_cam1_1, -cam2_t_cam2_cam1_1]
    ):
        if _cheirality_check(feature_a, feature_b, cam2_R_cam1, cam2_t_cam2_cam1):
            return cam2_R_cam1, cam2_t_cam2_cam1
    raise EightPointCalculationError(
        "None of the transformations pass the cheirality check."
    )


def _recover_all_r_t(e: np.ndarray) -> Tuple[Rotation, np.ndarray]:
    """Recover two possible rotations and one of the possible translations up to a scale from an Essential matrix.

    Args:
        e: essential matrix.
    Returns:
        Tuple of (cam2_R_cam1_1, cam2_R_cam1_2, cam2_t_cam2_cam1_1)
    """
    u, s, vh = np.linalg.svd(e)

    det_u = np.linalg.det(u)
    det_vh = np.linalg.det(vh)
    if not np.isclose(abs(det_u), 1):
        raise EightPointCalculationError("U is not a rotation matrix")
    if not np.isclose(abs(det_vh), 1):
        raise EightPointCalculationError("V_h is not a rotation matrix")

    # Make rotation matrix sub-components proper if they are improper
    if np.isclose(-1, det_u):
        u *= -1
    if np.isclose(-1, det_vh):
        vh *= -1

    if not np.isclose(0.0, s[-1]):
        raise EightPointCalculationError(
            "The smallest singular value of the Essential matrix is expected to be ~0"
        )

    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], dtype=float)
    t_x = u @ z @ u.T
    t_1 = np.array([-t_x[1, 2], t_x[0, 2], -t_x[0, 1]])
    R_1 = u @ w.T @ vh
    R_2 = u @ w @ vh

    return R_1, R_2, t_1


def _get_matching_coordinates(
    features_a: List[Feature], features_b: List[Feature], matches: List[Match]
) -> Tuple[np.ndarray, np.ndarray]:
    """Get two lists of matching image coordinates based on lists of features from
    both images and a list of matches.

    Args:
       features_a: Features from image A.
       features_b: Features from image B.
       matches: List of matches between image A and B.
    Returns:
        Two Nx2 matrices of corresponding image coordinates.
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

    Args:
        coords: Coordinates from an image.
    Returns:
        Tuple of [normalized coordinates, inverse transformation].
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

    Args:
        features_a List of features from the first image.
        features_b List of features from the second image.
        matches N matches between features_a and features_b.
    Returns:
        Tuple of [normalized coords a (Nx2 matrix), inverse transformation T1,
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

    Args:
        yT_y The Y matrix, constructed from eight matching normalized coordinate pairs.
    Returns:
        An estimate of the Essential matrix, not necessarily of rank 2.
    Raises:
        EightPointCalculationError If the Essential matrix cannot be estimated.
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
    Create E' from the estimated E matrix, enforcing that rank(E') == 2.
    (https://en.wikipedia.org/wiki/Eight-point_algorithm#Step_3:_Enforcing_the_internal_constraint)

    Args:
        e_est Estimated Essential matrix.
    Returns:
        Essential matrix fulfilling the internal constraints.
    """
    u, s, vh = np.linalg.svd(e_est)
    _logger.debug(f"Singular values of E_est: {s}")
    # Set the smallest singular value of E_est to 0
    s[2] = 0.0
    s_prime = np.diag(s)
    e = u @ s_prime @ vh
    return e


def _triangulate(
    feature_a: Feature, feature_b: Feature, P1: npt.NDArray, P2: npt.NDArray
) -> npt.NDArray:
    """Triangulate the 3D world position of the point corresponding to the matching
    feature pair in two different camera poses.

    Args:
        feature_a: The projection of the world point onto the first camera.
        feature_b: The projection of the world point onto the second camera.
        P1: The intrinsic+extrinsic 3x4 camera matrix of the first camera.
        P2: The intrinsic+extrinsic 3x4 camera matrix of the second camera.
    Returns:
        The 3D position of the point as a vector.
    """
    A = np.array(
        [
            [feature_a.y * P1[2, :] - P1[1, :]],
            [P1[0, :] - feature_a.x * P1[2, :]],
            [feature_b.y * P2[2, :] - P2[1, :]],
            [P2[0, :] - feature_b.x * P2[2, :]],
        ]
    ).squeeze()

    # The best estimate for the 3D position is the right singular vector of A corresponding to the smallest singular
    # value.
    _, _, vh = np.linalg.svd(A)
    x = vh[-1, :]

    # Convert back from homogeneous coordinates.
    x = (x / x[-1])[:-1]
    return x


def _cheirality_check(
    feature_a: Feature,
    feature_b: Feature,
    cam2_R_cam1: npt.NDArray,
    cam2_t_cam2_cam1: npt.NDArray,
    z_axis_index: int = 2,
) -> bool:
    """Return whether the transformation between the two camera poses places the feature in front both of the
    cameras."""
    # Place the first camera into the world frame's origin.
    P1 = Transform3D.from_rmat_t(
        np.eye(3, dtype=float), np.zeros((3,), dtype=float)
    ).Tmat
    P2 = Transform3D.from_rmat_t(cam2_R_cam1, cam2_t_cam2_cam1).Tmat
    cam1_t_cam1_feature = _triangulate(feature_a, feature_b, P1, P2)
    cam2_t_cam2_feature = (P2 @ [*cam1_t_cam1_feature, 1])[:-1]
    TOLERANCE = 1e-8
    if np.all(
        np.array([cam1_t_cam1_feature[z_axis_index], cam2_t_cam2_feature[z_axis_index]])
        >= (0 - TOLERANCE)
    ):
        return True
    return False
