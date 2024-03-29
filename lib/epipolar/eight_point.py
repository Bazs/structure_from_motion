"""Functions which estimate the Fundamental and Essential matrix, as well as recover rotation and translation based on
point correspondeces between different camera poses."""
import itertools
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from lib.common.feature import Feature
from lib.epipolar.triangulation import triangulate_point_correspondence
from lib.feature_matching.matching import Match
from lib.transforms.transforms import Transform3D

_logger = logging.getLogger(Path(__file__).stem)


class EightPointCalculationError(Exception):
    """Raised if the computation cannot proceed due to ill-conditioned input data."""

    pass


def estimate_r_t(
    camera_matrix: npt.NDArray,
    features_a: List[Feature],
    features_b: List[Feature],
    matches: List[Match],
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Estimate rotation and translation up to a scale based on eight feature matches between two images.

    Uses the Eight-Point algorithm to estimate the Essential matrix, then decomposes the Essential matrix
    into rotation and translation up to a scale.

    Args:
        camera_matrix: The camera intrinsic matrix.
        features_a: List of features from the first image.
        features_b: List of features from the second image.
        matches: Exactly eight matches between features_a and features_b.
    Returns:
        Tuple of cam2_R_cam1, cam2_t_cam2_cam1.
    """
    if not features_a or not features_b:
        raise ValueError("Need some matching features")

    e = estimate_essential_mat(
        camera_matrix=camera_matrix,
        features_a=features_a,
        features_b=features_b,
        matches=matches,
    )

    features_a_in_match_order = [features_a[match.a_index] for match in matches]
    features_b_in_match_order = [features_b[match.b_index] for match in matches]
    return recover_r_t_from_e(
        e=e,
        camera_matrix=camera_matrix,
        features_a=features_a_in_match_order,
        features_b=features_b_in_match_order,
    )


def recover_r_t_from_e(
    e: npt.NDArray,
    camera_matrix: npt.NDArray,
    features_a: list[Feature],
    features_b: list[Feature],
    distance_threshold: float | None = None,
):
    """Recover the rotation and translation up to a scale from an Essential matrix.

    The solution with the most amount of corresponding points passing the cheirality check
    is returned.

    Args:
        e: essential matrix.
        camera_matrix: Intrinsic camera matrix.
        features_a: Feature points from the first image.
        features_b: Corresponding feature points from the second image.
        distance_threshold: Threshold for filtering out far away points, i.e. points at infinity.
    Returns:
        Tuple of (cam2_R_cam1, cam2_t_cam2_cam1, mask). Mask is an array containing indices into
            features_a and features_b which passed the cheirality check.
    """
    features_a = [
        to_normalized_image_coords(feature, camera_matrix) for feature in features_a
    ]
    features_b = [
        to_normalized_image_coords(feature, camera_matrix) for feature in features_b
    ]

    r, t, mask = _recover_r_t(features_a, features_b, e, distance_threshold)

    return r, t, mask


def estimate_essential_mat(
    *,
    camera_matrix: npt.NDArray,
    features_a: List[Feature],
    features_b: List[Feature],
    matches: List[Match],
):
    """Estimate the Essential Matrix from eight point correspondences and camera intrinsic matrix.

    Args:
        camera_matrix: The camera intrinsic matrix.
        features_a: List of features from the first image.
        features_b: List of features from the second image.
        matches: Exactly eight matches between features_a and features_b.
    Returns:
        The 3x3 Essential Matrix (https://en.wikipedia.org/wiki/Essential_matrix)."""
    features_a = [
        to_normalized_image_coords(feature, camera_matrix) for feature in features_a
    ]
    features_b = [
        to_normalized_image_coords(feature, camera_matrix) for feature in features_b
    ]
    e = estimate_fundamental_mat(
        features_a=features_a, features_b=features_b, matches=matches
    )
    return e


def to_normalized_image_coords(feature: Feature, camera_matrix: npt.NDArray) -> Feature:
    """Calculate the normalized image coordinates from pixel coordinates and the intrinsic camera parameters."""
    f_x = camera_matrix[0][0]
    f_y = camera_matrix[1][1]
    c_x = camera_matrix[0][2]
    c_y = camera_matrix[1][2]
    return Feature(x=(feature.x - c_x) / f_x, y=(feature.y - c_y) / f_y)


def estimate_fundamental_mat(
    features_a: List[Feature],
    features_b: List[Feature],
    matches: List[Match],
) -> np.ndarray:
    """
    Estimate the Fundamental Matrix from eight point correspondences using the Eight-point algorithm.

    Args:
        features_a: List of features from the first image.
        features_b: List of features from the second image.
        matches: Exactly eight matches between features_a and features_b.
    Returns:
        The 3x3 Fundamental Matrix (https://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)).
    """
    if 8 != len(matches):
        raise ValueError("Exactly eight matches are needed")

    coords_a, T1, coords_b, T2 = _get_normalized_match_coordinates(
        features_a, features_b, matches
    )

    yT_y = _get_yT_y(coords_a, coords_b)
    e_est = _compute_f_est(yT_y)
    e = _enforce_fundamental_mat_constraints(e_est)

    # Apply the inverse transformation to un-normalize the fundamental matrix
    e = T2.T @ e @ T1

    # Enforce that the [2, 2] element == 1.0
    e /= e[2, 2]

    _logger.debug(f"Final fundamental mat: {e}")

    return e


def create_trivial_matches(num_features: int) -> list[Match]:
    """Create a list num_features matches, where a_index and b_index equals to the list index for each match."""
    return [
        Match(a_index=index, b_index=index, match_score=0.0)
        for index in range(num_features)
    ]


def _recover_r_t(
    features_a: list[Feature],
    features_b: list[Feature],
    e: np.ndarray,
    distance_threshold: float | None = None,
):
    """Recover the rotation and translation up to a scale from an Essential matrix.

    The solution with the most amount of corresponding points passing the cheirality check
    is returned.

    Args:
        features_a: Feature points from the first image. Must be in normalized image coordinates.
        features_b: Corresponding feature points from the second image. Must be in normalized image coordinates.
        e: essential matrix.
        distance_threshold: Threshold for filtering out far away points, i.e. points at infinity.
    Returns:
        Tuple of (cam2_R_cam1, cam2_t_cam2_cam1, mask). Mask is an array containing indices into
            features_a and features_b which passed the cheirality check.
    """
    # Number of correspondences passing the cheirality check per possible solution.
    num_good_correspondences = []
    # Rotation matrices of possible solutions in the same order as num_good_correspondences.
    rot_mats = []
    # Translation vectors of possible solutions in the same order as num_good_correspondences.
    translations = []
    passing_correspondence_indices_per_solution = []

    cam2_R_cam1_1, cam2_R_cam1_2, cam2_t_cam2_cam1_1 = _recover_all_r_t(e)
    for cam2_R_cam1, cam2_t_cam2_cam1 in itertools.product(
        [cam2_R_cam1_1, cam2_R_cam1_2], [cam2_t_cam2_cam1_1, -cam2_t_cam2_cam1_1]
    ):
        passing_correspondence_indices = np.nonzero(
            [
                _cheirality_check(
                    feature_a,
                    feature_b,
                    cam2_R_cam1,
                    cam2_t_cam2_cam1,
                    distance_threshold=distance_threshold,
                )
                for feature_a, feature_b in zip(features_a, features_b)
            ]
        )[0]
        passing_correspondence_indices_per_solution.append(
            passing_correspondence_indices
        )
        num_good_correspondences.append(
            np.count_nonzero(passing_correspondence_indices)
        )
        rot_mats.append(cam2_R_cam1)
        translations.append(cam2_t_cam2_cam1)
    if 0 == np.count_nonzero(num_good_correspondences):
        raise EightPointCalculationError(
            "None of the transformations pass the cheirality check."
        )
    best_solution_index = np.argmax(num_good_correspondences)
    return (
        rot_mats[best_solution_index],
        translations[best_solution_index],
        passing_correspondence_indices_per_solution[best_solution_index],
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


def _compute_f_est(yT_y: np.ndarray) -> np.ndarray:
    """
    Estimates the fundamental matrix from the Y matrix.
    (https://en.wikipedia.org/wiki/Eight-point_algorithm#Step_2:_Solving_the_equation)

    Args:
        yT_y The Y matrix, constructed from eight matching normalized coordinate pairs.
    Returns:
        An estimate of the fundamental matrix, not necessarily of rank 2.
    Raises:
        EightPointCalculationError If the fundamental matrix cannot be estimated.
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
            " fundamental matrix."
        )

    min_index = np.argmin(np.abs(w))
    v_min = v[:, min_index]
    f_est = v_min.reshape((3, 3))

    return f_est


def _enforce_fundamental_mat_constraints(f_est: np.ndarray) -> np.ndarray:
    """
    Create F' from the estimated F matrix, enforcing that rank(F') == 2.
    (https://en.wikipedia.org/wiki/Eight-point_algorithm#Step_3:_Enforcing_the_internal_constraint)

    Args:
        e_est Estimated Fundamental Matrix.
    Returns:
        Fundamental Matrix fulfilling the internal constraints.
    """
    u, s, vh = np.linalg.svd(f_est)
    _logger.debug(f"Singular values of F_est: {s}")
    # Set the smallest singular value of E_est to 0
    s[2] = 0.0
    s_prime = np.diag(s)
    f = u @ s_prime @ vh
    return f


def _cheirality_check(
    feature_a: Feature,
    feature_b: Feature,
    cam2_R_cam1: npt.NDArray,
    cam2_t_cam2_cam1: npt.NDArray,
    distance_threshold: float | None = None,
    z_axis_index: int = 2,
) -> bool:
    """Return whether the transformation between the two camera poses places the feature in front both of the
    cameras.

    Args:
        feature_a: Point in image 1.
        feature_b: Corresponding point in image 2.
        cam2_r_cam1: Rotation matrix rotating camera 1's frame into camera 2's frame.
        cam2_t_cam2_cam1: Translation vector from camera 2's frame into camera 1's frame.
        distance_threshold: If the triangulated point is farther away than this threshold from the
            camera, then the check is considered failed.
        z_axis_index: The axis of the triangulated point which is the one pointing outward from the camera frame.
    """
    if distance_threshold is None:
        distance_threshold = 50.0

    # Place the first camera into the world frame's origin.
    P1 = Transform3D.identity().Tmat
    P2 = Transform3D.from_rmat_t(cam2_R_cam1, cam2_t_cam2_cam1).Tmat
    cam1_t_cam1_feature = triangulate_point_correspondence(feature_a, feature_b, P1, P2)
    cam2_t_cam2_feature = (P2 @ [*cam1_t_cam1_feature, 1])[:-1]
    TOLERANCE = 1e-8
    if (
        np.all(
            np.array(
                [cam1_t_cam1_feature[z_axis_index], cam2_t_cam2_feature[z_axis_index]]
            )
            >= (0 - TOLERANCE)
        )
        and np.linalg.norm(cam1_t_cam1_feature) <= distance_threshold
    ):
        return True
    return False
