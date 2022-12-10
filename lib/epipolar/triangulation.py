import numpy as np
import numpy.typing as npt

from lib.common.feature import Feature

from ..transforms.transforms import Transform3D


def triangulate_point_correspondence(
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


def triangulate_points(
    features_a: list[Feature],
    features_b: list[Feature],
    intrinsic_camera_matrix: npt.NDArray[float],
    cam2_T_cam1: Transform3D,
) -> npt.NDArray[float]:
    if (3, 3) != intrinsic_camera_matrix.shape:
        raise ValueError(
            f"Camera intrinsic matrix is not 3x3, actual shape: {intrinsic_camera_matrix.shape}"
        )
    K_ext = np.hstack((intrinsic_camera_matrix, np.zeros((3, 1))))
    cam1_T_world = Transform3D.from_rmat_t(np.eye(3), np.zeros((3,)))
    cam2_T_world = cam2_T_cam1 @ cam1_T_world
    P1 = K_ext @ cam1_T_world.Tmat
    P2 = K_ext @ cam2_T_world.Tmat
    return np.array(
        [
            triangulate_point_correspondence(feature_a, feature_b, P1, P2)
            for feature_a, feature_b in zip(features_a, features_b)
        ]
    )
