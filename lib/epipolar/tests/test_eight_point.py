import logging
from math import dist

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
from scipy.spatial.transform import Rotation

from lib.common.feature import Feature
from lib.epipolar import eight_point
from lib.feature_matching.matching import Match
from lib.transforms.transforms import Transform3D

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(filename)s:%(lineno)s\t %(message)s",
)


_CAM_WIDTH_PX = 512
_CAM_HEIGHT_PX = 256


def test_get_matching_coordinates():
    features_a = [Feature(256, 128), Feature(128, 64)]
    features_b = [Feature(32, 64), Feature(16, 8)]
    matches = [Match(0, 1), Match(1, 0)]

    coords_a, coords_b = eight_point._get_matching_coordinates(
        features_a, features_b, matches
    )

    expected_coords_a = np.array(
        [[features_a[0].x, features_a[0].y], [features_a[1].x, features_a[1].y]]
    )
    expected_coords_b = np.array(
        [[features_b[1].x, features_b[1].y], [features_b[0].x, features_b[0].y]]
    )
    np.testing.assert_allclose(expected_coords_a, coords_a)
    np.testing.assert_allclose(expected_coords_b, coords_b)


def test_normalize_coords():
    input_coords = np.array([[10, 10], [15, 10], [5, 10]])
    normalized_coords, t = eight_point._normalize_coords(input_coords)
    expected_distance = np.sqrt(2.0) * 3.0 / 2.0
    expected_coords = np.array(
        [[0, 0], [expected_distance, 0], [-expected_distance, 0]]
    )
    np.testing.assert_allclose(expected_coords, normalized_coords)

    normalized_coords_homo = np.hstack([normalized_coords, np.ones((3, 1))])
    unnormalized_coords_homo = normalized_coords_homo @ np.linalg.inv(t).T
    unnormalized_coords = unnormalized_coords_homo[:, :-1]
    np.testing.assert_allclose(input_coords, unnormalized_coords)


def test_get_y_col():
    coord_a = np.array([2.0, 3.0])
    coord_b = np.array([7.0, 6.0])

    y_col = eight_point._get_y_col(coord_a, coord_b)
    assert (9,) == y_col.shape

    expected_y_col = np.array(
        [
            7.0 * 2.0,
            7.0 * 3.0,
            7.0,
            6.0 * 2.0,
            6.0 * 3.0,
            6.0,
            2.0,
            3.0,
            1.0,
        ]
    )

    np.testing.assert_allclose(expected_y_col, y_col)


@pytest.fixture
def camera_intrinsic_matrix() -> npt.NDArray:
    return _create_camera_matrix(
        f=100.0, cam_width_px=_CAM_WIDTH_PX, cam_height_px=_CAM_HEIGHT_PX
    )


def test_estimate_essential_matrix(camera_intrinsic_matrix):
    rng = np.random.default_rng(seed=5)
    NUM_POINTS = 8
    world_t_world_points = rng.random((NUM_POINTS, 3), dtype=np.float64)

    fig = plt.figure()
    ax_3d = fig.add_subplot(131, projection="3d")
    _plot_world_points(ax_3d, world_t_world_points)

    K = camera_intrinsic_matrix

    world_t_world_camera1 = np.array([1.5, 0.25, -1.0])
    camera1_Rmat_world = np.eye(3, dtype=float)
    camera1_Rvec_world, _ = cv.Rodrigues(camera1_Rmat_world)
    world_T_world_camera1 = Transform3D.from_rmat_t(
        camera1_Rmat_world, world_t_world_camera1
    )

    world_t_world_camera2 = np.array([2.5, 0.1, -1.5])
    camera2_R_world = Rotation.from_euler("XY", [-20.0, -50.0], degrees=True)
    camera2_Rmat_world = camera2_R_world.as_matrix()
    camera2_Rvec_world, _ = cv.Rodrigues(camera2_Rmat_world)
    world_T_world_camera2 = Transform3D.from_rmat_t(
        camera2_Rmat_world, world_t_world_camera2
    )

    world_T_camera2_camera1 = world_T_world_camera2.inv() * world_T_world_camera1
    print(world_T_camera2_camera1)

    _plot_camera(ax_3d, world_t_world_camera1, camera1_Rmat_world)
    _plot_camera(ax_3d, world_t_world_camera2, camera2_Rmat_world)

    cam1_points, _ = cv.projectPoints(
        world_t_world_points,
        camera1_Rvec_world,
        -world_t_world_camera1,
        K,
        None,
    )
    cam1_points = cam1_points.squeeze()
    cam2_points, _ = cv.projectPoints(
        world_t_world_points,
        camera2_Rvec_world,
        -world_t_world_camera2,
        K,
        None,
    )
    cam2_points = cam2_points.squeeze()

    cam1_ax = fig.add_subplot(132)
    _plot_camera_points(
        cam1_ax, cam1_points, _CAM_WIDTH_PX, _CAM_HEIGHT_PX, mode="scatter"
    )
    cam2_ax = fig.add_subplot(133)
    _plot_camera_points(
        cam2_ax, cam2_points, _CAM_WIDTH_PX, _CAM_HEIGHT_PX, mode="scatter"
    )

    # plt.show()

    features_1 = [Feature(x=point[0], y=point[1]) for point in cam1_points]
    features_2 = [Feature(x=point[0], y=point[1]) for point in cam2_points]
    matches = [
        Match(a_index=index, b_index=index, match_score=0.0)
        for index in range(len(features_1))
    ]
    e = eight_point.estimate_essential_mat(
        features_1,
        features_2,
        matches,
    )

    coords_a, coords_b = eight_point._get_matching_coordinates(
        features_1, features_2, matches
    )
    np.set_printoptions(precision=20)
    logging.info(
        f"Input coords for findFundamenalMat:\ncoords_a:\n{coords_a}\ncoords_b:\n{coords_b}"
    )
    e_cv, _ = cv.findFundamentalMat(coords_a, coords_b, method=cv.FM_8POINT)
    logging.info(f"OpenCV estimated essential mat: {e_cv}")

    np.testing.assert_almost_equal(e_cv, e, decimal=5)

    R1_cv, R2_cv, t_cv = cv.decomposeEssentialMat(e_cv)
    t_cv = np.squeeze(t_cv)
    R1, R2, t = eight_point._recover_all_r_t(e)

    def _allclose(expected: npt.NDArray, actual: npt.NDArray):
        ABSOLUTE_TOLERANCE = 1e-4
        return np.allclose(expected, actual, atol=ABSOLUTE_TOLERANCE)

    if not _allclose(t_cv, t):
        if not _allclose(-t_cv, t):
            pytest.fail(f"Incorrect translation, expected:\n{t_cv}\nactual:\n{t}")

    rotmat_comparison_failure_message = (
        f"Incorrect estimates for rotation matrices. Expected:\n{R1_cv}\nand\n{R2_cv}\n"
        f"Actual:\n{R1}\nand\n{R2}"
    )
    if _allclose(R1_cv, R1):
        if not _allclose(R2_cv, R2):
            pytest.fail(rotmat_comparison_failure_message)
    elif _allclose(R2_cv, R1):
        if not _allclose(R1_cv, R2):
            pytest.fail(rotmat_comparison_failure_message)
    else:
        pytest.fail(rotmat_comparison_failure_message)

    # eight_point._recover_r_t(features_1[0], features_2[0], e)


def test_estimate_essential_matrix_degenerate():
    rectangle_width = np.array([1.0, 0.0, 0.0])
    rectangle_height = np.array([0.0, 0.5, 0.0])
    rectangle_origin = np.zeros((3,), dtype=float)

    world_t_world_rectangle_a = np.array(
        [
            rectangle_origin,
            rectangle_origin + rectangle_width,
            rectangle_origin + rectangle_width + rectangle_height,
            rectangle_origin + rectangle_height,
        ]
    )

    world_t_world_rectangle_b = world_t_world_rectangle_a.copy()
    world_t_world_rectangleBOffset = np.array([2.0, 0.0, 0.0])
    world_t_world_rectangle_b += world_t_world_rectangleBOffset

    rect_a_rot = Rotation.from_euler("y", -40.0, degrees=True)
    world_t_world_rectangle_a = _rotate_rectangle(world_t_world_rectangle_a, rect_a_rot)

    rect_b_rot = Rotation.from_euler("y", 40.0, degrees=True)
    world_t_world_rectangle_b = _rotate_rectangle(world_t_world_rectangle_b, rect_b_rot)

    fig = plt.figure()
    ax_3d = fig.add_subplot(131, projection="3d")
    _plot_rectangle(ax_3d, world_t_world_rectangle_a)
    _plot_rectangle(ax_3d, world_t_world_rectangle_b)

    world_t_world_camera1 = np.array([1.5, 0.25, -1.0])
    camera1_Rmat_world = np.eye(3, dtype=float)
    camera1_Rvec_world, _ = cv.Rodrigues(camera1_Rmat_world)

    world_t_world_camera2 = np.array([2.5, 0.1, -1.5])
    camera2_R_world = Rotation.from_euler("XY", [-20.0, -50.0], degrees=True)
    camera2_Rmat_world = camera2_R_world.as_matrix()
    camera2_Rvec_world, _ = cv.Rodrigues(camera2_Rmat_world)

    _plot_camera(ax_3d, world_t_world_camera1, camera1_Rmat_world)
    _plot_camera(ax_3d, world_t_world_camera2, camera2_Rmat_world)

    cam_width_px = 512
    cam_height_px = 256

    # Calculate the focal length in pixel values, so that the angle of view is guaranteed
    fov_deg = 100
    base_angle_deg = (180 - fov_deg) / 2.0
    base_angle_rad = np.radians(base_angle_deg)
    f_x = np.tan(base_angle_rad) * cam_width_px / 2
    f_y = np.tan(base_angle_rad) * cam_height_px / 2
    # Choose the smallest focal length to ensure a minimum of guaranteed deg FOV with a
    # square pixel size
    f = min(f_x, f_y)

    # Build the camera intrinsic matrix
    K = _create_camera_matrix(f, cam_width_px, cam_height_px)

    world_t_world_allPoints = np.vstack(
        [world_t_world_rectangle_a, world_t_world_rectangle_b]
    )

    cam1_points, _ = cv.projectPoints(
        world_t_world_allPoints,
        camera1_Rvec_world,
        -world_t_world_camera1,
        K,
        None,
    )
    cam1_points = cam1_points.squeeze()
    cam1_ax = fig.add_subplot(132)
    _plot_camera_points(cam1_ax, cam1_points, cam_width_px, cam_height_px)

    cam2_points, _ = cv.projectPoints(
        world_t_world_allPoints,
        camera2_Rvec_world,
        -world_t_world_camera2,
        K,
        None,
    )
    cam2_points = cam2_points.squeeze()
    cam2_ax = fig.add_subplot(133)
    _plot_camera_points(cam2_ax, cam2_points, cam_width_px, cam_height_px)
    # plt.show()

    features_1 = [Feature(x=point[0], y=point[1]) for point in cam1_points]
    features_2 = [Feature(x=point[0], y=point[1]) for point in cam2_points]
    matches = [
        Match(a_index=index, b_index=index, match_score=0.0)
        for index in range(len(features_1))
    ]
    with pytest.raises(eight_point.EightPointCalculationError):
        e = eight_point.estimate_essential_mat(
            features_1,
            features_2,
            matches,
        )


def test_triangulate(camera_intrinsic_matrix):
    world_t_world_point = np.array([0.0, 0.0, 10.0])

    fig = plt.figure()
    ax_3d = fig.add_subplot(131, projection="3d")
    _plot_world_points(ax_3d, world_t_world_point.reshape((1, 3)))

    K = camera_intrinsic_matrix

    world_t_world_cam1 = np.array([0.0, 0.0, 5.0])
    world_R_cam1 = np.eye(3, dtype=float)
    cam1_Rvec_world, _ = cv.Rodrigues(world_R_cam1.T)
    cam1_T_world = Transform3D.from_rmat_t(world_R_cam1.T, -world_t_world_cam1).Tmat

    _plot_camera(ax_3d, world_t_world_cam1, world_R_cam1.T)

    world_t_world_cam2 = np.array([3.0, 0.0, 5.0])
    world_R_cam2 = Rotation.from_euler(
        "XYZ", [0.0, 30.0, 0.0], degrees=True
    ).as_matrix()
    cam2_Rvec_world, _ = cv.Rodrigues(world_R_cam2.T)
    cam2_T_world = Transform3D.from_rmat_t(world_R_cam2.T, -world_t_world_cam2).Tmat
    _plot_camera(ax_3d, world_t_world_cam2, world_R_cam2.T)

    cam1_point, _ = cv.projectPoints(
        world_t_world_point.reshape((1, 3)),
        cam1_Rvec_world,
        -world_t_world_cam1,
        K,
        distCoeffs=None,
    )
    cam1_point = cam1_point.reshape((1, 2))
    cam1_ax = fig.add_subplot(132)
    _plot_camera_points(
        cam1_ax,
        cam1_point,
        cam_width=_CAM_WIDTH_PX,
        cam_height=_CAM_HEIGHT_PX,
        mode="scatter",
    )
    cam2_point, _ = cv.projectPoints(
        world_t_world_point.reshape((1, 3)),
        cam2_Rvec_world,
        -world_t_world_cam2,
        K,
        distCoeffs=None,
    )
    cam2_point = cam2_point.reshape((1, 2))
    cam2_ax = fig.add_subplot(133)
    _plot_camera_points(
        cam2_ax,
        cam2_point,
        cam_width=_CAM_WIDTH_PX,
        cam_height=_CAM_HEIGHT_PX,
        mode="scatter",
    )

    feature_a = Feature(x=cam1_point[0, 0], y=cam1_point[0, 1])
    feature_b = Feature(x=cam2_point[0, 0], y=cam2_point[0, 1])

    # Pad the intrinsic matrix and calculate the intrinsic+extrinsic camera matrices.
    K_ext = np.hstack((K, np.zeros((3, 1))))
    P1 = K_ext @ cam1_T_world
    P2 = K_ext @ cam2_T_world

    # Sanity check that P1 is the correct camera matrix by projecting the point
    # with it and cross referencing the OpenCV projection result.
    cam1_point_check = P1 @ np.vstack(
        (world_t_world_point.reshape((3, 1)), np.array([[1.0]]))
    )
    cam1_point_check = (cam1_point_check / cam1_point_check[2])[:-1]
    np.testing.assert_almost_equal(cam1_point.reshape(-1), cam1_point_check.reshape(-1))

    world_t_world_point_estimated = eight_point._triangulate(
        feature_a, feature_b, P1, P2
    )
    np.testing.assert_allclose(
        world_t_world_point, world_t_world_point_estimated, atol=1e-10, rtol=0
    )

    # plt.show()


def _rotate_rectangle(
    world_t_world_rectangle: np.ndarray, rotation: Rotation
) -> np.ndarray:
    centroid = np.mean(world_t_world_rectangle, axis=0)
    rectangle_t_rectangle_rectangle = world_t_world_rectangle - centroid
    rotated_rectangle = rotation.apply(rectangle_t_rectangle_rectangle)
    return rotated_rectangle + centroid


def _plot_rectangle(ax: plt.axes, rectangle: np.ndarray) -> None:
    xs = np.append(rectangle[:, 0], rectangle[0, 0])
    ys = np.append(rectangle[:, 1], rectangle[0, 1])
    zs = np.append(rectangle[:, 2], rectangle[0, 2])
    ax.plot(xs, ys, zs)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def _plot_world_points(ax: plt.axes, points: np.ndarray) -> None:
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def _plot_camera(
    ax: plt.axes, world_t_world_camera: np.ndarray, camera_R_world: np.ndarray
) -> None:
    ax.scatter(
        world_t_world_camera[0],
        world_t_world_camera[1],
        world_t_world_camera[2],
    )

    arrow_length = 0.3
    arrow_end = world_t_world_camera + camera_R_world.dot(
        np.array([0.0, 0.0, arrow_length])
    )

    xs = np.array([world_t_world_camera[0], arrow_end[0]])
    ys = np.array([world_t_world_camera[1], arrow_end[1]])
    zs = np.array([world_t_world_camera[2], arrow_end[2]])

    ax.plot(xs, ys, zs)


def _create_camera_matrix(
    f: float, cam_width_px: int, cam_height_px: int
) -> npt.NDArray[float]:
    return np.array(
        [
            [f, 0.0, cam_width_px / 2.0],
            [0.0, f, cam_height_px / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _plot_camera_points(
    ax: plt.axes,
    camera_points: np.ndarray,
    cam_width: int,
    cam_height: int,
    mode: str = "plot",
):
    if "plot" == mode:
        ax.plot(camera_points[:, 0], camera_points[:, 1])
    elif "scatter" == mode:
        ax.scatter(camera_points[:, 0], camera_points[:, 1])
    else:
        raise ValueError(f"Unsupported mode {mode}")
    ax.set_xlim(0, cam_width)
    ax.set_ylim(0, cam_height)
    ax.set_aspect("equal")
