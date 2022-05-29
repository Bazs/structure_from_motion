import logging
import unittest

from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import numpy as np
import cv2.cv2 as cv

from lib.common.feature import Feature
from lib.epipolar import eight_point
from lib.feature_matching.matching import Match


class EightPointTest(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(filename)s:%(lineno)s\t %(message)s",
        )

    def test_get_matching_coordinates(self):
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

    def test_normalize_coords(self):
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

    def test_get_y_row(self):
        coord_a = np.array([2.0, 3.0])
        coord_b = np.array([7.0, 6.0])

        y_col = eight_point._get_y_col(coord_a, coord_b)
        self.assertEqual((9,), y_col.shape)

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

        # np.testing.assert_allclose(expected_y_col, y_col)

    def test_estimate_essential_matrix(self):
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
        world_t_world_rectangle_a = self._rotate_rectangle(
            world_t_world_rectangle_a, rect_a_rot
        )

        rect_b_rot = Rotation.from_euler("y", 40.0, degrees=True)
        world_t_world_rectangle_b = self._rotate_rectangle(
            world_t_world_rectangle_b, rect_b_rot
        )

        fig = plt.figure()
        ax_3d = fig.add_subplot(131, projection="3d")
        self._plot_rectangle(ax_3d, world_t_world_rectangle_a)
        self._plot_rectangle(ax_3d, world_t_world_rectangle_b)

        world_t_world_camera1 = np.array([1.5, 0.25, -1.0])
        camera1_Rmat_world = np.eye(3, dtype=float)
        camera1_Rvec_world, _ = cv.Rodrigues(camera1_Rmat_world)

        world_t_world_camera2 = np.array([2.5, 0.1, -1.5])
        camera2_R_world = Rotation.from_euler("XY", [-20.0, -50.0], degrees=True)
        camera2_Rmat_world = camera2_R_world.as_matrix()
        camera2_Rvec_world, _ = cv.Rodrigues(camera2_Rmat_world)

        self._plot_camera(ax_3d, world_t_world_camera1, camera1_Rmat_world)
        self._plot_camera(ax_3d, world_t_world_camera2, camera2_Rmat_world)

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
        K = np.array(
            [
                [f, 0.0, cam_width_px / 2.0],
                [0.0, f, cam_height_px / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )

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
        self._plot_camera_points(cam1_ax, cam1_points, cam_width_px, cam_height_px)

        cam2_points, _ = cv.projectPoints(
            world_t_world_allPoints,
            camera2_Rvec_world,
            -world_t_world_camera2,
            K,
            None,
        )
        cam2_points = cam2_points.squeeze()
        cam2_ax = fig.add_subplot(133)
        self._plot_camera_points(cam2_ax, cam2_points, cam_width_px, cam_height_px)
        # plt.show()

        features_1 = [Feature(x=point[0], y=point[1]) for point in cam1_points]
        features_2 = [Feature(x=point[0], y=point[1]) for point in cam2_points]
        matches = [
            Match(a_index=index, b_index=index, match_score=0.0)
            for index in range(len(features_1))
        ]

        image_size = (
            cam_height_px,
            cam_width_px,
        )

        e = eight_point.estimate_essential_mat(
            features_1,
            features_2,
            matches,
        )

        coords_a, coords_b = eight_point._get_matching_coordinates(
            features_1, features_2, matches
        )
        np.set_printoptions(precision=20)
        print(
            f"Input coords for findFundamenalMat:\ncoords_a:\n{coords_a}\ncoords_b:\n{coords_b}"
        )
        e_cv, _ = cv.findFundamentalMat(coords_a, coords_b, method=cv.FM_8POINT)
        print(f"OpenCV estimated essential mat: {e_cv}")
        # TODO compare results

        r, t = eight_point.recover_r_t(features_1[0], features_2[0], e)

    @staticmethod
    def _rotate_rectangle(
        world_t_world_rectangle: np.ndarray, rotation: Rotation
    ) -> np.ndarray:
        centroid = np.mean(world_t_world_rectangle, axis=0)
        rectangle_t_rectangle_rectangle = world_t_world_rectangle - centroid
        rotated_rectangle = rotation.apply(rectangle_t_rectangle_rectangle)
        return rotated_rectangle + centroid

    @staticmethod
    def _plot_rectangle(ax: plt.axes, rectangle: np.ndarray) -> None:
        xs = np.append(rectangle[:, 0], rectangle[0, 0])
        ys = np.append(rectangle[:, 1], rectangle[0, 1])
        zs = np.append(rectangle[:, 2], rectangle[0, 2])
        ax.plot(xs, ys, zs)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    @staticmethod
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

    @staticmethod
    def _plot_camera_points(
        ax: plt.axes, camera_points: np.ndarray, cam_width: int, cam_height: int
    ):
        ax.plot(camera_points[:, 0], camera_points[:, 1])
        ax.set_xlim(0, cam_width)
        ax.set_ylim(0, cam_height)
        ax.set_aspect("equal")


if __name__ == "__main__":
    unittest.main()
