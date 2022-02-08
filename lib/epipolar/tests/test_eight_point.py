import unittest

from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import numpy as np

from lib.common.feature import Feature
from lib.epipolar import eight_point
from lib.feature_matching.matching import Match


class EightPointTest(unittest.TestCase):
    def test_get_normalized_match_coordinates(self):
        image_size = (256, 512)
        features_a = [Feature(256, 128), Feature(128, 64)]
        features_b = [Feature(32, 64), Feature(16, 8)]
        matches = [Match(0, 1), Match(1, 0)]

        coords_a, coords_b = eight_point._get_normalized_match_coordinates(
            features_a, features_b, matches, image_size
        )

        expected_coords_a = np.array([[0.0, 0.0], [-0.5, -0.5]])
        np.testing.assert_allclose(expected_coords_a, coords_a)
        expected_coords_b = np.array(
            [
                [1.0 / 32.0 * 2.0 - 1.0, 1.0 / 32.0 * 2.0 - 1.0],
                [1.0 / 16.0 * 2.0 - 1.0, 1.0 / 4.0 * 2.0 - 1.0],
            ]
        )
        np.testing.assert_allclose(expected_coords_b, coords_b)

    def test_get_y_col(self):
        coord_a = np.array([2.0, 3.0])
        coord_b = np.array([7.0, 6.0])

        y_col = eight_point._get_y_col(coord_a, coord_b)
        self.assertEqual((9, 1), y_col.shape)

        expected_y_col = np.array(
            [
                [7.0 * 2.0],
                [7.0 * 3.0],
                [7.0],
                [6.0 * 2.0],
                [6.0 * 3.0],
                [6.0],
                [2.0],
                [3.0],
                [1.0],
            ]
        )

        np.testing.assert_allclose(expected_y_col, y_col)

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

        rect_a_rot = Rotation.from_euler("y", 20.0, degrees=True)
        world_t_world_rectangle_a = rect_a_rot.apply(world_t_world_rectangle_a)

        rect_b_rot = Rotation.from_euler("y", -20.0, degrees=True)
        world_t_world_rectangle_b = rect_b_rot.apply(world_t_world_rectangle_b)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        self._plot_rectangle(ax, world_t_world_rectangle_a)
        self._plot_rectangle(ax, world_t_world_rectangle_b)
        # plt.show()
        # TODO define camera poses and do projection

    @staticmethod
    def _rotate_rectangle(world_t_world_rectangle: np.ndarray, rotation: Rotation):
        # TODO transform rectangle the world origin before rotation
        pass

    @staticmethod
    def _plot_rectangle(ax: plt.axes, rectangle: np.ndarray):
        xs = np.append(rectangle[:, 0], rectangle[0, 0])
        ys = np.append(rectangle[:, 1], rectangle[0, 1])
        zs = np.append(rectangle[:, 2], rectangle[0, 2])
        ax.plot(xs, ys, zs, zdir="y")


if __name__ == "__main__":
    unittest.main()
