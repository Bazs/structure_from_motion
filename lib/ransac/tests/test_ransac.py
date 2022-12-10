import random
from math import isclose, sqrt
from typing import Sequence

import numpy as np
import numpy.typing as npt
from attr import define
from hypothesis import assume, example, given, note
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps
from matplotlib import pyplot as plt
from numpy.random import default_rng

from lib.ransac import ransac


@define
class LineModel:
    """Coefficients of the 2D line normal form equation of ax + by + c = 0."""

    a: float
    b: float
    c: float


def line_fitter_2d(points: Sequence[npt.NDArray]) -> LineModel:
    """Fit a line on two points."""
    if np.allclose(points[0], points[1]):
        raise ValueError("Cannot fit line on the same two points.")
    if len(points) != 2:
        raise ValueError(f"Two points are needed for line fitting, got {len(points)}.")
    dx = points[1][0] - points[0][0]
    if abs(dx) <= 1e-6:
        return LineModel(a=1, b=0, c=-points[0][0])
    slope = (points[1][1] - points[0][1]) / dx
    return LineModel(a=slope, b=-1, c=points[0][1] - slope * points[0][0])


@example(points=np.array([[1, 1], [0, 0]]))
@example(points=np.array([[1, 0], [0, 0]]))
@example(points=np.array([[0, 1], [0, 0]]))
@example(points=np.array([[1, 2], [1, 1]]))
@example(points=np.array([[0, 1], [1, 1]]))
@given(points=nps.arrays(dtype=float, shape=(2, 2), elements=st.floats(-100.0, 100.0)))
def test_line_fitter_2d(points: npt.NDArray):
    assume(not np.allclose(points[0, :], points[1, :]))
    model = line_fitter_2d(points)
    note(f"model: {model}")
    TOLERANCE = 1e-6
    np.testing.assert_allclose(
        0.0,
        model.a * points[0][0] + model.b * points[0][1] + model.c,
        rtol=0.0,
        atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        0.0,
        model.a * points[1][0] + model.b * points[1][1] + model.c,
        rtol=0.0,
        atol=TOLERANCE,
    )


def line_scorer_2d(model: LineModel, point: npt.NDArray) -> float:
    return abs(model.a * point[0] + model.b * point[1] + model.c) / sqrt(
        model.a ** 2 + model.b ** 2
    )


def test_ransac():
    line_start = np.array([4, 5])
    line_slope = 0.6
    num_line_points = 50
    dx = 0.3
    line_points = np.array(
        [
            line_start + np.array([i * dx, i * line_slope * dx])
            for i in range(num_line_points)
        ]
    ).reshape((-1, 2))

    min_x = np.amin(line_points[:, 0])
    min_y = np.amin(line_points[:, 1])
    max_x = np.amax(line_points[:, 0])
    max_y = np.amax(line_points[:, 1])
    rng = default_rng(seed=6)
    num_random_points = int(num_line_points / 2)
    noise_points = rng.random(size=(num_random_points, 2))
    noise_points[:, 0] *= max_x - min_x
    noise_points[:, 0] += min_x
    noise_points[:, 1] *= max_y - min_y
    noise_points[:, 1] += min_y

    all_points = np.vstack([line_points, noise_points])
    all_points_list = list(all_points)

    _, (all_points_ax, inliers_ax) = plt.subplots(nrows=1, ncols=2)
    all_points_ax.scatter(all_points[:, 0], all_points[:, 1])

    # Set the random seed for RANSAC.
    random.seed(5)

    model, inliers = ransac.fit_with_ransac(
        data=all_points_list,
        model_fit_data_count=2,
        model_fitter=line_fitter_2d,
        inlier_scorer=line_scorer_2d,
        inlier_threshold=0.2,
        min_num_extra_inliers=len(all_points_list) / 2,
        error_aggregation_method=ransac.ErrorAggregationMethod.RMS,
    )
    inliers = np.array(inliers)
    inliers_ax.scatter(inliers[:, 0], inliers[:, 1])

    # plt.show()

    ALLOWED_NOISE_INLIERS = 3
    num_extra_inliers = len(inliers) - len(line_points)
    assert 0 <= num_extra_inliers <= ALLOWED_NOISE_INLIERS
    pred_slope = -model.a / model.b
    assert isclose(line_slope, pred_slope, rel_tol=0, abs_tol=1e-7)
    exp_c = -np.sign(model.b) * (line_start[1] - line_start[0] * line_slope)
    assert isclose(exp_c, model.c)
