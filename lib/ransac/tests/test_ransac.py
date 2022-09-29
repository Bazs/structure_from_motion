import numpy as np
import numpy.typing as npt
from attr import define
from hypothesis import assume, example, given, note
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps

from lib.ransac import ransac


@define
class LineModel:
    a: float
    b: float
    c: float


def line_fitter_2d(points: npt.NDArray) -> LineModel:
    if np.allclose(points[0, :], points[1, :]):
        raise ValueError("Cannot fit line on the same two points.")
    if len(points) != 2:
        raise ValueError(f"Two points are needed for line fitting, got {len(points)}.")
    dx = points[1, 0] - points[0, 0]
    if abs(dx) <= 1e-6:
        return LineModel(a=1, b=0, c=-points[0][0])
    slope = (points[1, 1] - points[0, 1]) / dx
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
    pass


def test_ransac():
    pass
