import numpy as np
import numpy.typing as npt
from attr import define

from lib.ransac import ransac


@define
class LineModel:
    a: float
    b: float
    c: float


def line_fitter_2d(points: npt.NDArray) -> LineModel:
    if len(points) != 2:
        raise ValueError(f"Two points are needed for line fitting, got {len(points)}.")
    if np.linalg.det(points) > 1e-5:
        ab = np.linalg.solve(points, np.array([-1, -1]).reshape((2, 1)))
        norm = np.linalg.norm(ab)
        ab /= norm
        return LineModel(a=ab[0], b=ab[1], c=1 / norm)
    else:
        dx = points[1, 0] - points[0, 0]
        if dx <= 1e-5:
            return LineModel(a=1, b=0, c=0)
        slope = (points[1, 1] - points[0, 1]) / dx
        return LineModel(a=slope, b=-1, c=0)


def line_scorer_2d(model: LineModel, point: npt.NDArray) -> float:
    pass


def test_ransac():
    pass
