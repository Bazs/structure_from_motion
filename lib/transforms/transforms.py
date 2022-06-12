import numpy as np
import numpy.typing as npt
from transforms3d import affines


def compose_r_t(r: npt.NDArray, t: npt.NDArray) -> npt.NDArray:
    return affines.compose(t, r, np.ones((3,)))
