import numpy as np
import numpy.typing as npt

from lib.common.feature import Feature


def calculate_symmetric_epipolar_distance(
    feature_a: Feature, feature_b: Feature, e: npt.NDArray
) -> float:
    """Calculate the Symmetric Epipolar Distance (SED) for a pair of point correspondences between two images,
    and the Essential matrix relating the two images.

    The SED is the geometric distance of each point to their epipolar line. It is a biased estimate of the
    Reprojection Error, but is faster to compute.
    """
    # Calculate the algebraic distance.
    coord_a = np.array([feature_a.x, feature_a.y, 1.0])
    coord_b = np.array([feature_b.x, feature_b.y, 1.0])
    r = coord_a.T @ e @ coord_b

    # Calculate the epipolar lines.
    line_a = e @ coord_b
    line_b = e.T @ coord_a

    sed = (
        1.0 / np.sum(np.square(line_a[:2])) + 1.0 / np.sum(np.square(line_b[:2]))
    ) * r ** 2
    return sed
