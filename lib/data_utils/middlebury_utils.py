"""Utilities for loading metadta from the Middlebury vision dataset.
https://vision.middlebury.edu/mview/data/
"""

import re
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numpy.lib.index_tricks import IndexExpression

from lib.transforms.transforms import Transform3D


def load_camera_k_r_t(
    par_filepath: Path, file_index: int
) -> tuple[npt.NDArray[float], Transform3D]:
    """Load camera intrinsic parameters and extrinsic transformation from a parameters file.
    See https://vision.middlebury.edu/mview/data/ for the details
    of the file layout.

    Args:
      par_filepath: The *_.par.txt file.
      file_idnex: The index of the image to load the intrinsics for.
    Returns:
      Tuple containing the camera intrinsic matrix as a numpy matrix, and the extrinsic transform.
    """
    with par_filepath.open("rt") as par_file:
        num_entries = int(par_file.readline())
        if file_index > num_entries:
            raise ValueError(
                f"There are {num_entries} entries in {par_filepath}, requested entry no. {file_index}."
            )
        while line := par_file.readline():
            line_parts = line.split(" ")
            filename = line_parts[0]
            match = re.match(r"^.+?([\d]+)\.png$", filename)
            if match is None:
                raise RuntimeError(f"Could not decode filename {filename}.")
            line_file_index = int(match[1])
            if line_file_index == file_index:
                k = _extract_mat_from_line_parts(line_parts, np.s_[1:10], (3, 3))
                r = _extract_mat_from_line_parts(line_parts, np.s_[10:19], (3, 3))
                t = _extract_mat_from_line_parts(line_parts, np.s_[19:22], (3, 1))
                return k, Transform3D.from_rmat_t(r, t)
        raise ValueError(f"Could not find matching entry for file index {file_index}")


def _extract_mat_from_line_parts(
    line_parts: list[str], slice: IndexExpression, shape: tuple[int, ...]
) -> npt.NDArray[float]:
    return np.array([float(param) for param in line_parts[slice]]).reshape(shape)
