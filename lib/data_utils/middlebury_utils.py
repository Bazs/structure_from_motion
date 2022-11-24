"""Utilities for loading metadta from the Middlebury vision dataset.
https://vision.middlebury.edu/mview/data/
"""

import re
from pathlib import Path

import numpy as np
import numpy.typing as npt


def load_camera_intrinsics(par_filepath: Path, file_index: int) -> npt.NDArray[float]:
    """Load camera intrinsic parameters from a parameters file.
    See https://vision.middlebury.edu/mview/data/ for the details
    of the file layout.

    Args:
      par_filepath: The *_.par.txt file.
      file_idnex: The index of the image to load the intrinsics for.
    Returns:
      The camera intrinsic matrix as a numpy matrix.
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
                return np.array([float(param) for param in line_parts[1:10]]).reshape(
                    (3, 3)
                )
        raise ValueError(f"Could not find matching entry for file index {file_index}")
