from typing import Tuple

from pathlib import Path

import cv2 as cv
import numpy as np


def run_sfm() -> None:
    dataset_folder = Path("data/barcelona")
    if not dataset_folder.is_dir():
        raise FileNotFoundError(dataset_folder)

    test_image_1_filename = "DSCN8238.JPG"
    test_image_2_filename = "DSCN8238.JPG"

    window_name = "window"
    cv.namedWindow(
        window_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED
    )

    test_image_1, test_image_1_gray = _load_image_rgb_and_gray(
        dataset_folder / test_image_1_filename
    )
    test_image_2, test_image_2_gray = _load_image_rgb_and_gray(
        dataset_folder / test_image_2_filename
    )


def _load_image_rgb_and_gray(image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    image = cv.imread(str(image_path))
    image = _downscale_image(image)
    image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    return image, image_gray


def _downscale_image(image: np.ndarray) -> np.ndarray:
    downscale_factor = 8
    smaller_size = (
        int(image.shape[1] / downscale_factor),
        int(image.shape[0] / downscale_factor),
    )
    return cv.resize(image, smaller_size, interpolation=cv.INTER_LANCZOS4)
