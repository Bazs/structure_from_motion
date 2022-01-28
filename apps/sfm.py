from typing import List, Tuple

from pathlib import Path
import logging

import cv2 as cv
import numpy as np

from lib.common import feature
from lib.feature_matching import matching, ssd
from lib.harris import harris_detector as harris

_WINDOW_NAME = "SfM"


def run_sfm() -> None:
    dataset_folder = Path("data/barcelona")
    if not dataset_folder.is_dir():
        raise FileNotFoundError(dataset_folder)

    test_image_1_filename = "DSCN8235.JPG"
    test_image_2_filename = "DSCN8238.JPG"

    cv.namedWindow(
        _WINDOW_NAME, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED
    )

    logging.info("Loading test images")
    test_image_1, test_image_1_gray = _load_image_rgb_and_gray(
        dataset_folder / test_image_1_filename
    )
    test_image_2, test_image_2_gray = _load_image_rgb_and_gray(
        dataset_folder / test_image_2_filename
    )

    logging.info("Extracting features")
    image_1_corners = harris.detect_harris_corners(test_image_1_gray)
    _draw_features(test_image_1, image_1_corners)
    # _show_image(test_image_1)
    image_2_corners = harris.detect_harris_corners(test_image_2_gray)
    _draw_features(test_image_2, image_2_corners)
    # _show_image(test_image_2)

    logging.info("Matching features")
    ssd_error_function = _create_ssd_function(test_image_1_gray, test_image_2_gray)
    matches = matching.match_features(
        image_1_corners, image_2_corners, ssd_error_function
    )
    print(matches)


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


def _draw_features(image: np.ndarray, features: List[feature.Feature]) -> None:
    for feature in features:
        center = [int(feature.x), int(feature.y)]
        image = cv.circle(image, center, radius=1, color=(255, 0, 0), thickness=-1)


def _show_image(image: np.ndarray) -> None:
    cv.imshow(_WINDOW_NAME, image)
    cv.waitKey()


def _create_ssd_function(
    image_a: np.ndarray, image_b: np.ndarray
) -> matching.ErrorFunction:
    def ssd_error(feature_a: feature.Feature, feature_b: feature.Feature) -> float:
        return ssd.calculate_ssd(image_a, image_b, feature_a, feature_b)

    return ssd_error
