from typing import List, Tuple

from pathlib import Path
import logging

from cv2 import cv2 as cv
import matplotlib.pyplot as plt
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
    image_2_corners = harris.detect_harris_corners(test_image_2_gray)
    _draw_features(test_image_2, image_2_corners)

    logging.info("Matching features")
    ssd_error_function = _create_ssd_function(test_image_1_gray, test_image_2_gray)
    matches = matching.match_brute_force(
        image_1_corners, image_2_corners, ssd_error_function
    )
    match_errors = [
        match.match_error for match in matches if match.match_error != np.Infinity
    ]

    # Show a histogram of matching errors
    fig, ax = plt.subplots(1, 1)
    ax.hist(match_errors)
    ax.set_title("Matching Errors")
    fig.show()

    matches, image_1_corners = _filter_matches_features(matches, 60, image_1_corners)

    match_image = _draw_matches(
        test_image_1, test_image_2, image_1_corners, image_2_corners, matches
    )
    _show_image(match_image)


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
        return ssd.calculate_ssd(image_a, image_b, feature_a, feature_b, window_size=9)

    return ssd_error


def _filter_matches_features(
    matches: List[matching.Match],
    error_threshold: float,
    features_a: List[feature.Feature],
) -> Tuple[List[matching.Match], List[feature.Feature]]:
    """Filters matches and features_a by thresholding the match error.

    :param matches: List of matches.
    :param error_threshold: The match error threshold.
    :param features_a: The query feature list which was used in matching.
    :return: matches and features_a, where all indices corresponding to a match
    error above error_threshold are removed.
    """
    indices_to_delete = []
    for index in range(len(matches)):
        if matches[index].match_error > error_threshold:
            indices_to_delete.append(index)
    matches = np.delete(matches, indices_to_delete).tolist()
    features_a = np.delete(features_a, indices_to_delete).tolist()
    return (matches, features_a)


def _draw_matches(
    image_1: np.ndarray,
    image_2: np.ndarray,
    features_1: List[feature.Feature],
    features_2: List[feature.Feature],
    matches: List[matching.Match],
) -> np.ndarray:
    def to_cv_keypoints(features):
        return [cv.KeyPoint(feat.x, feat.y, 1) for feat in features]

    keypoints_1 = to_cv_keypoints(features_1)
    keypoints_2 = to_cv_keypoints(features_2)
    cv_matches = [
        cv.DMatch(index, match.match_index, 0) for index, match in enumerate(matches)
    ]
    match_image = cv.drawMatches(
        image_1, keypoints_1, image_2, keypoints_2, cv_matches, None
    )
    return match_image
