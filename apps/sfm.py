import functools
import logging
import os
from pathlib import Path
from typing import Callable, List, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2 as cv
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation

from lib.common import feature
from lib.data_utils.middlebury_utils import load_camera_k_r_t
from lib.epipolar.eight_point import create_trivial_matches, recover_r_t_from_e
from lib.epipolar.epipolar_ransac import estimate_essential_mat_with_ransac
from lib.epipolar.triangulation import triangulate_points
from lib.feature_matching import matching, ncc
from lib.harris import harris_detector as harris
from lib.ransac.ransac import ErrorAggregationMethod
from lib.transforms.transforms import Transform3D

_WINDOW_NAME = "SfM"

_DATASET_FOLDER = Path("data/temple")
_PARAMETERS_FILEPATH = _DATASET_FOLDER / "temple_par.txt"
_TEST_IMAGE_1_IDX = "0170"
_TEST_IMAGE_2_IDX = "0172"
_TEST_IMAGE_1_FILENAME = f"temple{_TEST_IMAGE_1_IDX}.png"
_TEST_IMAGE_2_FILENAME = f"temple{_TEST_IMAGE_2_IDX}.png"


@hydra.main(config_path="config", config_name="config")
def run_sfm(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())
    if not _DATASET_FOLDER.is_dir():
        raise FileNotFoundError(_DATASET_FOLDER)

    cv.namedWindow(
        _WINDOW_NAME, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED
    )

    logging.info("Loading test images")
    test_image_1, test_image_1_gray = _load_image_rgb_and_gray(
        _DATASET_FOLDER / _TEST_IMAGE_1_FILENAME, cfg.image_downscale_factor
    )
    test_image_2, test_image_2_gray = _load_image_rgb_and_gray(
        _DATASET_FOLDER / _TEST_IMAGE_2_FILENAME, cfg.image_downscale_factor
    )
    image_1_k, cam1_T_world = load_camera_k_r_t(
        _PARAMETERS_FILEPATH, int(_TEST_IMAGE_1_IDX)
    )
    image_2_k, cam2_T_world = load_camera_k_r_t(
        _PARAMETERS_FILEPATH, int(_TEST_IMAGE_2_IDX)
    )
    if not np.allclose(image_1_k, image_2_k):
        raise ValueError(
            f"Camera intrinsics params are different for the images, which is "
            "currently not supported."
        )

    logging.info("Extracting features")
    image_1_corners = harris.detect_harris_corners(
        test_image_1_gray, num_corners=cfg.num_harris_corners
    )
    _draw_features(test_image_1, image_1_corners)
    image_2_corners = harris.detect_harris_corners(
        test_image_2_gray, num_corners=cfg.num_harris_corners
    )
    _draw_features(test_image_2, image_2_corners)

    logging.info("Matching features")
    ncc_function = functools.partial(ncc.calculate_ncc, window_size=cfg.ncc_window_size)
    ssd_score_function = _create_score_function(
        test_image_1_gray, test_image_2_gray, ncc_function
    )
    matches = matching.match_brute_force(
        image_1_corners,
        image_2_corners,
        ssd_score_function,
        validation_strategies={
            matching.ValidationStrategy.RATIO_TEST,
            matching.ValidationStrategy.CROSSCHECK,
        },
        ratio_test_threshold=cfg.ratio_test_threshold,
    )
    match_scores = [
        match.match_score for match in matches if match.match_score != np.Infinity
    ]
    match_image = _draw_matches(
        test_image_1,
        test_image_2,
        image_1_corners,
        image_2_corners,
        matches,
    )
    _show_image(match_image)
    # cv.waitKey()

    # Show a histogram of matching scores
    fig, ax = plt.subplots(1, 1)
    ax.hist(match_scores)
    ax.set_title("Matching Scores")
    # fig.show()

    matches = _filter_matches(matches, cfg.match_score_threshold)

    logging.info("Estimating Essential Matrix")
    e, inlier_feature_pairs = estimate_essential_mat_with_ransac(
        image_1_k,
        features_a=image_1_corners,
        features_b=image_2_corners,
        matches=matches,
        sed_inlier_threshold=cfg.ransac.sed_inlier_threshold,
        error_aggregation_method=ErrorAggregationMethod.RMS,
        min_num_extra_inliers=cfg.ransac.min_num_extra_inliers,
        max_iterations=cfg.ransac.max_iterations,
    )
    inlier_features_a = [pair[0] for pair in inlier_feature_pairs]
    inlier_features_b = [pair[1] for pair in inlier_feature_pairs]
    match_image = _draw_matches(
        test_image_1,
        test_image_2,
        inlier_features_a,
        inlier_features_b,
        create_trivial_matches(len(inlier_feature_pairs)),
    )
    _show_image(match_image)
    # cv.waitKey()

    logging.info("Recovering Relative Pose")
    r, t, inlier_mask = recover_r_t_from_e(
        e=e,
        camera_matrix=image_1_k,
        features_a=inlier_features_a,
        features_b=inlier_features_b,
    )
    # Check against the OpenCV solution
    _, r_cv, t_cv, _ = cv.recoverPose(
        e,
        np.array(
            [
                [feature_pair[0].x, feature_pair[0].y]
                for feature_pair in inlier_feature_pairs
            ]
        ),
        np.array(
            [
                [feature_pair[1].x, feature_pair[1].y]
                for feature_pair in inlier_feature_pairs
            ]
        ),
        image_1_k,
    )
    if not np.allclose(r, r_cv, rtol=0.0, atol=1e-5) or not np.allclose(
        t, np.squeeze(t_cv), rtol=0.0, atol=1e-5
    ):
        raise RuntimeError(
            f"OpenCV pose estimate\nR:\n{r_cv}\nt:\n{t_cv}\ndiffers from estimated pose\nR:\n{r}\nt:\n{t}"
        )

    cam2_T_cam1 = Transform3D.from_rmat_t(r, t)
    logging.info("Calculated transform:")
    _print_transform(cam2_T_cam1)

    # Only keep matches which passed the cheirality check.
    inlier_features_a = np.take(inlier_features_a, inlier_mask)
    inlier_features_b = np.take(inlier_features_b, inlier_mask)
    match_image = _draw_matches(
        test_image_1,
        test_image_2,
        inlier_features_a,
        inlier_features_b,
        create_trivial_matches(len(inlier_features_a)),
    )
    _show_image(match_image)
    # cv.waitKey()

    logging.info("Triangulating points")
    world_t_world_points = triangulate_points(
        inlier_features_a,
        inlier_features_b,
        intrinsic_camera_matrix=image_1_k,
        cam2_T_cam1=cam2_T_cam1,
    )

    # Check against OpenCV solution.
    K_ext = np.hstack((image_1_k, np.zeros((3, 1))))
    cam1_T_world = Transform3D.from_rmat_t(np.eye(3), np.zeros((3,)))
    cam2_T_world = cam2_T_cam1 @ cam1_T_world
    P1 = K_ext @ cam1_T_world.Tmat
    P2 = K_ext @ cam2_T_world.Tmat
    world_t_world_points_cv = cv.triangulatePoints(
        P1,
        P2,
        np.array([[feature.x, feature.y] for feature in inlier_features_a]).T,
        np.array([[feature.x, feature.y] for feature in inlier_features_b]).T,
    )
    world_t_world_points_cv = world_t_world_points_cv.T
    world_t_world_points_cv /= world_t_world_points_cv[:, -1].reshape((-1, 1))
    world_t_world_points_cv = world_t_world_points_cv[:, :-1]

    fig = plt.figure()
    ax_3d = fig.add_subplot(111, projection="3d")
    ax_3d.set_title("Estimated 3D points.")
    ax_3d.scatter(
        world_t_world_points[:, 0],
        world_t_world_points[:, 1],
        world_t_world_points[:, 2],
    )
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    fig.show()
    cv.waitKey()


def _load_image_rgb_and_gray(
    image_path: Path, downscale_factor: float
) -> Tuple[np.ndarray, np.ndarray]:
    image = cv.imread(str(image_path))
    image = _downscale_image(image, downscale_factor)
    image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    return image, image_gray


def _downscale_image(image: np.ndarray, downscale_factor: float) -> np.ndarray:
    smaller_size = (
        int(image.shape[1] / downscale_factor),
        int(image.shape[0] / downscale_factor),
    )
    return cv.resize(image, smaller_size, interpolation=cv.INTER_LANCZOS4)


def _draw_features(image: np.ndarray, features: List[feature.Feature]) -> None:
    for feat in features:
        center = [int(feat.x), int(feat.y)]
        image = cv.circle(image, center, radius=1, color=(255, 0, 0), thickness=-1)


def _show_image(image: np.ndarray) -> None:
    cv.imshow(_WINDOW_NAME, image)


def _create_score_function(
    image_a: np.ndarray,
    image_b: np.ndarray,
    full_score_function: Callable[
        [np.ndarray, np.ndarray, feature.Feature, feature.Feature, int], float
    ],
) -> matching.ScoreFunction:
    def ssd_score(feature_a: feature.Feature, feature_b: feature.Feature) -> float:
        return full_score_function(image_a, image_b, feature_a, feature_b)

    return ssd_score


def _filter_matches(
    matches: List[matching.Match],
    score_threshold: float,
) -> List[matching.Match]:
    """Filters matches by thresholding the match score.

    :param matches: List of matches.
    :param score_threshold: The match score threshold.
    :return: matches where indices corresponding to a match score above score_threshold are removed.
    """
    indices_to_delete = []
    for index in range(len(matches)):
        if matches[index].match_score > score_threshold:
            indices_to_delete.append(index)
    matches = np.delete(matches, indices_to_delete).tolist()
    return matches


def _draw_matches(
    image_1: np.ndarray,
    image_2: np.ndarray,
    features_1: List[feature.Feature],
    features_2: List[feature.Feature],
    matches: List[matching.Match],
) -> np.ndarray:
    def to_cv_keypoints(features):
        return [cv.KeyPoint(feat.x, feat.y, 1) for feat in features]

    image_1 = image_1.copy()
    image_2 = image_2.copy()

    keypoints_1 = to_cv_keypoints(features_1)
    keypoints_2 = to_cv_keypoints(features_2)
    cv_matches = [cv.DMatch(match.a_index, match.b_index, 0) for match in matches]
    match_image = cv.drawMatches(
        image_1, keypoints_1, image_2, keypoints_2, cv_matches, None
    )

    return match_image


def _print_transform(transform: Transform3D):
    rot = Rotation.from_matrix(transform.Rmat)
    logging.info(
        f"Transformation:\nrotation XYZ:\n{rot.as_euler('XYZ', degrees=True)} deg\nt:\n{transform.t}"
    )
