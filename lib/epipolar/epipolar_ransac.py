from __future__ import annotations

from functools import partial
from multiprocessing.sharedctypes import Value
from typing import Tuple

import numpy.typing as npt

from lib.common.feature import Feature
from lib.epipolar.eight_point import estimate_essential_mat, to_normalized_image_coords
from lib.epipolar.sed import calculate_symmetric_epipolar_distance
from lib.feature_matching.matching import Match
from lib.ransac.ransac import fit_with_ransac

FeaturePair = Tuple[Feature, Feature]


def calculate_sed_inlier_score(
    e: npt.NDArray, matching_features: FeaturePair, camera_matrix: npt.NDArray
) -> float:
    feature_a = to_normalized_image_coords(matching_features[0], camera_matrix)
    feature_b = to_normalized_image_coords(matching_features[1], camera_matrix)
    return calculate_symmetric_epipolar_distance(
        feature_a=feature_a, feature_b=feature_b, e=e
    )


def eight_point_model_fitter(
    matching_features: list[FeaturePair], camera_matrix: npt.NDArray
) -> npt.NDArray:
    if 8 != len(matching_features):
        raise ValueError("Eight feature pairs are expected.")

    matches = [Match(a_index=idx, b_index=idx) for idx in range(len(matching_features))]
    features_a = [pair[0] for pair in matching_features]
    features_b = [pair[1] for pair in matching_features]
    return estimate_essential_mat(
        camera_matrix=camera_matrix,
        features_a=features_a,
        features_b=features_b,
        matches=matches,
    )


def estimate_essential_mat_with_ransac(
    camera_matrix: npt.NDArray,
    features_a: list[Feature],
    features_b: list[Feature],
    matches: list[Match],
    sed_inlier_threshold: float,
) -> Tuple[npt.NDArray, list[FeaturePair]]:
    feature_pairs = [
        (features_a[match.a_index], features_b[match.b_index]) for match in matches
    ]
    e, inlier_feature_pairs = fit_with_ransac(
        feature_pairs,
        model_fit_data_count=8,
        model_fitter=partial(eight_point_model_fitter, camera_matrix=camera_matrix),
        inlier_scorer=partial(calculate_sed_inlier_score, camera_matrix=camera_matrix),
        inlier_threshold=sed_inlier_threshold,
    )
    if e is None:
        raise ValueError("Could not estimate Essential Matrix with RANSAC.")
    return e, inlier_feature_pairs
