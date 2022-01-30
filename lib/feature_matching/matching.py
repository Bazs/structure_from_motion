from enum import Enum
from typing import Callable, List, NewType
import dataclasses

import numpy as np

from lib.common.feature import Feature

ErrorFunction = NewType("ErrorFunction", Callable[[Feature, Feature], float])


@dataclasses.dataclass(frozen=False)
class Match:
    best_match_index: int = -1
    best_match_error: float = np.Infinity


class ValidationStrategy(Enum):
    """Validation strategy when matching features."""

    NONE = 0  # No validation is done
    CROSSCHECK = 1  # Features (i, j) are only a match, if the Score(i,j) is the lowest out of all scores for both i and j
    RATIO_TEST = (
        2  # The ratio of the two best scores need to be larger than a threshold
    )


def match_brute_force(
    features_a: List[Feature],
    features_b: List[Feature],
    error_function: ErrorFunction,
    validation_strategy: ValidationStrategy = ValidationStrategy.NONE,
) -> List[Match]:
    """
    Matches two list features using a brute-force pairwise method.

    :param features_a: First list of features.
    :param features_b: Second list of features.
    :param error_function: An error function which returns the error of a particular match.
    :param validation_strategy: The validation strategy to use.
    :return: List of matches for every feature in features_a, in the order of features_a.
    """
    if validation_strategy != ValidationStrategy.NONE:
        raise NotImplementedError

    feature_a_matches = [Match() for _ in range(len(features_a))]

    for a_index, feature_a in enumerate(features_a):
        for b_index, feature_b in enumerate(features_b):
            error = error_function(feature_a, feature_b)
            if feature_a_matches[a_index].best_match_error > error:
                feature_a_matches[a_index].best_match_error = error
                feature_a_matches[a_index].best_match_index = b_index

    return feature_a_matches
