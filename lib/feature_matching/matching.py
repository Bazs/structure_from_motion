import dataclasses
import heapq
from enum import Enum
from typing import Callable, List, NewType

import numpy as np

from lib.common.feature import Feature

# Interface definition of a matching function.
ScoreFunction = NewType("ScoreFunction", Callable[[Feature, Feature], float])


@dataclasses.dataclass
class Match:
    a_index: int = -1
    b_index: int = -1
    # A lower score indicates a better match in all cases.
    match_score: float = np.Infinity

    def __lt__(self, other) -> bool:
        """Comparison operator based on match score."""
        return self.match_score < other.match_score


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
    score_function: ScoreFunction,
    *,
    validation_strategy: ValidationStrategy = ValidationStrategy.NONE,
    ratio_test_threshold: float = 0.5
) -> List[Match]:
    """
    Matches two list features using a brute-force pairwise method.

    :param features_a: First list of features.
    :param features_b: Second list of features.
    :param score_function: A score function which returns the score of a particular match.
    :param validation_strategy: The validation strategy to use.
    :param ratio_test_threshold: The maximum ratio between the scores of the best and second best match if
    validation_strategy == RATIO_TEST. Matches not meeting this criterion will be set to Match().
    :return: List of matches for every feature in features_a, in the order of features_a.
    """
    if validation_strategy == ValidationStrategy.CROSSCHECK:
        raise NotImplementedError

    all_matches_for_all_features_a: List[List[Match]] = [
        [] for _ in range(len(features_a))
    ]

    for a_index, feature_a in enumerate(features_a):
        for b_index, feature_b in enumerate(features_b):
            score = score_function(feature_a, feature_b)
            heapq.heappush(
                all_matches_for_all_features_a[a_index],
                Match(a_index=a_index, b_index=b_index, match_score=score),
            )

    if validation_strategy == ValidationStrategy.NONE:
        matches = [
            matches_for_feature[0]
            for matches_for_feature in all_matches_for_all_features_a
        ]
    elif validation_strategy == ValidationStrategy.RATIO_TEST:
        matches = _filter_by_ratio_test(
            all_matches_for_all_features_a, ratio_test_threshold
        )

    return matches


def _filter_by_ratio_test(
    all_matches_for_all_features: List[List[Match]], ratio_test_threshold: float
) -> List[Match]:
    matches = []
    for all_matches_for_feature in all_matches_for_all_features:
        if len(all_matches_for_feature) > 1:
            if (
                all_matches_for_feature[0].match_score
                / all_matches_for_feature[1].match_score
            ) <= ratio_test_threshold:
                matches.append(all_matches_for_feature[0])
            else:
                matches.append(Match())
        elif len(all_matches_for_feature) == 1:
            matches.append(all_matches_for_feature[0])
        else:
            matches.append(Match())
    return matches
