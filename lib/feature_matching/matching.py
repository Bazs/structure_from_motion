import dataclasses
import heapq
from enum import Enum
from typing import Callable, Dict, List, NewType, Set

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

    CROSSCHECK = 1  # Features (i, j) are only a match, if the Score(i,j) is the lowest out of all scores for both i and j
    RATIO_TEST = (
        2  # The ratio of the two best scores need to be larger than a threshold
    )


def match_brute_force(
    features_a: List[Feature],
    features_b: List[Feature],
    score_function: ScoreFunction,
    *,
    validation_strategies: ValidationStrategy | Set[ValidationStrategy] | None = None,
    ratio_test_threshold: float = 0.5
) -> List[Match]:
    """
    Matches two list features using a brute-force pairwise method.

    :param features_a: First list of features.
    :param features_b: Second list of features.
    :param score_function: A score function which returns the score of a particular match.
    :param validation_strategies: The validation strategies to use.
    :param ratio_test_threshold: The maximum ratio between the scores of the best and second best match if
    validation_strategy == RATIO_TEST. Matches not meeting this criterion will be set to Match().
    :return: List of matches for every feature in features_a, in the order of features_a.
    """
    matches_for_a_features: List[List[Match]] = [[] for _ in range(len(features_a))]

    for a_index, feature_a in enumerate(features_a):
        for b_index, feature_b in enumerate(features_b):
            score = score_function(feature_a, feature_b)
            heapq.heappush(
                matches_for_a_features[a_index],
                Match(a_index=a_index, b_index=b_index, match_score=score),
            )

    if validation_strategies is None:
        validation_strategies = set()
    elif not isinstance(validation_strategies, set):
        validation_strategies = set([validation_strategies])

    if ValidationStrategy.RATIO_TEST in validation_strategies:
        matches_for_a_features = _filter_by_ratio_test(
            matches_for_a_features, ratio_test_threshold
        )
    if ValidationStrategy.CROSSCHECK in validation_strategies:
        matches_for_a_features = _filter_by_crosscheck(matches_for_a_features)

    matches = [matches_for_feature[0] for matches_for_feature in matches_for_a_features]

    return matches


def _filter_by_ratio_test(
    all_matches_for_all_features: List[List[Match]], ratio_test_threshold: float
) -> List[List[Match]]:
    matches = []
    for all_matches_for_feature in all_matches_for_all_features:
        if len(all_matches_for_feature) > 1:
            if (
                all_matches_for_feature[0].match_score
                / all_matches_for_feature[1].match_score
            ) <= ratio_test_threshold:
                matches.append([all_matches_for_feature[0]])
        elif len(all_matches_for_feature) == 1:
            matches.append([all_matches_for_feature[0]])
    return matches


def _filter_by_crosscheck(
    all_matches_for_a_features: List[List[Match]],
) -> List[List[Match]]:
    best_matches_for_b_features: Dict[int, Match] = {}
    for matches_for_a in all_matches_for_a_features:
        best_match_for_a = matches_for_a[0]
        if (
            best_match_for_a.b_index not in best_matches_for_b_features
            or best_matches_for_b_features[best_match_for_a.b_index].match_score
            > best_match_for_a.match_score
        ):
            best_matches_for_b_features[best_match_for_a.b_index] = best_match_for_a

    filtered_matches = [
        [matches_for_a[0]]
        for matches_for_a in all_matches_for_a_features
        if matches_for_a[0] == best_matches_for_b_features[matches_for_a[0].b_index]
    ]
    return filtered_matches
