import unittest

from lib.common.feature import Feature
from lib.feature_matching import matching


class MatchingTest(unittest.TestCase):
    def test_match_features_no_validation(self):
        features_a = [Feature(1, 1), Feature(2, 2), Feature(3, 3)]
        features_b = [Feature(4, 4), Feature(5, 5), Feature(6, 6), Feature(7, 7)]

        def _mock_matching_function(feature_a: Feature, feature_b: Feature):
            # Dict of {feature_a_index: {if_matched_with_feature_b_index: score}}
            feature_a_match_map = {
                0: {0: 10, 1: 20, 2: 30, 3: 7},
                1: {0: 30, 1: 9, 2: 20, 3: 15},
                2: {0: 20, 1: 30, 2: 8, 3: 31},
            }
            possible_matches_for_a = feature_a_match_map[features_a.index(feature_a)]
            score = possible_matches_for_a[features_b.index(feature_b)]
            return score

        matches = matching.match_brute_force(
            features_a,
            features_b,
            _mock_matching_function,
        )
        expected_matches = [
            matching.Match(match_index=3, match_score=7),
            matching.Match(match_index=1, match_score=9),
            matching.Match(match_index=2, match_score=8),
        ]
        self.assertEqual(expected_matches, matches)

    def test_match_features_ratio_test(self):
        features_a = [Feature(1, 1), Feature(2, 2)]
        features_b = [Feature(3, 3), Feature(4, 4), Feature(5, 5)]

        def _mock_matching_function(feature_a: Feature, feature_b: Feature):
            # Dict of {feature_a_index: {if_matched_with_feature_b_index: score}}
            feature_a_match_map = {
                0: {0: 10, 1: 5, 2: 20},  # Ratio is exactly 0.5, the threshold
                1: {0: 10, 1: 6, 2: 7},  # Ratio is above 0.5
            }
            possible_matches_for_a = feature_a_match_map[features_a.index(feature_a)]
            score = possible_matches_for_a[features_b.index(feature_b)]
            return score

        matches = matching.match_brute_force(
            features_a,
            features_b,
            _mock_matching_function,
            validation_strategy=matching.ValidationStrategy.RATIO_TEST,
            ratio_test_threshold=0.5,
        )

        expected_matches = [matching.Match(1, 5.0), matching.Match()]
        self.assertEqual(expected_matches, matches)
