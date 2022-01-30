import unittest

from lib.common.feature import Feature
from lib.feature_matching import matching


class MatchingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features_a = [Feature(1, 1), Feature(2, 2), Feature(3, 3)]
        self.features_b = [Feature(4, 4), Feature(5, 5), Feature(6, 6), Feature(7, 7)]

    def _mock_matching_function(self, feature_a: Feature, feature_b: Feature):
        # Dict of {feature_a_index: {if_matched_with_feature_b_index: error}}
        feature_a_match_map = {
            0: {0: 10, 1: 20, 2: 30, 3: 7},
            1: {0: 30, 1: 9, 2: 20, 3: 15},
            2: {0: 20, 1: 30, 2: 8, 3: 31},
        }
        possible_matches_for_a = feature_a_match_map[self.features_a.index(feature_a)]
        score = possible_matches_for_a[self.features_b.index(feature_b)]
        return score

    def test_match_features(self):
        matches = matching.match_brute_force(
            self.features_a,
            self.features_b,
            self._mock_matching_function,
        )
        expected_matches = [
            matching.Match(best_match_index=3, best_match_error=7),
            matching.Match(best_match_index=1, best_match_error=9),
            matching.Match(best_match_index=2, best_match_error=8),
        ]
        self.assertEqual(expected_matches, matches)
