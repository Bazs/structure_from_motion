import unittest

from common.feature import Feature
from feature_matching import matching


def test_matching_function(feature_a: Feature, feature_b: Feature) -> float:
    pass


class MatchingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features_a = [Feature(1, 1), Feature(2, 2), Feature(3, 3)]
        self.features_b = [Feature(4, 4), Feature(5, 5), Feature(6, 6)]

    def test_match_features(self):
        pass
