import random
from typing import Any, Callable, Sequence, Tuple


def fit_with_ransac(
    data: Sequence,
    model_fit_data_count: int,
    model_fitter: Callable[[Sequence], Any],
    inlier_scorer: Callable[..., float],
    inlier_threshold: float,
    max_iterations: int = 100,
) -> Tuple[Any, Sequence]:
    best_model = None
    best_model_inliers = []

    for _ in range(max_iterations):
        samples = random.sample(data, k=model_fit_data_count)
        model = model_fitter(samples)
        inliers = [
            data_point
            for data_point in data
            if inlier_scorer(model, data_point) <= inlier_threshold
        ]
        if len(inliers) > len(best_model_inliers):
            best_model_inliers = inliers
            best_model = model

    return best_model, best_model_inliers
