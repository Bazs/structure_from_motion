import logging
import random
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

# _logger = logging.getLogger(Path(__file__).name)


def fit_with_ransac(
    data: Sequence,
    model_fit_data_count: int,
    model_fitter: Callable[[Sequence], Any],
    inlier_scorer: Callable[[Any, Any], float],
    inlier_threshold: float,
    max_iterations: int = 100,
) -> Tuple[Optional[Any], Sequence]:
    """Fit a model using RANSAC.

    Args:
        data: A sequence of data points.
        model_fit_data_count: The minimum amount of data points required to fit the model.
        model_fitter: A callable which takes model_fit_data_count elements of the type of the data input sequence,
            and returns an object representing the fitted model.
        inlier_scorer: A callable with the signature (model, data_point) -> float, which evaluates how well any
            element of data fits the model returned by model_fitter.
        inlier_threshold: If inlier_scorer returns a value at most this, then the corresponding element of data
            is considered to fit the model under evaluation.
        max_iterations: The number of iterations to run the algorithm for.
    Returns:
        Tuple of {the model returned by model_fitter which has the most inliers, the elements of data which fit the
            returned model}.
    """
    best_model = None
    best_model_inliers = []

    for _ in range(max_iterations):
        samples = random.sample(data, k=model_fit_data_count)
        model = model_fitter(samples)
        inlier_scores = [inlier_scorer(model, data_point) for data_point in data]
        logging.info("Inlier scores: %s", inlier_scores)
        inliers = [
            data_point
            for data_point, inlier_score in zip(data, inlier_scores)
            if inlier_score <= inlier_threshold
        ]
        if len(inliers) > len(best_model_inliers):
            best_model_inliers = inliers
            best_model = model

    if best_model is None:
        raise ValueError(
            "No model could be found with at least one inlier. Check model_fitter, inlier_scorer, and"
            " inlier_threshold, as this can only happen if there is a mistake in those arguments."
        )

    return best_model, best_model_inliers
