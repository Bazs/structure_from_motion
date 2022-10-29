import logging
import random
from math import inf
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple


def fit_with_ransac(
    data: Sequence,
    model_fit_data_count: int,
    model_fitter: Callable[[Sequence], Any],
    inlier_scorer: Callable[[Any, Any], float],
    inlier_threshold: float,
    max_iterations: int = 1000,
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
    best_model_error = inf

    for _ in range(max_iterations):
        samples = random.sample(data, k=model_fit_data_count)
        model = model_fitter(samples)
        data_scores = [inlier_scorer(model, data_point) for data_point in data]
        logging.info("Data scores: %s", data_scores)
        inliers = [
            data_point
            for data_point, inlier_score in zip(data, data_scores)
            if inlier_score <= inlier_threshold
        ]
        if model_fit_data_count <= len(inliers) > len(best_model_inliers):
            model_error = sum(
                [
                    inler_score
                    for inler_score in data_scores
                    if inler_score <= inlier_threshold
                ]
            )
            if model_error < best_model_error:
                best_model_inliers = inliers
                best_model = model
                best_model_error = model_error

    if best_model is None:
        raise ValueError(
            f"No model could be found with at least {model_fit_data_count} inliers. Check model_fitter, "
            "inlier_scorer, and inlier_threshold, as this can only happen if there is a mistake in those arguments."
        )

    return best_model, best_model_inliers
