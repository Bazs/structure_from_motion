import copy
import logging
import random
from enum import Enum
from math import inf
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import tqdm


class ErrorAggregationMethod(Enum):
    SUM = "sum"
    SQUARE = "square"
    MEAN = "mean"
    RMS = "rms"


def fit_with_ransac(
    data: Sequence,
    model_fit_data_count: int,
    model_fitter: Callable[[Sequence], Any],
    inlier_scorer: Callable[[Any, Any], float],
    inlier_threshold: float,
    min_num_extra_inliers: int | None = None,
    error_aggregation_method: ErrorAggregationMethod | None = None,
    max_iterations: int | None = None,
) -> Tuple[Optional[Any], Sequence]:
    """Fit a model using RANSAC. Use the built-in random module for selecting candidates to fit model on.

    Args:
        data: A sequence of data points.
        model_fit_data_count: The minimum amount of data points required to fit the model.
        model_fitter: A callable which takes model_fit_data_count elements of the type of the data input sequence,
            and returns an object representing the fitted model.
        inlier_scorer: A callable with the signature (model, data_point) -> float, which evaluates how well any
            element of data fits the model returned by model_fitter.
        inlier_threshold: If inlier_scorer returns a value at most this, then the corresponding element of data
            is considered to fit the model under evaluation.
        min_num_extra_inliers: The minimum number of inliers from among the data points not used to fit the model,
            which should have an error below inlier_threshold in order to consider the model a good fit. Default is 0.
        error_aggregation_method: Method used to aggregate the score before comparing models. Default is RMS.
        max_iterations: The number of iterations to run the algorithm for.
    Returns:
        Tuple of {the model returned by model_fitter which has the most inliers, the elements of data which fit the
            returned model}.
    """
    if max_iterations is None:
        max_iterations = 100
    if error_aggregation_method is None:
        error_aggregation_method = ErrorAggregationMethod.RMS
    if min_num_extra_inliers is None:
        min_num_extra_inliers = 0

    best_model = None
    best_model_inliers = []
    best_model_error = inf

    data = copy.deepcopy(data)

    for _ in tqdm.tqdm(range(max_iterations), "RANSAC iterations"):
        random.shuffle(data)
        maybe_inliers = data[:model_fit_data_count]
        rest_of_data = data[model_fit_data_count:]
        model = model_fitter(maybe_inliers)
        rest_of_data_scores = [
            inlier_scorer(model, data_point) for data_point in rest_of_data
        ]
        logging.debug("Data scores: %s", rest_of_data_scores)
        also_inliers = [
            data_point
            for data_point, score in zip(rest_of_data, rest_of_data_scores)
            if score <= inlier_threshold
        ]
        if min_num_extra_inliers <= len(also_inliers):
            all_inliers = maybe_inliers + also_inliers
            inlier_errors = [
                inlier_scorer(model, data_point) for data_point in all_inliers
            ]
            model_error = _aggregate_error(
                inlier_errors, aggregation_method=error_aggregation_method
            )
            if model_error < best_model_error:
                best_model_inliers = all_inliers
                best_model = model
                best_model_error = model_error

    if best_model is None:
        raise ValueError(
            f"No model could be found with at least {min_num_extra_inliers + model_fit_data_count} inliers."
        )

    return best_model, best_model_inliers


def _aggregate_error(
    errors: list[float], aggregation_method: ErrorAggregationMethod
) -> float:
    if ErrorAggregationMethod.SUM.value == aggregation_method.value:
        return sum(errors)
    elif ErrorAggregationMethod.SQUARE.value == aggregation_method.value:
        return np.sum(np.square(errors)).item()
    elif ErrorAggregationMethod.MEAN.value == aggregation_method.value:
        return np.mean(errors).item()
    elif ErrorAggregationMethod.RMS.value == aggregation_method.value:
        return np.sqrt(np.mean(np.square(errors))).item()
    else:
        raise NotImplementedError(aggregation_method)
