from typing import Any, Callable, Sequence


def fit_with_ransac(
    data: Sequence,
    model_fit_data_count: int,
    model_fitter: Callable[[Sequence], Any],
    inlier_scorer: Callable[..., float],
    inlier_threshold: float,
    min_inlier_count: int,
):
    pass
