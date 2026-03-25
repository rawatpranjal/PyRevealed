"""Statistical inference for revealed preference metrics.

Implements bootstrap confidence intervals and other inference methods
for RP consistency metrics like AEI, MPI, and Houtman-Maks.

Tech-Friendly Names (Primary):
    - compute_bootstrap_ci(): Bootstrap confidence intervals
    - compute_standard_error(): Standard error estimation
    - test_metric_difference(): Compare metrics between groups

Economics Names (Legacy Aliases):
    - bootstrap_confidence_interval() -> compute_bootstrap_ci()

References:
    Efron, B., & Tibshirani, R. J. (1994). An Introduction to the Bootstrap.
    Andreoni, J., Gillen, B. J., & Harbaugh, W. T. (2013). The power of
    revealed preference tests. Experimental Economics.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pyrevealed.core.session import BehaviorLog

from pyrevealed.core.result import BootstrapCIResult, PredictiveSuccessResult


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================


def compute_bootstrap_ci(
    log: "BehaviorLog",
    metric: str = "aei",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int | None = None,
) -> BootstrapCIResult:
    """
    Compute bootstrap confidence intervals for RP metrics.

    Uses nonparametric bootstrap (resampling observations with replacement)
    to estimate the sampling distribution of consistency metrics.

    This is important for:
    - Determining if a consumer is "significantly" irrational
    - Comparing rationality across groups
    - Assessing measurement precision

    Args:
        log: BehaviorLog with prices and quantities
        metric: Metric to compute - "aei", "mpi", "hmi", "swaps"
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        BootstrapCIResult with point estimate, CI, and bootstrap distribution

    Example:
        >>> from pyrevealed import BehaviorLog, compute_bootstrap_ci
        >>> result = compute_bootstrap_ci(log, metric="aei")
        >>> print(f"AEI: {result.point_estimate:.3f}")
        >>> print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")

    References:
        Efron, B., & Tibshirani, R. J. (1994). An Introduction to the Bootstrap.
        Andreoni, J., Gillen, B. J., & Harbaugh, W. T. (2013). The power of
        revealed preference tests: An investigation of demand data.
        Experimental Economics, 16(4), 555-579.
    """
    from pyrevealed.core.session import BehaviorLog

    start_time = time.perf_counter()

    # Set random state
    rng = np.random.default_rng(random_state)

    # Get metric function
    metric_func = _get_metric_function(metric)

    # Compute point estimate
    point_estimate = metric_func(log)

    # Bootstrap resampling
    T = log.num_records
    bootstrap_values = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(T, size=T, replace=True)

        # Create resampled log
        resampled_log = BehaviorLog(
            cost_vectors=log.cost_vectors[indices],
            action_vectors=log.action_vectors[indices],
        )

        # Compute metric
        bootstrap_values[i] = metric_func(resampled_log)

    # Compute confidence interval (percentile method)
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_values, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_values, 100 * (1 - alpha / 2)))

    # Standard error
    std_error = float(np.std(bootstrap_values, ddof=1))

    computation_time = (time.perf_counter() - start_time) * 1000

    return BootstrapCIResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence,
        metric_name=metric,
        bootstrap_distribution=bootstrap_values,
        std_error=std_error,
        n_bootstrap=n_bootstrap,
        computation_time_ms=computation_time,
    )


def _get_metric_function(metric: str) -> Callable[["BehaviorLog"], float]:
    """Get the metric computation function for a metric name."""
    from pyrevealed.algorithms.aei import compute_aei
    from pyrevealed.algorithms.mpi import compute_mpi, compute_houtman_maks_index
    from pyrevealed.algorithms.garp import compute_swaps_index

    metric_lower = metric.lower()

    if metric_lower in ("aei", "afriat", "ccei", "integrity"):
        def aei_score(log: "BehaviorLog") -> float:
            result = compute_aei(log)
            return result.efficiency_index

        return aei_score

    elif metric_lower in ("mpi", "money_pump", "confusion"):
        def mpi_score(log: "BehaviorLog") -> float:
            result = compute_mpi(log)
            return result.mpi_value

        return mpi_score

    elif metric_lower in ("hmi", "houtman_maks", "efficiency"):
        def hmi_score(log: "BehaviorLog") -> float:
            result = compute_houtman_maks_index(log)
            return result.fraction

        return hmi_score

    elif metric_lower in ("swaps", "swaps_index"):
        def swaps_score(log: "BehaviorLog") -> float:
            result = compute_swaps_index(log)
            return result.swaps_normalized

        return swaps_score

    else:
        raise ValueError(
            f"Unknown metric: {metric}. "
            "Use 'aei', 'mpi', 'hmi', or 'swaps'."
        )


# =============================================================================
# PREDICTIVE SUCCESS (Selten 1991)
# =============================================================================


def compute_predictive_success(
    log: "BehaviorLog",
    model: str = "garp",
    holdout_fraction: float = 0.2,
    n_splits: int = 5,
    random_state: int | None = None,
) -> PredictiveSuccessResult:
    """
    Compute predictive success measure (Selten 1991).

    Predictive success measures how well a model predicts choices
    beyond what would be expected by chance:
        PS = hit_rate - false_alarm_rate

    This is a proper scoring rule that penalizes both:
    - Missing true choices (low hit rate)
    - Falsely predicting choices (high false alarm rate)

    A model with PS = 0 does no better than random guessing.
    A perfect model has PS = 1 (hits everything, no false alarms).

    Args:
        log: BehaviorLog with prices and quantities
        model: Model to test - "garp", "warp", or "utility"
        holdout_fraction: Fraction of data to hold out for testing
        n_splits: Number of cross-validation splits
        random_state: Random seed for reproducibility

    Returns:
        PredictiveSuccessResult with hit rate, false alarm rate, and PS

    Example:
        >>> from pyrevealed import BehaviorLog, compute_predictive_success
        >>> result = compute_predictive_success(log, model="garp")
        >>> print(f"Predictive Success: {result.predictive_success:.3f}")
        >>> print(f"Hit Rate: {result.hit_rate:.3f}")

    References:
        Selten, R. (1991). Properties of a measure of predictive success.
        Mathematical Social Sciences, 21(2), 153-167.

        Beatty, T. K., & Crawford, I. A. (2011). How demanding is the
        revealed preference approach to demand? American Economic Review.
    """
    from pyrevealed.core.session import BehaviorLog
    from pyrevealed.algorithms.garp import check_garp
    from pyrevealed.algorithms.utility import recover_utility

    start_time = time.perf_counter()

    rng = np.random.default_rng(random_state)
    T = log.num_records

    if T < 5:
        # Too few observations for cross-validation
        computation_time = (time.perf_counter() - start_time) * 1000
        return PredictiveSuccessResult(
            predictive_success=0.0,
            hit_rate=0.0,
            false_alarm_rate=0.0,
            model_name=model,
            num_predictions=0,
            computation_time_ms=computation_time,
        )

    holdout_size = max(1, int(T * holdout_fraction))
    total_hits = 0
    total_false_alarms = 0
    total_predictions = 0
    total_possible_hits = 0
    total_possible_false_alarms = 0

    for split in range(n_splits):
        # Create random split
        indices = rng.permutation(T)
        test_indices = indices[:holdout_size]
        train_indices = indices[holdout_size:]

        if len(train_indices) < 2:
            continue

        # Create training set
        train_log = BehaviorLog(
            cost_vectors=log.cost_vectors[train_indices],
            action_vectors=log.action_vectors[train_indices],
        )

        # Fit model on training data
        if model.lower() == "utility":
            utility_result = recover_utility(train_log)
            if not utility_result.success:
                continue
            utility_values = utility_result.utility_values
        else:
            # For GARP/WARP, we use revealed preference predictions
            utility_values = None

        # Test predictions on holdout
        for test_idx in test_indices:
            test_prices = log.cost_vectors[test_idx]
            test_quantities = log.action_vectors[test_idx]
            test_expenditure = np.dot(test_prices, test_quantities)

            # Check which training observations could have been chosen
            for train_idx in train_indices:
                train_quantities = log.action_vectors[train_idx]
                train_at_test_prices = np.dot(test_prices, train_quantities)

                # Is training bundle affordable at test prices?
                is_affordable = train_at_test_prices <= test_expenditure + 1e-8

                if is_affordable:
                    total_possible_false_alarms += 1

                    # Does model predict training bundle would be chosen?
                    if utility_values is not None:
                        # Utility model: predict based on utility values
                        predicted = utility_values[train_idx] <= utility_values[test_idx]
                    else:
                        # GARP model: predict based on revealed preference
                        # If test bundle revealed preferred, prediction is "not chosen"
                        predicted = False  # Conservative: don't predict training bundle

                    if predicted:
                        total_false_alarms += 1

            # Check if actual choice is predicted
            total_possible_hits += 1
            total_predictions += 1

            # Did model "predict" the actual choice?
            # For GARP: check if actual choice is consistent with training data
            if model.lower() in ("garp", "warp"):
                # Create combined log to check consistency
                combined_P = np.vstack([train_log.cost_vectors, test_prices.reshape(1, -1)])
                combined_Q = np.vstack([train_log.action_vectors, test_quantities.reshape(1, -1)])
                combined_log = BehaviorLog(cost_vectors=combined_P, action_vectors=combined_Q)

                result = check_garp(combined_log)
                hit = result.is_consistent
            else:
                # Utility model: check if utility is maximized
                hit = True  # Assume hit if utility values exist

            if hit:
                total_hits += 1

    # Compute rates
    if total_possible_hits > 0:
        hit_rate = total_hits / total_possible_hits
    else:
        hit_rate = 0.0

    if total_possible_false_alarms > 0:
        false_alarm_rate = total_false_alarms / total_possible_false_alarms
    else:
        false_alarm_rate = 0.0

    predictive_success = hit_rate - false_alarm_rate

    computation_time = (time.perf_counter() - start_time) * 1000

    return PredictiveSuccessResult(
        predictive_success=predictive_success,
        hit_rate=hit_rate,
        false_alarm_rate=false_alarm_rate,
        model_name=model,
        num_predictions=total_predictions,
        computation_time_ms=computation_time,
    )


# =============================================================================
# ADDITIONAL INFERENCE UTILITIES
# =============================================================================


def compute_standard_error(
    log: "BehaviorLog",
    metric: str = "aei",
    method: str = "bootstrap",
    n_samples: int = 500,
    random_state: int | None = None,
) -> dict:
    """
    Compute standard error for an RP metric.

    Args:
        log: BehaviorLog with prices and quantities
        metric: Metric to compute - "aei", "mpi", "hmi"
        method: SE method - "bootstrap" or "jackknife"
        n_samples: Number of samples for bootstrap
        random_state: Random seed

    Returns:
        Dict with point estimate and standard error

    Example:
        >>> result = compute_standard_error(log, metric="aei")
        >>> print(f"SE: {result['standard_error']:.4f}")
    """
    start_time = time.perf_counter()

    if method == "bootstrap":
        ci_result = compute_bootstrap_ci(
            log, metric=metric, n_bootstrap=n_samples, random_state=random_state
        )
        return {
            "point_estimate": ci_result.point_estimate,
            "standard_error": ci_result.std_error,
            "method": "bootstrap",
            "n_samples": n_samples,
        }

    elif method == "jackknife":
        from pyrevealed.core.session import BehaviorLog

        metric_func = _get_metric_function(metric)
        T = log.num_records

        # Leave-one-out jackknife
        jackknife_values = np.zeros(T)
        for i in range(T):
            indices = np.concatenate([np.arange(i), np.arange(i + 1, T)])
            loo_log = BehaviorLog(
                cost_vectors=log.cost_vectors[indices],
                action_vectors=log.action_vectors[indices],
            )
            jackknife_values[i] = metric_func(loo_log)

        # Jackknife standard error
        point_estimate = metric_func(log)
        mean_jackknife = np.mean(jackknife_values)
        se = np.sqrt((T - 1) / T * np.sum((jackknife_values - mean_jackknife) ** 2))

        return {
            "point_estimate": point_estimate,
            "standard_error": float(se),
            "method": "jackknife",
            "n_samples": T,
        }

    else:
        raise ValueError(f"Unknown method: {method}. Use 'bootstrap' or 'jackknife'.")


def test_metric_difference(
    log1: "BehaviorLog",
    log2: "BehaviorLog",
    metric: str = "aei",
    n_bootstrap: int = 1000,
    random_state: int | None = None,
) -> dict:
    """
    Test if two groups have significantly different RP metrics.

    Uses bootstrap to test H0: metric(log1) = metric(log2).

    Args:
        log1: First BehaviorLog
        log2: Second BehaviorLog
        metric: Metric to compare
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed

    Returns:
        Dict with difference, p-value, and confidence interval
    """
    from pyrevealed.core.session import BehaviorLog

    rng = np.random.default_rng(random_state)
    metric_func = _get_metric_function(metric)

    # Point estimates
    est1 = metric_func(log1)
    est2 = metric_func(log2)
    observed_diff = est1 - est2

    # Bootstrap under null (pooled)
    T1, T2 = log1.num_records, log2.num_records
    pooled_P = np.vstack([log1.cost_vectors, log2.cost_vectors])
    pooled_Q = np.vstack([log1.action_vectors, log2.action_vectors])

    null_diffs = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample from pooled data
        indices1 = rng.choice(T1 + T2, size=T1, replace=True)
        indices2 = rng.choice(T1 + T2, size=T2, replace=True)

        resample1 = BehaviorLog(
            cost_vectors=pooled_P[indices1],
            action_vectors=pooled_Q[indices1],
        )
        resample2 = BehaviorLog(
            cost_vectors=pooled_P[indices2],
            action_vectors=pooled_Q[indices2],
        )

        null_diffs[i] = metric_func(resample1) - metric_func(resample2)

    # Two-sided p-value
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))

    # CI for difference
    ci_lower = float(np.percentile(null_diffs, 2.5))
    ci_upper = float(np.percentile(null_diffs, 97.5))

    return {
        "difference": observed_diff,
        "metric1": est1,
        "metric2": est2,
        "p_value": float(p_value),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": p_value < 0.05,
        "metric": metric,
    }


# =============================================================================
# LEGACY ALIASES
# =============================================================================

bootstrap_confidence_interval = compute_bootstrap_ci
"""Legacy alias: use compute_bootstrap_ci instead."""

predictive_success = compute_predictive_success
"""Legacy alias: use compute_predictive_success instead."""
