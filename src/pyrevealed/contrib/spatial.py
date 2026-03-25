"""Spatial/Ideal point preference analysis via revealed preferences.

Implements finding a user's "ideal point" in feature space based on their choices,
using various distance metrics (Euclidean, Manhattan, etc.).

Based on Chapters 11.2-11.4 of Chambers & Echenique (2016) "Revealed Preference Theory".
The model is defined as:
    x ⪰i z iff d(x, yi) ≤ d(z, yi)
where yi is agent i's ideal point and d is the distance metric.

This module provides both heuristic (optimization-based) and exact (LP-based)
methods for testing Euclidean rationalizability:
- find_ideal_point(): Heuristic optimization-based ideal point finder
- check_euclidean_rationality_exact(): Exact LP-based test (Theorem 11.11)
- find_ideal_point_general(): General metric ideal point finder

Tech-Friendly Names (Primary):
    - find_preference_anchor(): Find user's ideal point in embedding space
    - find_ideal_point_general(): Find ideal point with any metric
    - determine_best_metric(): Model selection for metric type
    - check_euclidean_rationality_exact(): Exact test via Theorem 11.11

Economics Names (Legacy Aliases):
    - find_ideal_point() -> find_preference_anchor()

References:
    Chambers & Echenique (2016), Chapter 11, Theorem 11.11
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, linprog

from pyrevealed.core.session import SpatialSession
from pyrevealed.core.result import IdealPointResult, GeneralMetricResult
from pyrevealed.core.exceptions import SolverError


def find_ideal_point(
    session: SpatialSession,
    method: str = "SLSQP",
    max_iterations: int = 1000,
) -> IdealPointResult:
    """
    Find the ideal point that best explains user's choices.

    The ideal point model assumes the user prefers items closer to their
    ideal location in the feature space:
        U(item) = -||item - ideal_point||²

    For each choice set, the user should choose the item closest to their
    ideal point. See Chambers & Echenique (2016) Chapter 11, p.164.

    This function uses optimization to find the ideal point. For the exact
    revealed preference test (Theorem 11.11), one would check whether for all
    convex combinations λ with Σλk·yk = Σλk·nk, we have Σλk(yk·yk) < Σλk(nk·nk).

    Args:
        session: SpatialSession with item features and choice data
        method: Scipy optimization method ('SLSQP', 'L-BFGS-B', 'Powell')
        max_iterations: Maximum optimization iterations

    Returns:
        IdealPointResult with estimated ideal point and diagnostics

    Example:
        >>> import numpy as np
        >>> from pyrevealed import SpatialSession, find_ideal_point
        >>> features = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        >>> choice_sets = [[0, 1], [0, 2], [0, 3]]
        >>> choices = [0, 0, 0]  # Always chose item 0 (origin)
        >>> session = SpatialSession(features, choice_sets, choices)
        >>> result = find_ideal_point(session)
        >>> np.allclose(result.ideal_point, [0, 0], atol=0.1)
        True
    """
    start_time = time.perf_counter()

    T = session.num_observations

    # Initial guess: centroid of chosen items
    chosen_features = session.item_features[session.choices]
    x0 = np.mean(chosen_features, axis=0)

    # Objective: minimize sum of violations (hinge loss)
    def objective(ideal: NDArray[np.float64]) -> float:
        """Sum of constraint violations (hinge loss)."""
        total_loss = 0.0

        for t, (choice_set, chosen) in enumerate(
            zip(session.choice_sets, session.choices)
        ):
            chosen_feature = session.item_features[chosen]
            chosen_dist_sq = np.sum((ideal - chosen_feature) ** 2)

            for item_idx in choice_set:
                if item_idx != chosen:
                    other_feature = session.item_features[item_idx]
                    other_dist_sq = np.sum((ideal - other_feature) ** 2)

                    # Hinge loss: penalize if unchosen is closer
                    # margin = chosen_dist - other_dist (should be negative)
                    margin = np.sqrt(chosen_dist_sq) - np.sqrt(other_dist_sq)
                    if margin > 0:
                        total_loss += margin**2

        return total_loss

    # Optimize
    result = minimize(objective, x0, method=method, options={"maxiter": max_iterations})

    ideal_point = result.x

    # Compute violations and diagnostics
    violations = []
    distances_to_chosen = []

    for t, (choice_set, chosen) in enumerate(zip(session.choice_sets, session.choices)):
        chosen_feature = session.item_features[chosen]
        chosen_dist = np.linalg.norm(ideal_point - chosen_feature)
        distances_to_chosen.append(chosen_dist)

        for item_idx in choice_set:
            if item_idx != chosen:
                other_feature = session.item_features[item_idx]
                other_dist = np.linalg.norm(ideal_point - other_feature)

                # Violation if unchosen item is closer
                if other_dist < chosen_dist - 1e-10:
                    violations.append((t, item_idx))

    num_violations = len(violations)
    is_euclidean_rational = num_violations == 0

    # Explained variance: 1 - (violation rate)
    # Simple metric: fraction of choices that are "correct"
    num_correct = T - len(set(v[0] for v in violations))
    explained_variance = num_correct / T if T > 0 else 1.0

    mean_distance_to_chosen = (
        np.mean(distances_to_chosen) if distances_to_chosen else 0.0
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return IdealPointResult(
        ideal_point=ideal_point,
        is_euclidean_rational=is_euclidean_rational,
        violations=violations,
        num_violations=num_violations,
        explained_variance=explained_variance,
        mean_distance_to_chosen=mean_distance_to_chosen,
        computation_time_ms=elapsed_ms,
    )


def check_euclidean_rationality(
    session: SpatialSession,
) -> tuple[bool, list[tuple[int, int]]]:
    """
    Check if choices are consistent with some ideal point (Euclidean rationality).

    This is a quick check that finds violations without full optimization.

    Args:
        session: SpatialSession with choice data

    Returns:
        Tuple of (is_rational, violations) where violations is a list of
        (choice_set_idx, unchosen_item_idx) pairs
    """
    result = find_ideal_point(session)
    return result.is_euclidean_rational, result.violations


def check_euclidean_rationality_exact(
    session: SpatialSession,
    tolerance: float = 1e-8,
) -> dict:
    """
    Exact test for Euclidean rationalizability using Theorem 11.11.

    Tests whether the choice data can be rationalized by an ideal point
    model with Euclidean distance. This is based on the characterization
    in Chambers & Echenique (2016), Theorem 11.11.

    The test checks: for all convex combinations λ ≥ 0 satisfying:
        Σ_k λ_k · y_k = Σ_k λ_k · n_k
    we must have:
        Σ_k λ_k · ||y_k||² ≤ Σ_k λ_k · ||n_k||²

    where y_k is the chosen item and n_k is an unchosen item in choice set k.

    This is implemented by solving an LP: find λ ≥ 0 such that:
        Σ_k λ_k · (y_k - n_k) = 0  (equality constraint)
        Σ_k λ_k · (||y_k||² - ||n_k||²) > 0  (violation of rationality)

    If no such λ exists, the data is Euclidean rationalizable.

    Args:
        session: SpatialSession with item features and choice data
        tolerance: Numerical tolerance for LP

    Returns:
        Dictionary with:
        - 'is_euclidean_rational': Whether data satisfies Theorem 11.11
        - 'violating_combination': The λ weights if a violation was found
        - 'violation_magnitude': How badly the condition is violated
        - 'ideal_point': Recovered ideal point if rationalizable

    References:
        Chambers & Echenique (2016), Chapter 11, Theorem 11.11, p.166
    """
    start_time = time.perf_counter()

    T = session.num_observations
    D = session.item_features.shape[1]  # Dimension of feature space

    # Build pairs: for each choice set, pair the chosen with each unchosen
    pairs = []
    for t, (choice_set, chosen) in enumerate(zip(session.choice_sets, session.choices)):
        y_t = session.item_features[chosen]
        for unchosen_idx in choice_set:
            if unchosen_idx != chosen:
                n_t = session.item_features[unchosen_idx]
                pairs.append((y_t, n_t, t, chosen, unchosen_idx))

    K = len(pairs)

    if K == 0:
        # No comparisons possible
        return {
            "is_euclidean_rational": True,
            "violating_combination": None,
            "violation_magnitude": 0.0,
            "ideal_point": np.mean(session.item_features[session.choices], axis=0),
            "computation_time_ms": (time.perf_counter() - start_time) * 1000,
        }

    # LP to find violating combination
    # Variables: λ_0, ..., λ_{K-1}, slack_positive
    # We want to maximize Σ_k λ_k · (||y_k||² - ||n_k||²)
    # Subject to: Σ_k λ_k · (y_k - n_k) = 0 (D equality constraints)
    #             Σ_k λ_k = 1 (normalization)
    #             λ_k ≥ 0

    # Build constraint matrices
    # Equality constraints: Σ_k λ_k · (y_k[d] - n_k[d]) = 0 for each dimension d
    A_eq = np.zeros((D + 1, K))
    for k, (y_k, n_k, _, _, _) in enumerate(pairs):
        diff = y_k - n_k
        A_eq[:D, k] = diff  # Columns are (y_k - n_k)
    A_eq[D, :] = 1.0  # Sum of λ_k = 1

    b_eq = np.zeros(D + 1)
    b_eq[D] = 1.0  # Normalization

    # Objective: maximize Σ_k λ_k · (||y_k||² - ||n_k||²)
    # Equivalent to minimize -Σ_k λ_k · (||y_k||² - ||n_k||²)
    c = np.zeros(K)
    for k, (y_k, n_k, _, _, _) in enumerate(pairs):
        c[k] = -(np.dot(y_k, y_k) - np.dot(n_k, n_k))

    # Bounds: λ_k ≥ 0
    bounds = [(0, None)] * K

    try:
        result = linprog(
            c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )

        if result.success:
            lambdas = result.x
            # The objective is -Σ λ_k * (||y_k||² - ||n_k||²)
            # If this is negative, then Σ λ_k * (||y_k||² - ||n_k||²) > 0 (violation)
            violation_value = -result.fun

            if violation_value > tolerance:
                # Found a violating combination
                # Identify which pairs contribute most
                contributing_pairs = []
                for k, lamb in enumerate(lambdas):
                    if lamb > tolerance:
                        y_k, n_k, t, chosen, unchosen = pairs[k]
                        contributing_pairs.append({
                            "choice_set": t,
                            "chosen": chosen,
                            "unchosen": unchosen,
                            "weight": lamb,
                        })

                return {
                    "is_euclidean_rational": False,
                    "violating_combination": lambdas,
                    "violation_magnitude": violation_value,
                    "contributing_pairs": contributing_pairs,
                    "ideal_point": None,
                    "computation_time_ms": (time.perf_counter() - start_time) * 1000,
                }
            else:
                # No violation found - data is rationalizable
                # Recover ideal point using optimization
                ip_result = find_ideal_point(session)

                return {
                    "is_euclidean_rational": True,
                    "violating_combination": None,
                    "violation_magnitude": 0.0,
                    "ideal_point": ip_result.ideal_point,
                    "num_heuristic_violations": ip_result.num_violations,
                    "computation_time_ms": (time.perf_counter() - start_time) * 1000,
                }
        else:
            # LP infeasible - this means the equality constraints have no solution
            # with non-negative weights summing to 1, so vacuously satisfied
            ip_result = find_ideal_point(session)

            return {
                "is_euclidean_rational": True,
                "violating_combination": None,
                "violation_magnitude": 0.0,
                "ideal_point": ip_result.ideal_point,
                "lp_status": "infeasible_constraints",
                "computation_time_ms": (time.perf_counter() - start_time) * 1000,
            }

    except Exception as e:
        raise SolverError(
            f"LP solver failed during Euclidean rationality test. Original error: {e}"
        ) from e


def find_ideal_point_heuristic(
    session: SpatialSession,
    method: str = "SLSQP",
    max_iterations: int = 1000,
) -> IdealPointResult:
    """
    Find ideal point using heuristic optimization (hinge loss minimization).

    This is an alias for find_ideal_point(), renamed to clarify that it uses
    optimization-based heuristics rather than exact methods.

    For the exact revealed preference test, use check_euclidean_rationality_exact().

    Args:
        session: SpatialSession with item features and choice data
        method: Scipy optimization method
        max_iterations: Maximum optimization iterations

    Returns:
        IdealPointResult with estimated ideal point
    """
    return find_ideal_point(session, method=method, max_iterations=max_iterations)


def compute_preference_strength(
    session: SpatialSession,
    ideal_point: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute how strongly each choice matches the ideal point model.

    Returns a score for each choice where:
    - High positive = strong preference (chosen much closer than alternatives)
    - Near zero = close call
    - Negative = violation (chose farther item)

    Args:
        session: SpatialSession with choice data
        ideal_point: D-dimensional ideal point

    Returns:
        T-length array of preference strength scores
    """
    T = session.num_observations
    strengths = np.zeros(T)

    for t, (choice_set, chosen) in enumerate(zip(session.choice_sets, session.choices)):
        chosen_dist = np.linalg.norm(ideal_point - session.item_features[chosen])

        # Get minimum distance to unchosen items
        min_unchosen_dist = float("inf")
        for item_idx in choice_set:
            if item_idx != chosen:
                dist = np.linalg.norm(ideal_point - session.item_features[item_idx])
                min_unchosen_dist = min(min_unchosen_dist, dist)

        # Strength = (min_unchosen - chosen) / chosen
        # Positive means chosen was closer, negative means violation
        if chosen_dist > 1e-10:
            strengths[t] = (min_unchosen_dist - chosen_dist) / chosen_dist
        else:
            strengths[t] = min_unchosen_dist  # Ideal point is exactly at chosen

    return strengths


def find_multiple_ideal_points(
    session: SpatialSession,
    n_points: int = 2,
    method: str = "SLSQP",
) -> list[tuple[NDArray[np.float64], float]]:
    """
    Find multiple ideal points to explain inconsistent choices.

    If a single ideal point has many violations, multiple ideal points
    may better explain the observed choice patterns.

    Args:
        session: SpatialSession with choice data
        n_points: Number of ideal points to find
        method: Optimization method

    Returns:
        List of (ideal_point, explained_fraction) tuples, sorted by quality
    """
    T = session.num_observations

    results = []

    # Find first ideal point
    result1 = find_ideal_point(session, method=method)
    results.append((result1.ideal_point, result1.explained_variance))

    if n_points == 1:
        return results

    # For additional points, cluster the violations and find ideal points for each
    if result1.violations:
        # Get indices of violated choices
        violated_choice_indices = list(set(v[0] for v in result1.violations))

        # Create a sub-session with just the violated choices
        if len(violated_choice_indices) >= 2:
            sub_choice_sets = [session.choice_sets[i] for i in violated_choice_indices]
            sub_choices = [session.choices[i] for i in violated_choice_indices]

            sub_session = SpatialSession(
                item_features=session.item_features,
                choice_sets=sub_choice_sets,
                choices=sub_choices,
            )

            result2 = find_ideal_point(sub_session, method=method)
            explained = len(violated_choice_indices) / T * result2.explained_variance
            results.append((result2.ideal_point, explained))

    return results[:n_points]


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

# find_preference_anchor: Tech-friendly name for find_ideal_point
find_preference_anchor = find_ideal_point
"""
Find the user's preference anchor (ideal point) in embedding space.

This is the tech-friendly alias for find_ideal_point.

The preference anchor is the location in feature space that the user
seems to prefer. Items closer to this anchor are more likely to be chosen.

Use this for:
- Recommendation explainability ("You prefer items near this anchor")
- Personalization (recommend items close to anchor)
- Understanding user preference structure

Example:
    >>> from pyrevealed import EmbeddingChoiceLog, find_preference_anchor
    >>> result = find_preference_anchor(user_choices)
    >>> print(f"User's anchor: {result.ideal_point}")

Returns:
    PreferenceAnchorResult with ideal_point and explained_variance
"""

# validate_embedding_consistency: Tech-friendly name for check_euclidean_rationality
validate_embedding_consistency = check_euclidean_rationality
"""
Check if user choices are consistent in embedding space.

This is the tech-friendly alias for check_euclidean_rationality.

Verifies that the user's choices can be explained by a single preference
anchor. Inconsistency suggests multiple users or erratic behavior.
"""

# compute_signal_strength: Tech-friendly name for compute_preference_strength
compute_signal_strength = compute_preference_strength
"""
Compute the signal strength of user preferences.

This is the tech-friendly alias for compute_preference_strength.

Higher signal strength means clearer, more consistent preferences.
Low signal strength indicates noisy or random choices.
"""

# find_multiple_anchors: Tech-friendly name for find_multiple_ideal_points
find_multiple_anchors = find_multiple_ideal_points
"""
Find multiple preference anchors to explain inconsistent choices.

This is the tech-friendly alias for find_multiple_ideal_points.

If multiple anchors explain the data better than a single anchor,
this indicates heterogeneous preferences in the choice data.

Example:
    >>> from pyrevealed import EmbeddingChoiceLog, find_multiple_anchors
    >>> anchors = find_multiple_anchors(user_choices, n=2)
    >>> print(f"Found {len(anchors)} anchors")
"""


# =============================================================================
# GENERAL METRIC SPACES (Chapter 11.3-11.4)
# =============================================================================


def find_ideal_point_general(
    session: SpatialSession,
    metric: str = "L2",
    p: float = 2.0,
    method: str = "SLSQP",
    max_iterations: int = 1000,
    bounds: tuple[float, float] | None = None,
) -> GeneralMetricResult:
    """
    Find ideal point using a general distance metric.

    Extends the Euclidean preference model to arbitrary metrics:
    - L1 (Manhattan): d(x,y) = Σ|x_i - y_i|
    - L2 (Euclidean): d(x,y) = √(Σ(x_i - y_i)²)
    - Linf (Chebyshev): d(x,y) = max|x_i - y_i|
    - Minkowski: d(x,y) = (Σ|x_i - y_i|^p)^(1/p)

    Args:
        session: SpatialSession with item features and choice data
        metric: Distance metric ("L1", "L2", "Linf", "minkowski")
        p: Minkowski parameter (p=1 -> L1, p=2 -> L2, p=inf -> Linf)
        method: Scipy optimization method
        max_iterations: Maximum optimization iterations
        bounds: Optional (min, max) bounds for each dimension. If None, bounds
            are inferred from the data range with 10% padding.

    Returns:
        GeneralMetricResult with ideal point and metric analysis

    Example:
        >>> from pyrevealed import SpatialSession, find_ideal_point_general
        >>> result = find_ideal_point_general(session, metric="L1")
        >>> print(f"Ideal point: {result.ideal_point}")
        >>> print(f"Violations: {result.num_violations}")

    References:
        Chambers & Echenique (2016), Chapter 11.3-11.4
    """
    start_time = time.perf_counter()

    T = session.num_observations
    D = session.item_features.shape[1]  # Number of dimensions

    # Initial guess: centroid of chosen items
    chosen_features = session.item_features[session.choices]
    x0 = np.mean(chosen_features, axis=0)

    # Compute bounds if not provided
    if bounds is None:
        # Infer bounds from data with 10% padding
        feature_min = np.min(session.item_features, axis=0)
        feature_max = np.max(session.item_features, axis=0)
        feature_range = feature_max - feature_min
        padding = 0.1 * np.maximum(feature_range, 1e-6)  # At least some padding
        scipy_bounds = list(zip(feature_min - padding, feature_max + padding))
    else:
        scipy_bounds = [bounds] * D

    # Distance function based on metric
    def distance(x: NDArray, y: NDArray) -> float:
        if metric == "L1":
            return np.sum(np.abs(x - y))
        elif metric == "L2":
            return np.sqrt(np.sum((x - y) ** 2))
        elif metric == "Linf":
            return np.max(np.abs(x - y))
        elif metric == "minkowski":
            return np.sum(np.abs(x - y) ** p) ** (1 / p)
        else:
            return np.sqrt(np.sum((x - y) ** 2))  # Default to L2

    # Objective: minimize violations
    def objective(ideal: NDArray[np.float64]) -> float:
        total_loss = 0.0

        for t, (choice_set, chosen) in enumerate(
            zip(session.choice_sets, session.choices)
        ):
            chosen_feature = session.item_features[chosen]
            chosen_dist = distance(ideal, chosen_feature)

            for item_idx in choice_set:
                if item_idx != chosen:
                    other_feature = session.item_features[item_idx]
                    other_dist = distance(ideal, other_feature)

                    # Hinge loss
                    margin = chosen_dist - other_dist
                    if margin > 0:
                        total_loss += margin ** 2

        return total_loss

    # Optimize with bounds
    result = minimize(
        objective, x0, method=method,
        bounds=scipy_bounds,
        options={"maxiter": max_iterations}
    )

    ideal_point = result.x

    # Compute violations with NaN-safe distance checks
    violations = []
    for t, (choice_set, chosen) in enumerate(zip(session.choice_sets, session.choices)):
        chosen_dist = distance(ideal_point, session.item_features[chosen])

        # Skip if chosen_dist is NaN or infinite
        if not np.isfinite(chosen_dist):
            continue

        for item_idx in choice_set:
            if item_idx != chosen:
                other_dist = distance(ideal_point, session.item_features[item_idx])
                # Check for NaN/Inf before comparison
                if np.isfinite(other_dist) and other_dist < chosen_dist - 1e-10:
                    violations.append((t, item_idx))

    is_rationalizable = len(violations) == 0

    # Compute explained variance
    num_correct = T - len(set(v[0] for v in violations))
    explained_variance = num_correct / T if T > 0 else 1.0

    # Set metric params
    metric_params = {"p": p} if metric == "minkowski" else {}

    computation_time = (time.perf_counter() - start_time) * 1000

    return GeneralMetricResult(
        ideal_point=ideal_point,
        metric_type=metric,
        metric_params=metric_params,
        is_rationalizable=is_rationalizable,
        violations=violations,
        best_metric=metric,
        metric_comparison={metric: len(violations)},
        explained_variance=explained_variance,
        computation_time_ms=computation_time,
    )


def determine_best_metric(
    session: SpatialSession,
    candidates: list[str] | None = None,
    method: str = "SLSQP",
) -> GeneralMetricResult:
    """
    Find the distance metric that best explains the choice data.

    Tries multiple metrics and selects the one with fewest violations.

    Args:
        session: SpatialSession with item features and choice data
        candidates: List of metrics to try (default: ["L1", "L2", "Linf"])
        method: Optimization method

    Returns:
        GeneralMetricResult for the best metric

    Example:
        >>> from pyrevealed import SpatialSession, determine_best_metric
        >>> result = determine_best_metric(session)
        >>> print(f"Best metric: {result.best_metric}")
        >>> print(f"Violations: {result.num_violations}")
    """
    if candidates is None:
        candidates = ["L1", "L2", "Linf"]

    results = {}
    best_result = None
    best_violations = float("inf")

    for metric in candidates:
        result = find_ideal_point_general(session, metric=metric, method=method)
        results[metric] = len(result.violations)

        if len(result.violations) < best_violations:
            best_violations = len(result.violations)
            best_result = result

    # Update best result with comparison info
    if best_result is not None:
        # Create new result with metric comparison
        return GeneralMetricResult(
            ideal_point=best_result.ideal_point,
            metric_type=best_result.metric_type,
            metric_params=best_result.metric_params,
            is_rationalizable=best_result.is_rationalizable,
            violations=best_result.violations,
            best_metric=best_result.metric_type,
            metric_comparison=results,
            explained_variance=best_result.explained_variance,
            computation_time_ms=best_result.computation_time_ms,
        )

    # Fallback to L2
    return find_ideal_point_general(session, metric="L2", method=method)


def test_metric_rationality(
    session: SpatialSession,
    metric: str = "L2",
    p: float = 2.0,
) -> tuple[bool, list[tuple[int, int]]]:
    """
    Test if choices are rationalizable under a given metric.

    Args:
        session: SpatialSession with choice data
        metric: Distance metric to use
        p: Minkowski parameter (if metric="minkowski")

    Returns:
        Tuple of (is_rational, violations)
    """
    result = find_ideal_point_general(session, metric=metric, p=p)
    return result.is_rationalizable, result.violations


# =============================================================================
# ADDITIONAL TECH-FRIENDLY ALIASES
# =============================================================================

find_anchor_general = find_ideal_point_general
"""
Find preference anchor using any distance metric.

Tech-friendly alias for find_ideal_point_general.
"""

select_best_metric = determine_best_metric
"""
Select the best distance metric for the data.

Tech-friendly alias for determine_best_metric.
"""
