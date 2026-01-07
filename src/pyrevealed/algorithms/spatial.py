"""Spatial/Ideal point preference analysis via revealed preferences.

Implements finding a user's "ideal point" in feature space based on their choices,
using Euclidean preference model (prefer items closer to ideal point).

Based on the Euclidean preference model from Chambers & Echenique (2016)
Chapter 11, Section 11.2.1 (pp.164-172). The model is defined as:
    x ⪰i z iff ||x - yi|| ≤ ||z - yi||
where yi is agent i's ideal point.

Note: This implementation uses optimization to find the ideal point that
minimizes choice violations. For the exact revealed preference test, see
Theorem 11.11 (pp.171-172) which provides an algebraic condition for
Euclidean rationalizability.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from pyrevealed.core.session import SpatialSession
from pyrevealed.core.result import IdealPointResult


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
    Find multiple ideal points (e.g., for shared account detection).

    If a single ideal point has many violations, the account might be shared
    by multiple users with different preferences.

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
- Detecting account sharing (multiple anchors = multiple users)

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
Find multiple preference anchors (for shared accounts/multi-user detection).

This is the tech-friendly alias for find_multiple_ideal_points.

If multiple anchors explain the data well, this suggests the account
is shared by multiple users with different preferences.

Example:
    >>> from pyrevealed import EmbeddingChoiceLog, find_multiple_anchors
    >>> anchors = find_multiple_anchors(user_choices, n=2)
    >>> if len(anchors) > 1 and anchors[1][1] > 0.3:
    ...     flag_as_shared_account(user_id)
"""
