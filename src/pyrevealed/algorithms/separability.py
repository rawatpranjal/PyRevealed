"""Separability testing for groups of goods.

Tests whether utility can be decomposed into independent sub-utilities
for different groups of goods (weak separability).

IMPORTANT: This module uses a HEURISTIC APPROXIMATION for separability testing.
The exact test from Chambers & Echenique (2016) Chapter 4, Theorem 4.4 (pp.63-64)
requires solving nonlinear Afriat inequalities:

    Uk ≤ Ul + λl·p¹l·(x¹k - x¹l) + (λl/μl)·(Vk - Vl)    (4.1)
    Vk ≤ Vl + μl·p²l·(x²k - x²l)                         (4.2)

These inequalities are nonlinear (unlike standard Afriat inequalities) and
require specialized solvers. This implementation instead uses:
1. AEI (Afriat Efficiency Index) within each group
2. Cross-correlation between groups

This is a practical approximation that works well in many cases but is NOT
equivalent to the exact Theorem 4.4 test.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import SeparabilityResult
from pyrevealed.core.exceptions import DataValidationError, ValueRangeError
from pyrevealed.algorithms.garp import check_garp
from pyrevealed.algorithms.aei import compute_aei


def check_separability(
    session: ConsumerSession,
    group_a: list[int],
    group_b: list[int],
    tolerance: float = 1e-6,
) -> SeparabilityResult:
    """
    Test if two groups of goods are weakly separable (HEURISTIC APPROXIMATION).

    Weak separability means the utility function can be written as:
        U(x_A, x_B) = V(u_A(x_A), u_B(x_B))

    where x_A and x_B are consumption of goods in groups A and B.

    If separable, the groups can be priced independently without considering
    cross-elasticity effects.

    WARNING: This is a HEURISTIC approximation, not the exact Theorem 4.4 test
    from Chambers & Echenique (2016). The heuristic checks:
    1. Within-group GARP consistency (via AEI) for each group
    2. Low cross-correlation between groups

    The exact test (Theorem 4.4) requires solving nonlinear Afriat inequalities,
    which is computationally harder. See Cherchye et al. (2014) for algorithms.

    Args:
        session: ConsumerSession with prices and quantities
        group_a: List of good indices in Group A
        group_b: List of good indices in Group B
        tolerance: Numerical tolerance for GARP checks

    Returns:
        SeparabilityResult with separability test results

    Example:
        >>> import numpy as np
        >>> from pyrevealed import ConsumerSession, test_separability
        >>> # Rides (goods 0,1) and Eats (goods 2,3)
        >>> prices = np.array([
        ...     [1.0, 1.5, 2.0, 2.5],
        ...     [1.5, 1.0, 2.5, 2.0],
        ... ])
        >>> quantities = np.array([
        ...     [2.0, 1.0, 1.0, 0.5],
        ...     [1.0, 2.0, 0.5, 1.0],
        ... ])
        >>> session = ConsumerSession(prices=prices, quantities=quantities)
        >>> result = test_separability(session, [0, 1], [2, 3])
        >>> result.is_separable
        True
    """
    start_time = time.perf_counter()

    # Validate groups don't overlap and cover all goods
    all_indices = set(group_a) | set(group_b)
    if len(all_indices) != len(group_a) + len(group_b):
        overlap = set(group_a) & set(group_b)
        raise DataValidationError(
            f"Groups must not overlap. Found overlapping indices: {list(overlap)}. "
            f"Hint: Each good should belong to exactly one group for separability testing."
        )

    N = session.num_goods
    for idx in all_indices:
        if idx < 0 or idx >= N:
            raise ValueRangeError(
                f"Good index {idx} out of range [0, {N}). "
                f"Hint: Indices must refer to valid goods in the session (0 to {N-1})."
            )

    # Create sub-sessions for each group
    session_a = _extract_subsession(session, group_a)
    session_b = _extract_subsession(session, group_b)

    # Check GARP within each group
    aei_a = compute_aei(session_a, tolerance=tolerance)
    aei_b = compute_aei(session_b, tolerance=tolerance)

    # Compute cross-effect strength
    cross_effect = _compute_cross_effect(session, group_a, group_b)

    # Separability test:
    # 1. Each group should satisfy GARP internally (or close to it)
    # 2. Cross-effects should be minimal
    within_a_consistent = aei_a.efficiency_index
    within_b_consistent = aei_b.efficiency_index

    # Separable if both groups are internally consistent and cross-effects are low
    is_separable = (
        within_a_consistent > 0.9 and
        within_b_consistent > 0.9 and
        cross_effect < 0.2
    )

    # Generate recommendation
    if is_separable:
        recommendation = "price_independently"
    elif cross_effect > 0.5:
        recommendation = "unified_strategy"
    else:
        recommendation = "partial_independence"

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return SeparabilityResult(
        is_separable=is_separable,
        group_a_indices=list(group_a),
        group_b_indices=list(group_b),
        cross_effect_strength=cross_effect,
        within_group_a_consistency=within_a_consistent,
        within_group_b_consistency=within_b_consistent,
        recommendation=recommendation,
        computation_time_ms=elapsed_ms,
    )


def _extract_subsession(
    session: ConsumerSession,
    good_indices: list[int],
) -> ConsumerSession:
    """Extract a sub-session with only specified goods."""
    prices = session.prices[:, good_indices]
    quantities = session.quantities[:, good_indices]
    return ConsumerSession(prices=prices, quantities=quantities)


def _compute_cross_effect(
    session: ConsumerSession,
    group_a: list[int],
    group_b: list[int],
) -> float:
    """
    Compute cross-price effect between groups.

    Measures how much prices in one group affect quantities in the other.
    Returns a value in [0, 1] where 0 = no cross-effect, 1 = strong effect.
    """
    T = session.num_observations

    if T < 3:
        return 0.0  # Not enough data

    # Normalize prices and quantities
    prices_a = session.prices[:, group_a]
    prices_b = session.prices[:, group_b]
    quantities_a = session.quantities[:, group_a]
    quantities_b = session.quantities[:, group_b]

    # Compute price indices for each group (expenditure weighted)
    exp_a = np.sum(prices_a * quantities_a, axis=1)
    exp_b = np.sum(prices_b * quantities_b, axis=1)

    # Compute average price per group
    avg_price_a = np.mean(prices_a, axis=1)
    avg_price_b = np.mean(prices_b, axis=1)

    # Compute total quantity per group
    total_qty_a = np.sum(quantities_a, axis=1)
    total_qty_b = np.sum(quantities_b, axis=1)

    # Cross-correlation: how much does price_B correlate with quantity_A?
    # If separable, this should be low (after controlling for price_A)
    cross_corr_ab = _partial_correlation(avg_price_b, total_qty_a, avg_price_a)
    cross_corr_ba = _partial_correlation(avg_price_a, total_qty_b, avg_price_b)

    # Average absolute cross-correlation
    cross_effect = (abs(cross_corr_ab) + abs(cross_corr_ba)) / 2

    return min(cross_effect, 1.0)


def _partial_correlation(x: NDArray, y: NDArray, control: NDArray) -> float:
    """Compute partial correlation between x and y, controlling for control."""
    if len(x) < 3:
        return 0.0

    # Residualize x and y on control
    def residualize(arr: NDArray, ctrl: NDArray) -> NDArray:
        if np.std(ctrl) < 1e-10:
            return arr - np.mean(arr)
        coef = np.cov(arr, ctrl)[0, 1] / np.var(ctrl)
        return arr - coef * ctrl

    x_resid = residualize(x, control)
    y_resid = residualize(y, control)

    # Correlation of residuals
    if np.std(x_resid) < 1e-10 or np.std(y_resid) < 1e-10:
        return 0.0

    corr = np.corrcoef(x_resid, y_resid)[0, 1]
    return corr if not np.isnan(corr) else 0.0


def find_separable_partition(
    session: ConsumerSession,
    max_groups: int = 3,
) -> list[list[int]]:
    """
    Automatically discover separable groups of goods.

    Uses hierarchical clustering on the preference graph to find
    groups that can be treated independently.

    Args:
        session: ConsumerSession with prices and quantities
        max_groups: Maximum number of groups to find

    Returns:
        List of lists, where each inner list contains good indices in that group
    """
    N = session.num_goods

    if N < 2:
        return [list(range(N))]

    # Compute pairwise "togetherness" score based on consumption patterns
    togetherness = np.zeros((N, N))

    for t in range(session.num_observations):
        q = session.quantities[t]
        total = np.sum(q)
        if total > 0:
            shares = q / total
            # Goods consumed together in similar proportions have high togetherness
            togetherness += np.outer(shares, shares)

    # Normalize
    togetherness /= session.num_observations

    # Convert to distance matrix
    distance = 1 - togetherness / (togetherness.max() + 1e-10)
    np.fill_diagonal(distance, 0)

    # Simple agglomerative clustering
    groups = [[i] for i in range(N)]

    while len(groups) > max_groups:
        # Find closest pair of groups
        min_dist = float('inf')
        merge_i, merge_j = 0, 1

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                # Average linkage
                avg_dist = np.mean([
                    distance[gi, gj]
                    for gi in groups[i]
                    for gj in groups[j]
                ])
                if avg_dist < min_dist:
                    min_dist = avg_dist
                    merge_i, merge_j = i, j

        # Merge groups
        groups[merge_i].extend(groups[merge_j])
        del groups[merge_j]

    return groups


def compute_cannibalization(
    session: ConsumerSession,
    group_a: list[int],
    group_b: list[int],
) -> dict[str, float]:
    """
    Compute cannibalization metrics between two product groups.

    Useful for superapp analysis (e.g., Uber Rides vs Eats).

    Args:
        session: ConsumerSession with prices and quantities
        group_a: Indices of first product group
        group_b: Indices of second product group

    Returns:
        Dictionary with cannibalization metrics:
        - 'a_to_b': How much A cannibalizes B (0-1)
        - 'b_to_a': How much B cannibalizes A (0-1)
        - 'symmetric': Average cannibalization
        - 'net_direction': Positive if A cannibalizes B more
    """
    T = session.num_observations

    if T < 2:
        return {
            'a_to_b': 0.0,
            'b_to_a': 0.0,
            'symmetric': 0.0,
            'net_direction': 0.0,
        }

    # Compute expenditure shares
    exp_a = np.sum(session.prices[:, group_a] * session.quantities[:, group_a], axis=1)
    exp_b = np.sum(session.prices[:, group_b] * session.quantities[:, group_b], axis=1)
    total_exp = exp_a + exp_b

    # Avoid division by zero
    total_exp = np.maximum(total_exp, 1e-10)

    share_a = exp_a / total_exp
    share_b = exp_b / total_exp

    # Cannibalization: when one share increases, does the other decrease?
    # Beyond what income effects would predict

    # Simple metric: negative correlation of share changes
    if T < 3:
        corr = 0.0
    else:
        delta_a = np.diff(share_a)
        delta_b = np.diff(share_b)
        if np.std(delta_a) > 1e-10 and np.std(delta_b) > 1e-10:
            corr = np.corrcoef(delta_a, delta_b)[0, 1]
            corr = 0.0 if np.isnan(corr) else corr
        else:
            corr = 0.0

    # Negative correlation indicates cannibalization
    symmetric = max(0, -corr)

    # Direction: which group's growth is associated with the other's decline?
    # Compute asymmetric impacts
    a_growth = np.mean(np.diff(exp_a))
    b_growth = np.mean(np.diff(exp_b))

    if a_growth > 0 and b_growth < 0:
        a_to_b = min(1.0, -b_growth / (a_growth + 1e-10))
        b_to_a = 0.0
    elif b_growth > 0 and a_growth < 0:
        a_to_b = 0.0
        b_to_a = min(1.0, -a_growth / (b_growth + 1e-10))
    else:
        a_to_b = symmetric / 2
        b_to_a = symmetric / 2

    return {
        'a_to_b': a_to_b,
        'b_to_a': b_to_a,
        'symmetric': symmetric,
        'net_direction': a_to_b - b_to_a,
    }


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

# test_feature_independence: Tech-friendly name for check_separability
test_feature_independence = check_separability
"""
Test if two feature groups are independent (can be optimized separately).

This is the tech-friendly alias for check_separability.

Use this to determine if product categories can be priced/optimized
independently without considering cross-effects.

Example:
    >>> from pyrevealed import BehaviorLog, test_feature_independence
    >>> # Test if Rides and Eats are independent for a superapp user
    >>> result = test_feature_independence(user_log, group_a=[0, 1], group_b=[2, 3])
    >>> if result.is_separable:
    ...     print("Can price independently")

Returns:
    FeatureIndependenceResult with is_separable and cross_effect_strength
"""

# discover_independent_groups: Tech-friendly name for find_separable_partition
discover_independent_groups = find_separable_partition
"""
Auto-discover groups of features that can be treated independently.

This is the tech-friendly alias for find_separable_partition.

Uses clustering to find natural groupings of features where
cross-effects are minimal.
"""

# compute_cross_impact: Tech-friendly name for compute_cannibalization
compute_cross_impact = compute_cannibalization
"""
Compute how much one feature group impacts another.

This is the tech-friendly alias for compute_cannibalization.

Measures cross-elasticity effects between feature groups.
High cross-impact means changes in one group significantly affect the other.
"""
