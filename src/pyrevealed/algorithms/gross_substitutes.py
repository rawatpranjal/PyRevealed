"""Gross substitutes test for cross-price relationships."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import GrossSubstitutesResult, SubstitutionMatrixResult


def check_gross_substitutes(
    session: ConsumerSession,
    good_g: int,
    good_h: int,
    price_change_threshold: float = 0.05,
    tolerance: float = 1e-10,
) -> GrossSubstitutesResult:
    """
    Test if two goods are gross substitutes based on revealed preference data.

    Gross substitutes: when the price of good g increases (other prices constant)
    and quantity of g decreases, we should see quantity of h increase.

    Gross complements: when p_g increases and x_g decreases, x_h also decreases.

    The algorithm:
    1. Finds observation pairs where price of g changed significantly
    2. Checks if other prices stayed relatively constant
    3. Analyzes the direction of quantity changes
    4. Classifies the relationship based on majority of informative pairs

    Args:
        session: ConsumerSession with prices and quantities
        good_g: Index of first good
        good_h: Index of second good (potential substitute)
        price_change_threshold: Minimum relative price change to consider (default 5%)
        tolerance: Numerical tolerance

    Returns:
        GrossSubstitutesResult with relationship classification and confidence

    Example:
        >>> import numpy as np
        >>> from pyrevealed import ConsumerSession, check_gross_substitutes
        >>> # Prices for goods 0 and 1 over 3 observations
        >>> prices = np.array([[1.0, 2.0], [2.0, 2.0], [1.0, 1.0]])
        >>> quantities = np.array([[4.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
        >>> session = ConsumerSession(prices=prices, quantities=quantities)
        >>> result = check_gross_substitutes(session, good_g=0, good_h=1)
        >>> print(f"Relationship: {result.relationship}")

    References:
        Hicks, J. R. (1939). Value and Capital. Oxford University Press.
    """
    start_time = time.perf_counter()

    T = session.num_observations
    N = session.num_goods

    if good_g < 0 or good_g >= N or good_h < 0 or good_h >= N:
        raise ValueError(f"Good indices must be in [0, {N})")
    if good_g == good_h:
        raise ValueError("good_g and good_h must be different")

    P = session.prices       # T x N
    Q = session.quantities   # T x N

    substitutes_pairs: list[tuple[int, int]] = []
    complements_pairs: list[tuple[int, int]] = []

    # Compare all pairs of observations
    for i in range(T):
        for j in range(i + 1, T):
            # Check if price of g changed significantly while others ~constant
            pg_i, pg_j = P[i, good_g], P[j, good_g]
            ph_i, ph_j = P[i, good_h], P[j, good_h]

            # Skip if prices are near zero
            if pg_i < tolerance or pg_j < tolerance:
                continue

            # Relative price change for good g
            rel_change_g = abs(pg_j - pg_i) / pg_i

            # Check if price of h stayed relatively constant
            rel_change_h = abs(ph_j - ph_i) / max(ph_i, tolerance)

            # We want: significant change in p_g, small change in p_h
            if rel_change_g < price_change_threshold:
                continue  # Not enough price movement in g
            if rel_change_h > rel_change_g * 0.5:
                continue  # Too much change in h relative to g

            # Check other prices didn't change too much
            other_goods = [k for k in range(N) if k != good_g and k != good_h]
            if other_goods:
                other_changes = [
                    abs(P[j, k] - P[i, k]) / max(P[i, k], tolerance)
                    for k in other_goods
                ]
                if max(other_changes) > rel_change_g * 0.5:
                    continue  # Other prices changed too much

            # Get quantity changes
            xg_i, xg_j = Q[i, good_g], Q[j, good_g]
            xh_i, xh_j = Q[i, good_h], Q[j, good_h]

            # Direction of price change for g
            pg_increased = pg_j > pg_i + tolerance
            pg_decreased = pg_j < pg_i - tolerance

            # Direction of quantity changes
            xg_increased = xg_j > xg_i + tolerance
            xg_decreased = xg_j < xg_i - tolerance
            xh_increased = xh_j > xh_i + tolerance
            xh_decreased = xh_j < xh_i - tolerance

            # Gross substitutes pattern:
            # p_g up, x_g down => x_h up (or p_g down, x_g up => x_h down)
            if pg_increased and xg_decreased:
                if xh_increased:
                    substitutes_pairs.append((i, j))
                elif xh_decreased:
                    complements_pairs.append((i, j))
            elif pg_decreased and xg_increased:
                if xh_decreased:
                    substitutes_pairs.append((i, j))
                elif xh_increased:
                    complements_pairs.append((i, j))

    # Determine relationship
    n_subs = len(substitutes_pairs)
    n_comp = len(complements_pairs)
    informative_pairs = n_subs + n_comp

    if informative_pairs == 0:
        relationship = "inconclusive"
        are_substitutes = False
        are_complements = False
        confidence = 0.0
        supporting = []
        violating = []
    elif n_subs > n_comp:
        relationship = "substitutes"
        are_substitutes = True
        are_complements = False
        confidence = n_subs / informative_pairs
        supporting = substitutes_pairs
        violating = complements_pairs
    elif n_comp > n_subs:
        relationship = "complements"
        are_substitutes = False
        are_complements = True
        confidence = n_comp / informative_pairs
        supporting = complements_pairs
        violating = substitutes_pairs
    else:
        relationship = "independent"
        are_substitutes = False
        are_complements = False
        confidence = 0.5
        supporting = []
        violating = []

    computation_time = (time.perf_counter() - start_time) * 1000

    return GrossSubstitutesResult(
        are_substitutes=are_substitutes,
        are_complements=are_complements,
        relationship=relationship,
        supporting_pairs=supporting,
        violating_pairs=violating,
        confidence_score=confidence,
        good_g_index=good_g,
        good_h_index=good_h,
        computation_time_ms=computation_time,
    )


def compute_substitution_matrix(
    session: ConsumerSession,
    price_change_threshold: float = 0.05,
) -> SubstitutionMatrixResult:
    """
    Compute pairwise substitution relationships for all goods.

    Returns an N x N matrix where entry [g, h] indicates the relationship
    between goods g and h.

    Args:
        session: ConsumerSession
        price_change_threshold: Minimum price change to consider

    Returns:
        SubstitutionMatrixResult with relationship matrix

    Example:
        >>> from pyrevealed import ConsumerSession, compute_substitution_matrix
        >>> result = compute_substitution_matrix(session)
        >>> print(f"Substitute pairs: {result.substitute_pairs}")
        >>> print(f"Complement pairs: {result.complement_pairs}")
    """
    start_time = time.perf_counter()

    N = session.num_goods
    relationship_matrix = np.empty((N, N), dtype=object)
    confidence_matrix = np.zeros((N, N))

    for g in range(N):
        for h in range(N):
            if g == h:
                relationship_matrix[g, h] = "self"
                confidence_matrix[g, h] = 1.0
            elif g < h:
                result = check_gross_substitutes(session, g, h, price_change_threshold)
                relationship_matrix[g, h] = result.relationship
                relationship_matrix[h, g] = result.relationship
                confidence_matrix[g, h] = result.confidence_score
                confidence_matrix[h, g] = result.confidence_score

    computation_time = (time.perf_counter() - start_time) * 1000

    return SubstitutionMatrixResult(
        relationship_matrix=relationship_matrix,
        confidence_matrix=confidence_matrix,
        num_goods=N,
        computation_time_ms=computation_time,
    )


def check_law_of_demand(
    session: ConsumerSession,
    good: int,
    price_change_threshold: float = 0.05,
    tolerance: float = 1e-10,
) -> dict:
    """
    Check if a good satisfies the law of demand (own-price effect is negative).

    The law of demand states that when price increases, quantity demanded
    decreases (holding other factors constant).

    Args:
        session: ConsumerSession
        good: Index of the good to test
        price_change_threshold: Minimum price change to consider
        tolerance: Numerical tolerance

    Returns:
        Dictionary with:
        - satisfies_law: True if law of demand holds
        - supporting_pairs: Pairs where law holds
        - violating_pairs: Pairs where law is violated (Giffen good behavior)
        - confidence: Fraction of pairs supporting the law
    """
    T = session.num_observations
    N = session.num_goods
    P = session.prices
    Q = session.quantities

    supporting_pairs: list[tuple[int, int]] = []
    violating_pairs: list[tuple[int, int]] = []

    for i in range(T):
        for j in range(i + 1, T):
            pg_i, pg_j = P[i, good], P[j, good]

            if pg_i < tolerance or pg_j < tolerance:
                continue

            rel_change = abs(pg_j - pg_i) / pg_i
            if rel_change < price_change_threshold:
                continue

            # Check other prices
            other_goods = [k for k in range(N) if k != good]
            if other_goods:
                other_changes = [
                    abs(P[j, k] - P[i, k]) / max(P[i, k], tolerance)
                    for k in other_goods
                ]
                if max(other_changes) > rel_change * 0.5:
                    continue

            xg_i, xg_j = Q[i, good], Q[j, good]

            # Law of demand: price up => quantity down
            price_up = pg_j > pg_i + tolerance
            price_down = pg_j < pg_i - tolerance
            qty_up = xg_j > xg_i + tolerance
            qty_down = xg_j < xg_i - tolerance

            if (price_up and qty_down) or (price_down and qty_up):
                supporting_pairs.append((i, j))
            elif (price_up and qty_up) or (price_down and qty_down):
                violating_pairs.append((i, j))

    total = len(supporting_pairs) + len(violating_pairs)
    confidence = len(supporting_pairs) / total if total > 0 else 0.5

    return {
        "satisfies_law": len(violating_pairs) == 0 and len(supporting_pairs) > 0,
        "supporting_pairs": supporting_pairs,
        "violating_pairs": violating_pairs,
        "confidence": confidence,
        "num_informative_pairs": total,
    }


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

# test_cross_price_effect: Tech-friendly name for check_gross_substitutes
test_cross_price_effect = check_gross_substitutes
"""
Test how changes in one item's price affect demand for another item.

This is the tech-friendly alias for check_gross_substitutes.

Use this to understand cross-price relationships between products:
- Substitutes: Price of A up → Demand for B up (users switch)
- Complements: Price of A up → Demand for B down (bought together)
- Independent: No clear relationship

Example:
    >>> from pyrevealed import BehaviorLog, test_cross_price_effect
    >>> result = test_cross_price_effect(user_log, good_g=0, good_h=1)
    >>> if result.are_substitutes:
    ...     print("Users treat these as substitutes")
"""

compute_cross_price_matrix = compute_substitution_matrix
"""
Compute all pairwise cross-price relationships.

Returns an N x N matrix of relationships between all goods.
"""
