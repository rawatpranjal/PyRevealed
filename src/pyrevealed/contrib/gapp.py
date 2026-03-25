"""GAPP (Generalized Axiom of Price Preference) test.

Tests consistency of revealed price preferences. The dual perspective
to GARP - instead of testing consistency of quantity preferences,
GAPP tests consistency of price preferences.

Based on Deb et al. (2022).
"""

from __future__ import annotations

import time

import numpy as np

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import GAPPResult
from pyrevealed.graph.transitive_closure import floyd_warshall_transitive_closure


def check_gapp(
    session: ConsumerSession,
    tolerance: float = 1e-10,
) -> GAPPResult:
    """
    Check if consumer data satisfies GAPP (price preference consistency).

    GAPP is the dual of GARP. While GARP tests if the consumer has
    consistent preferences over bundles, GAPP tests if the consumer
    has consistent preferences over price vectors.

    Price s is revealed preferred to price t (p^s R_p p^t) if:
    - The bundle bought at t would cost <= at prices s
    - Formally: p^s @ x^t <= p^t @ x^t

    Interpretation: "I prefer shopping when prices are like s rather
    than when prices are like t, because my desired bundle is cheaper."

    GAPP is violated if there exists a cycle in price preferences:
    - s R_p* t (s transitively price-preferred to t)
    - AND t P_p s (t strictly price-preferred to s)

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Numerical tolerance for floating-point comparisons

    Returns:
        GAPPResult with consistency status and violations

    Example:
        >>> from pyrevealed import ConsumerSession, check_gapp
        >>> result = check_gapp(session)
        >>> if result.prefers_lower_prices:
        ...     print("Consumer has consistent price preferences")
    """
    start_time = time.perf_counter()

    E = session.expenditure_matrix  # T x T: E[i,j] = p^i @ x^j
    T = session.num_observations
    own_exp = session.own_expenditures  # Shape: (T,)

    # =========================================================================
    # Build price preference matrix R_p
    # R_p[s,t] = True iff p^s @ x^t <= p^t @ x^t
    # Interpretation: price vector s is weakly preferred to price vector t
    # (the bundle bought at t would cost the same or less under prices s)
    # =========================================================================

    # E[s,t] = p^s @ x^t
    # own_exp[t] = p^t @ x^t
    # R_p[s,t] = True iff E[s,t] <= own_exp[t]

    R_p = E <= own_exp[np.newaxis, :] + tolerance

    # =========================================================================
    # Build strict price preference matrix P_p
    # P_p[s,t] = True iff p^s @ x^t < p^t @ x^t
    # =========================================================================

    P_p = E < own_exp[np.newaxis, :] - tolerance

    # Remove self-preferences from strict relation
    np.fill_diagonal(P_p, False)

    num_price_preferences = int(np.sum(R_p)) - T  # Subtract diagonal

    # =========================================================================
    # Compute transitive closure of R_p
    # =========================================================================

    R_p_star = floyd_warshall_transitive_closure(R_p)

    # =========================================================================
    # Check for GAPP violations
    # Violation: R_p*[s,t] AND P_p[t,s]
    # (s is transitively price-preferred to t, but t is strictly preferred to s)
    # =========================================================================

    violation_matrix = R_p_star & P_p.T

    is_consistent = not np.any(violation_matrix)

    # Find violation pairs
    violations: list[tuple[int, int]] = []
    if not is_consistent:
        violation_pairs = np.argwhere(violation_matrix)
        for pair in violation_pairs:
            s, t = int(pair[0]), int(pair[1])
            violations.append((s, t))

    # =========================================================================
    # Compare with GARP for reference
    # =========================================================================

    R = own_exp[:, np.newaxis] >= E - tolerance
    P = own_exp[:, np.newaxis] > E + tolerance
    np.fill_diagonal(P, False)
    R_star = floyd_warshall_transitive_closure(R)
    garp_violation = R_star & P.T
    garp_consistent = not np.any(garp_violation)

    computation_time = (time.perf_counter() - start_time) * 1000

    return GAPPResult(
        is_consistent=is_consistent,
        violations=violations,
        price_preference_matrix=R_p,
        strict_price_preference=P_p,
        transitive_closure=R_p_star,
        num_price_preferences=num_price_preferences,
        garp_consistent=garp_consistent,
        computation_time_ms=computation_time,
    )


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

validate_price_preferences = check_gapp
"""
Validate that user has consistent price preferences.

This is the tech-friendly alias for check_gapp. Tests if the user
consistently prefers situations where their desired items are cheaper.
"""
