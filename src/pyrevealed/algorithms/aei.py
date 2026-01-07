"""Afriat Efficiency Index (AEI) computation via binary search."""

from __future__ import annotations

import time

import numpy as np

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import AEIResult, GARPResult
from pyrevealed.core.types import Cycle
from pyrevealed.graph.transitive_closure import floyd_warshall_transitive_closure


def compute_aei(
    session: ConsumerSession,
    tolerance: float = 1e-6,
    max_iterations: int = 50,
) -> AEIResult:
    """
    Compute Afriat Efficiency Index using binary search.

    The AEI measures how close consumer behavior is to perfect rationality.
    It is defined as:

        AEI = sup{e in [0,1] : data satisfies GARP with efficiency e}

    where GARP with efficiency e means we deflate budgets by factor e:
        R_e[i,j] = True iff e * (p_i @ x_i) >= p_i @ x_j

    Interpretation:
    - AEI = 1.0: Perfectly consistent (satisfies GARP)
    - AEI = 0.5: Consumer wastes ~50% of budget on inconsistent choices
    - AEI = 0.0: Completely irrational behavior

    The algorithm uses binary search to find the supremum efficiently:
    1. If GARP holds at e=1, return AEI=1.0
    2. Otherwise, binary search between [0, 1] to find largest e where GARP holds

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Convergence tolerance for binary search (default: 1e-6)
        max_iterations: Maximum iterations for binary search (default: 50)

    Returns:
        AEIResult with efficiency index and supporting data

    Example:
        >>> import numpy as np
        >>> from pyrevealed import ConsumerSession, compute_aei
        >>> prices = np.array([[1.0, 2.0], [2.0, 1.0]])
        >>> quantities = np.array([[4.0, 1.0], [1.0, 4.0]])
        >>> session = ConsumerSession(prices=prices, quantities=quantities)
        >>> result = compute_aei(session)
        >>> print(f"AEI: {result.efficiency_index:.4f}")
    """
    start_time = time.perf_counter()

    # First check if data satisfies GARP at e=1 (perfect consistency)
    from pyrevealed.algorithms.garp import check_garp

    garp_result = check_garp(session)

    if garp_result.is_consistent:
        computation_time = (time.perf_counter() - start_time) * 1000
        return AEIResult(
            efficiency_index=1.0,
            is_perfectly_consistent=True,
            garp_result_at_threshold=garp_result,
            binary_search_iterations=0,
            tolerance=tolerance,
            computation_time_ms=computation_time,
        )

    # Binary search for AEI
    e_low = 0.0
    e_high = 1.0
    iterations = 0
    last_consistent_e = 0.0
    last_consistent_result: GARPResult | None = None

    while (e_high - e_low > tolerance) and (iterations < max_iterations):
        e_mid = (e_low + e_high) / 2

        # Check GARP at efficiency level e_mid
        is_consistent, garp_at_e = _check_garp_at_efficiency(
            session, e_mid, tolerance=1e-10
        )

        if is_consistent:
            e_low = e_mid
            last_consistent_e = e_mid
            last_consistent_result = garp_at_e
        else:
            e_high = e_mid

        iterations += 1

    # Final efficiency index
    aei = last_consistent_e

    # Get final GARP result at the threshold
    if last_consistent_result is None:
        # Edge case: even e=0 doesn't satisfy GARP (shouldn't happen normally)
        _, last_consistent_result = _check_garp_at_efficiency(
            session, 0.0, tolerance=1e-10
        )

    computation_time = (time.perf_counter() - start_time) * 1000

    return AEIResult(
        efficiency_index=aei,
        is_perfectly_consistent=False,
        garp_result_at_threshold=last_consistent_result,
        binary_search_iterations=iterations,
        tolerance=tolerance,
        computation_time_ms=computation_time,
    )


def _check_garp_at_efficiency(
    session: ConsumerSession,
    efficiency: float,
    tolerance: float = 1e-10,
) -> tuple[bool, GARPResult]:
    """
    Check GARP at a given efficiency level e.

    Modified revealed preference relation:
        R_e[i,j] = True iff e * (p_i @ x_i) >= p_i @ x_j

    Args:
        session: ConsumerSession
        efficiency: Efficiency parameter e in [0, 1]
        tolerance: Numerical tolerance

    Returns:
        Tuple of (is_consistent, GARPResult)
    """
    E = session.expenditure_matrix
    own_exp = session.own_expenditures

    # Modified revealed preference with efficiency deflation
    # R_e[i,j] = (e * p_i @ x_i >= p_i @ x_j)
    R_e = (efficiency * own_exp[:, np.newaxis]) >= E - tolerance

    # P_e[i,j] = (e * p_i @ x_i > p_i @ x_j)
    P_e = (efficiency * own_exp[:, np.newaxis]) > E + tolerance
    np.fill_diagonal(P_e, False)

    # Transitive closure
    R_e_star = floyd_warshall_transitive_closure(R_e)

    # GARP violation check
    violation_matrix = R_e_star & P_e.T
    is_consistent = not np.any(violation_matrix)

    # Find violations if any (simplified for efficiency)
    violations: list[Cycle] = []
    if not is_consistent:
        # Just find the first violation pair for the result
        violation_pairs = np.argwhere(violation_matrix)
        if len(violation_pairs) > 0:
            i, j = int(violation_pairs[0, 0]), int(violation_pairs[0, 1])
            violations = [(i, j, i)]  # Simplified cycle representation

    result = GARPResult(
        is_consistent=is_consistent,
        violations=violations,
        direct_revealed_preference=R_e,
        transitive_closure=R_e_star,
        strict_revealed_preference=P_e,
        computation_time_ms=0.0,  # Not tracked for internal calls
    )

    return is_consistent, result


def compute_varian_index(
    session: ConsumerSession,
    tolerance: float = 1e-6,
    max_iterations: int = 50,
) -> float:
    """
    Compute Varian's index of efficiency (alternative to Afriat's).

    Varian's index finds the smallest uniform efficiency e such that
    all observations can be rationalized. It is equivalent to AEI for
    most practical purposes.

    Args:
        session: ConsumerSession
        tolerance: Convergence tolerance
        max_iterations: Maximum binary search iterations

    Returns:
        Efficiency index in [0, 1]

    Note:
        This is functionally equivalent to compute_aei but included for
        completeness as referenced in the literature.
    """
    return compute_aei(session, tolerance, max_iterations).efficiency_index


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

# compute_integrity_score: Tech-friendly name for compute_aei
compute_integrity_score = compute_aei
"""
Compute the behavioral integrity score (0-1).

This is the tech-friendly alias for compute_aei (Afriat Efficiency Index).

The integrity score measures how "clean" the behavioral signal is:
- 1.0 = Perfect integrity, fully consistent user behavior
- 0.9+ = High integrity, minor noise
- 0.7-0.9 = Moderate integrity, some confusion or noise
- <0.7 = Low integrity, likely bot or multiple users

Use this for:
- Bot detection (bots have low integrity scores)
- Account sharing detection (multiple users = inconsistent behavior)
- Data quality assessment before ML training

Example:
    >>> from pyrevealed import BehaviorLog, compute_integrity_score
    >>> result = compute_integrity_score(user_log)
    >>> if result.integrity_score < 0.85:
    ...     flag_for_review(user_id)

Returns:
    IntegrityResult with integrity_score in [0, 1]
"""
