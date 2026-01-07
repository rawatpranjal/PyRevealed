"""Differentiable rationality (smooth preferences) test.

Tests whether consumer behavior is consistent with smooth, differentiable
utility functions. This is stronger than GARP, requiring:
1. SARP - no indifferent preference cycles
2. Price-quantity uniqueness - different prices imply different quantities

Based on Chiappori & Rochet (1987).
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import DifferentiableResult
from pyrevealed.core.types import Cycle
from pyrevealed.graph.transitive_closure import floyd_warshall_transitive_closure


def check_differentiable(
    session: ConsumerSession,
    tolerance: float = 1e-10,
) -> DifferentiableResult:
    """
    Check if consumer data satisfies differentiable rationality.

    Differentiable rationality requires:
    1. SARP (Strict Axiom of Revealed Preference): No indifferent cycles.
       Unlike GARP which allows weak preferences, SARP requires that if
       x^t is revealed preferred to x^s, then x^s cannot be revealed
       preferred to x^t.

    2. Price-Quantity Uniqueness: If p^t != p^s then x^t != x^s.
       This ensures the demand function is well-defined and differentiable.

    Together, these conditions ensure utility is smooth/differentiable,
    enabling meaningful comparative statics analysis.

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Numerical tolerance for floating-point comparisons

    Returns:
        DifferentiableResult with differentiability status and violations

    Example:
        >>> from pyrevealed import ConsumerSession, check_differentiable
        >>> result = check_differentiable(session)
        >>> if result.is_differentiable:
        ...     print("Preferences are smooth")
        >>> else:
        ...     print(f"Found {result.num_sarp_violations} SARP violations")
    """
    start_time = time.perf_counter()

    E = session.expenditure_matrix  # T x T
    T = session.num_observations
    own_exp = session.own_expenditures  # Shape: (T,)

    # Direct revealed preference: R[i,j] = True iff p_i @ x_i >= p_i @ x_j
    R = own_exp[:, np.newaxis] >= E - tolerance

    # Transitive closure of R
    R_star = floyd_warshall_transitive_closure(R)

    # =========================================================================
    # Part 1: Check SARP (Strict Axiom of Revealed Preference)
    # SARP violated if R*[i,j] AND R*[j,i] for i != j
    # (i.e., both transitively revealed preferred to each other)
    # =========================================================================

    # SARP violation matrix: both directions of transitive preference
    sarp_violation_matrix = R_star & R_star.T

    # Remove diagonal (self-preference is fine)
    np.fill_diagonal(sarp_violation_matrix, False)

    satisfies_sarp = not np.any(sarp_violation_matrix)

    # Find SARP violation cycles (indifferent cycles)
    sarp_violations: list[Cycle] = []
    if not satisfies_sarp:
        sarp_violations = _find_sarp_violations(R, R_star, sarp_violation_matrix)

    # =========================================================================
    # Part 2: Check Price-Quantity Uniqueness
    # Violated if p^t != p^s but x^t = x^s for any t, s
    # =========================================================================

    uniqueness_violations: list[tuple[int, int]] = []

    for t in range(T):
        for s in range(t + 1, T):  # Only check upper triangle
            prices_equal = np.allclose(
                session.prices[t], session.prices[s], rtol=tolerance, atol=tolerance
            )
            quantities_equal = np.allclose(
                session.quantities[t],
                session.quantities[s],
                rtol=tolerance,
                atol=tolerance,
            )

            if not prices_equal and quantities_equal:
                uniqueness_violations.append((t, s))

    satisfies_uniqueness = len(uniqueness_violations) == 0

    # =========================================================================
    # Combine results
    # =========================================================================

    is_differentiable = satisfies_sarp and satisfies_uniqueness

    computation_time = (time.perf_counter() - start_time) * 1000

    return DifferentiableResult(
        is_differentiable=is_differentiable,
        satisfies_sarp=satisfies_sarp,
        satisfies_uniqueness=satisfies_uniqueness,
        sarp_violations=sarp_violations,
        uniqueness_violations=uniqueness_violations,
        direct_revealed_preference=R,
        transitive_closure=R_star,
        computation_time_ms=computation_time,
    )


def _find_sarp_violations(
    R: NDArray[np.bool_],
    R_star: NDArray[np.bool_],
    violation_matrix: NDArray[np.bool_],
) -> list[Cycle]:
    """
    Find cycles representing SARP violations (indifferent preference cycles).

    Args:
        R: Direct revealed preference matrix
        R_star: Transitive closure of R
        violation_matrix: R_star & R_star.T matrix

    Returns:
        List of cycles as tuples of observation indices
    """
    violations: list[Cycle] = []
    seen_cycles: set[frozenset[int]] = set()

    # Find pairs (i, j) with mutual transitive preference
    violation_pairs = np.argwhere(violation_matrix)

    for pair in violation_pairs:
        i, j = int(pair[0]), int(pair[1])
        if i >= j:  # Only process each pair once
            continue

        # Reconstruct a cycle i -> ... -> j -> ... -> i
        path_i_to_j = _reconstruct_path_bfs(R, i, j)
        path_j_to_i = _reconstruct_path_bfs(R, j, i)

        if path_i_to_j is not None and path_j_to_i is not None:
            # Combine paths into cycle
            cycle = tuple(path_i_to_j[:-1] + path_j_to_i)

            cycle_set = frozenset(cycle[:-1])
            if cycle_set not in seen_cycles:
                seen_cycles.add(cycle_set)
                violations.append(cycle)

    return violations


def _reconstruct_path_bfs(
    R: NDArray[np.bool_],
    start: int,
    end: int,
) -> list[int] | None:
    """Reconstruct shortest path from start to end using BFS on R."""
    T = R.shape[0]
    queue: deque[tuple[int, list[int]]] = deque([(start, [start])])
    visited = {start}

    while queue:
        current, path = queue.popleft()

        if current == end and len(path) > 1:
            return path

        for next_node in range(T):
            if R[current, next_node] and next_node not in visited:
                visited.add(next_node)
                queue.append((next_node, path + [next_node]))

    return None


def check_sarp(
    session: ConsumerSession,
    tolerance: float = 1e-10,
) -> tuple[bool, list[Cycle]]:
    """
    Check if consumer data satisfies SARP (Strict Axiom of Revealed Preference).

    SARP is violated if there exist observations t, s with mutual revealed
    preference (both x^t R* x^s and x^s R* x^t).

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Numerical tolerance for comparisons

    Returns:
        Tuple of (is_consistent, list of violation cycles)
    """
    result = check_differentiable(session, tolerance)
    return result.satisfies_sarp, result.sarp_violations


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

validate_smooth_preferences = check_differentiable
"""
Validate that user preferences are smooth (differentiable).

This is the tech-friendly alias for check_differentiable. Smooth preferences
enable demand function derivatives for price sensitivity analysis.
"""

validate_sarp = check_sarp
"""Validate SARP (no indifferent preference cycles)."""
