"""Acyclical P test (strict preference acyclicity).

Tests whether the strict revealed preference relation is acyclic.
This is MORE LENIENT than GARP - it ignores weak preference violations.

Based on Dziewulski (2023).
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import AcyclicalPResult
from pyrevealed.core.types import Cycle
from pyrevealed.graph.transitive_closure import floyd_warshall_transitive_closure


def check_acyclical_p(
    session: ConsumerSession,
    tolerance: float = 1e-10,
) -> AcyclicalPResult:
    """
    Check if the strict revealed preference relation is acyclic.

    The Acyclical P test is MORE LENIENT than GARP. It only considers
    strict preference relations:
    - P[t,s] = True iff p^t @ x^s < p^t @ x^t
      (bundle s was strictly cheaper when t was chosen)

    A consumer passes this test if there are no cycles in P, even if
    cycles exist in the weak preference relation R. This captures
    "approximately rational" behavior where apparent violations may
    be due to indifference between similar options.

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Numerical tolerance for floating-point comparisons

    Returns:
        AcyclicalPResult with consistency status and violations

    Example:
        >>> from pyrevealed import ConsumerSession, check_acyclical_p
        >>> result = check_acyclical_p(session)
        >>> if result.is_approximately_rational:
        ...     print("Behavior is approximately rational")
        >>> if result.strict_violations_only:
        ...     print("GARP fails but only due to weak preferences")
    """
    start_time = time.perf_counter()

    E = session.expenditure_matrix  # T x T
    T = session.num_observations
    own_exp = session.own_expenditures  # Shape: (T,)

    # =========================================================================
    # Build strict preference matrix P
    # P[t,s] = True iff p^t @ x^s < p^t @ x^t
    # Interpretation: bundle s was strictly cheaper than what was spent at t
    # =========================================================================

    P = own_exp[:, np.newaxis] > E + tolerance

    # Remove self-loops (can't strictly prefer to yourself)
    np.fill_diagonal(P, False)

    num_strict_preferences = int(np.sum(P))

    # =========================================================================
    # Compute transitive closure of P
    # =========================================================================

    P_star = floyd_warshall_transitive_closure(P)

    # =========================================================================
    # Check for cycles: P is acyclic iff no node can reach itself
    # After transitive closure, check if P*[i,j] AND P*[j,i] for any i,j
    # (this indicates a cycle containing both i and j)
    # =========================================================================

    # Check for strict preference cycles
    cycle_matrix = P_star & P_star.T
    np.fill_diagonal(cycle_matrix, False)

    has_cycle = np.any(cycle_matrix)
    is_consistent = not has_cycle

    # Find violation cycles
    violations: list[Cycle] = []
    if not is_consistent:
        violations = _find_p_cycles(P, P_star, cycle_matrix)

    # =========================================================================
    # Compare with GARP for reference
    # =========================================================================

    R = own_exp[:, np.newaxis] >= E - tolerance
    R_star = floyd_warshall_transitive_closure(R)
    garp_violation_matrix = R_star & P.T
    garp_consistent = not np.any(garp_violation_matrix)

    computation_time = (time.perf_counter() - start_time) * 1000

    return AcyclicalPResult(
        is_consistent=is_consistent,
        violations=violations,
        strict_preference_matrix=P,
        transitive_closure=P_star,
        num_strict_preferences=num_strict_preferences,
        garp_consistent=garp_consistent,
        computation_time_ms=computation_time,
    )


def _find_p_cycles(
    P: NDArray[np.bool_],
    P_star: NDArray[np.bool_],
    cycle_matrix: NDArray[np.bool_],
) -> list[Cycle]:
    """
    Find cycles in the strict preference relation P.

    Args:
        P: Strict preference matrix
        P_star: Transitive closure of P
        cycle_matrix: P_star & P_star.T matrix

    Returns:
        List of cycles as tuples of observation indices
    """
    violations: list[Cycle] = []
    seen_cycles: set[frozenset[int]] = set()

    # Find pairs that are in a cycle
    violation_pairs = np.argwhere(cycle_matrix)

    for pair in violation_pairs:
        i, j = int(pair[0]), int(pair[1])
        if i >= j:  # Only process each pair once
            continue

        # Reconstruct cycle
        path_i_to_j = _reconstruct_path_bfs(P, i, j)
        path_j_to_i = _reconstruct_path_bfs(P, j, i)

        if path_i_to_j is not None and path_j_to_i is not None:
            cycle = tuple(path_i_to_j[:-1] + path_j_to_i)

            cycle_set = frozenset(cycle[:-1])
            if cycle_set not in seen_cycles:
                seen_cycles.add(cycle_set)
                violations.append(cycle)

    return violations


def _reconstruct_path_bfs(
    P: NDArray[np.bool_],
    start: int,
    end: int,
) -> list[int] | None:
    """Reconstruct shortest path from start to end using BFS on P."""
    T = P.shape[0]
    queue: deque[tuple[int, list[int]]] = deque([(start, [start])])
    visited = {start}

    while queue:
        current, path = queue.popleft()

        if current == end and len(path) > 1:
            return path

        for next_node in range(T):
            if P[current, next_node] and next_node not in visited:
                visited.add(next_node)
                queue.append((next_node, path + [next_node]))

    return None


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

validate_strict_consistency = check_acyclical_p
"""
Validate strict behavioral consistency only.

This is the tech-friendly alias for check_acyclical_p. More lenient than
full consistency validation - passes if only weak violations exist.
"""
