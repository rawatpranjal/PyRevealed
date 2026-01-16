"""GARP (Generalized Axiom of Revealed Preference) detection algorithm."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import GARPResult, WARPResult
from pyrevealed.core.types import Cycle
from pyrevealed.graph.transitive_closure import floyd_warshall_transitive_closure
from pyrevealed._kernels import bfs_find_path_numba, find_violation_pairs_numba


def check_garp(
    session: ConsumerSession,
    tolerance: float = 1e-10,
) -> GARPResult:
    """
    Check if consumer data satisfies GARP using Warshall's algorithm.

    GARP (Generalized Axiom of Revealed Preference) states that revealed
    preferences must be acyclic when considering both weak and strict
    preferences. A violation occurs when there exists a cycle in the
    transitive closure that includes at least one strict preference.

    The algorithm:
    1. Compute direct revealed preference matrix R:
       R[i,j] = True iff p_i @ x_i >= p_i @ x_j
       (bundle j was affordable when i was chosen, so i is weakly preferred)

    2. Compute strict revealed preference matrix P:
       P[i,j] = True iff p_i @ x_i > p_i @ x_j
       (bundle j was strictly cheaper, so i is strictly preferred)

    3. Compute transitive closure R* of R using Floyd-Warshall

    4. Check for violations: GARP is violated if exists i,j such that
       R*[i,j] = True AND P[j,i] = True
       (i is transitively preferred to j, but j is strictly preferred to i)

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Numerical tolerance for floating-point comparisons

    Returns:
        GARPResult with consistency flag, violation cycles, and matrices

    Example:
        >>> import numpy as np
        >>> from pyrevealed import ConsumerSession, check_garp
        >>> # Consistent data: when A is cheap, buy more A; when B is cheap, buy more B
        >>> prices = np.array([[1.0, 2.0], [2.0, 1.0]])
        >>> quantities = np.array([[4.0, 1.0], [1.0, 4.0]])
        >>> session = ConsumerSession(prices=prices, quantities=quantities)
        >>> result = check_garp(session)
        >>> result.is_consistent
        True
    """
    start_time = time.perf_counter()

    E = session.expenditure_matrix  # T x T

    # Own expenditures (diagonal): what was actually spent at each observation
    own_exp = session.own_expenditures  # Shape: (T,)

    # Direct revealed preference: R[i,j] = True iff p_i @ x_i >= p_i @ x_j
    # Interpretation: bundle j was affordable when i was chosen
    # Vectorized: compare each row's own expenditure to all cross-expenditures
    R = own_exp[:, np.newaxis] >= E - tolerance

    # Strict revealed preference: P[i,j] = True iff p_i @ x_i > p_i @ x_j
    # Interpretation: bundle j was strictly cheaper than what was spent
    P = own_exp[:, np.newaxis] > E + tolerance

    # Remove self-loops from P (can't strictly prefer to yourself)
    np.fill_diagonal(P, False)

    # Transitive closure of R using Floyd-Warshall
    R_star = floyd_warshall_transitive_closure(R)

    # GARP violation check:
    # Violation exists if R*[i,j] AND P[j,i] for any i,j
    # This means: i is transitively revealed preferred to j (via R*),
    # BUT j is strictly revealed preferred to i (via P)
    violation_matrix = R_star & P.T

    is_consistent = not np.any(violation_matrix)

    # Find all violation cycles if not consistent
    violations: list[Cycle] = []
    if not is_consistent:
        violations = _find_violation_cycles(R, P, R_star, violation_matrix)

    computation_time = (time.perf_counter() - start_time) * 1000

    return GARPResult(
        is_consistent=is_consistent,
        violations=violations,
        direct_revealed_preference=R,
        transitive_closure=R_star,
        strict_revealed_preference=P,
        computation_time_ms=computation_time,
    )


def _find_violation_cycles(
    R: NDArray[np.bool_],
    P: NDArray[np.bool_],
    R_star: NDArray[np.bool_],
    violation_matrix: NDArray[np.bool_],
) -> list[Cycle]:
    """
    Find cycles that violate GARP.

    A violation cycle is a sequence i1 -> i2 -> ... -> in -> i1 where:
    - Each consecutive pair is connected by revealed preference (R)
    - At least one edge is strict preference (P)

    Uses Numba JIT for fast violation pair finding and path reconstruction.

    Args:
        R: Direct revealed preference matrix
        P: Strict revealed preference matrix
        R_star: Transitive closure of R
        violation_matrix: R_star & P.T (pre-computed)

    Returns:
        List of violation cycles as tuples of observation indices
    """
    violations: list[Cycle] = []
    seen_cycles: set[frozenset[int]] = set()

    # Find pairs (i, j) where R*[i,j] and P[j,i] using Numba kernel
    R_star_c = np.ascontiguousarray(R_star, dtype=np.bool_)
    P_c = np.ascontiguousarray(P, dtype=np.bool_)
    violation_pairs = find_violation_pairs_numba(R_star_c, P_c)

    R_c = np.ascontiguousarray(R, dtype=np.bool_)

    for idx in range(violation_pairs.shape[0]):
        i, j = int(violation_pairs[idx, 0]), int(violation_pairs[idx, 1])

        # Reconstruct a path from i to j using R, then add the strict edge back
        path = _reconstruct_path_bfs(R_c, i, j)

        if path is not None:
            # The cycle is: path from i to j, then j -> i (strict preference)
            cycle = tuple(path)

            # Avoid duplicate cycles (same nodes, different starting points)
            cycle_set = frozenset(cycle[:-1])  # Exclude repeated first node
            if cycle_set not in seen_cycles:
                seen_cycles.add(cycle_set)
                violations.append(cycle)

    return violations


def _reconstruct_path_bfs(
    R: NDArray[np.bool_],
    start: int,
    end: int,
) -> list[int] | None:
    """
    Reconstruct shortest path from start to end using BFS on R.

    Uses Numba JIT for fast path finding.

    Args:
        R: Direct revealed preference adjacency matrix (must be contiguous)
        start: Starting node index
        end: Ending node index

    Returns:
        List of node indices forming the path (ending with start to complete cycle),
        or None if no path exists
    """
    path_arr = bfs_find_path_numba(R, np.int64(start), np.int64(end))

    if len(path_arr) == 0 or path_arr[0] == -1:
        return None

    return list(path_arr)


def check_warp(
    session: ConsumerSession,
    tolerance: float = 1e-10,
) -> WARPResult:
    """
    Check if consumer data satisfies WARP (Weak Axiom of Revealed Preference).

    WARP is a weaker condition than GARP. It only checks for direct (length-2)
    violations: if x_i R x_j, then NOT x_j P x_i.

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Numerical tolerance for comparisons

    Returns:
        WARPResult with is_consistent flag and list of violating pairs
    """
    start_time = time.perf_counter()

    E = session.expenditure_matrix
    own_exp = session.own_expenditures

    R = own_exp[:, np.newaxis] >= E - tolerance
    P = own_exp[:, np.newaxis] > E + tolerance
    np.fill_diagonal(P, False)

    # WARP violation: R[i,j] AND P[j,i]
    violation_matrix = R & P.T

    violations = [
        (int(i), int(j))
        for i, j in np.argwhere(violation_matrix)
        if i < j  # Avoid duplicates
    ]

    computation_time = (time.perf_counter() - start_time) * 1000

    return WARPResult(
        is_consistent=len(violations) == 0,
        violations=violations,
        computation_time_ms=computation_time,
    )


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

# validate_consistency: Tech-friendly name for check_garp
validate_consistency = check_garp
"""
Validate that user behavior is internally consistent.

This is the tech-friendly alias for check_garp (GARP = Generalized Axiom
of Revealed Preference). Consistent behavior indicates:
- Single user (not a shared account)
- Not a bot (bots make random inconsistent choices)
- Not confused by the UI

Returns a ConsistencyResult with:
- is_valid: True if behavior is consistent
- inconsistencies: List of detected inconsistencies

Example:
    >>> from pyrevealed import BehaviorLog, validate_consistency
    >>> result = validate_consistency(user_log)
    >>> if not result.is_valid:
    ...     print(f"Found {result.num_violations} inconsistencies")
"""

# validate_consistency_weak: Tech-friendly name for check_warp
validate_consistency_weak = check_warp
"""
Weak consistency check (only checks direct contradictions).

Faster than full validate_consistency but may miss transitive inconsistencies.
"""
