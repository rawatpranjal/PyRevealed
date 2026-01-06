"""GARP (Generalized Axiom of Revealed Preference) detection algorithm."""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import GARPResult
from pyrevealed.core.types import Cycle
from pyrevealed.graph.transitive_closure import floyd_warshall_transitive_closure


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
    T = session.num_observations

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

    # Find pairs (i, j) where R*[i,j] and P[j,i]
    violation_pairs = np.argwhere(violation_matrix)

    for pair in violation_pairs:
        i, j = int(pair[0]), int(pair[1])

        # Reconstruct a path from i to j using R, then add the strict edge back
        path = _reconstruct_path_bfs(R, i, j)

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

    Args:
        R: Direct revealed preference adjacency matrix
        start: Starting node index
        end: Ending node index

    Returns:
        List of node indices forming the path (ending with start to complete cycle),
        or None if no path exists
    """
    T = R.shape[0]
    queue: deque[tuple[int, list[int]]] = deque([(start, [start])])
    visited = {start}

    while queue:
        current, path = queue.popleft()

        if current == end and len(path) > 1:
            # Found path to end, return with start appended to complete cycle
            return path + [start]

        for next_node in range(T):
            if R[current, next_node] and next_node not in visited:
                visited.add(next_node)
                queue.append((next_node, path + [next_node]))

    return None


def check_warp(
    session: ConsumerSession,
    tolerance: float = 1e-10,
) -> tuple[bool, list[tuple[int, int]]]:
    """
    Check if consumer data satisfies WARP (Weak Axiom of Revealed Preference).

    WARP is a weaker condition than GARP. It only checks for direct (length-2)
    violations: if x_i R x_j, then NOT x_j P x_i.

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Numerical tolerance for comparisons

    Returns:
        Tuple of (is_consistent, list of violating pairs)
    """
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

    return len(violations) == 0, violations
