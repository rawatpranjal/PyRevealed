"""HARP (Homothetic Axiom of Revealed Preference) test for homotheticity."""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import HARPResult, GARPResult
from pyrevealed.core.types import Cycle


def check_harp(
    session: ConsumerSession,
    tolerance: float = 1e-10,
) -> HARPResult:
    """
    Check if consumer data satisfies HARP (Homothetic Axiom of Revealed Preference).

    HARP tests whether preferences are homothetic - demand scales proportionally
    with income/wealth. This is a stronger condition than GARP.

    For homothetic preferences, the product of expenditure ratios around any
    cycle must be <= 1. Formally:
    - Define r_ij = (p_i @ x_i) / (p_i @ x_j) (expenditure ratio)
    - HARP is violated if there exists a cycle i_1 -> i_2 -> ... -> i_n -> i_1
      such that r_{i_1,i_2} * r_{i_2,i_3} * ... * r_{i_n,i_1} > 1

    The algorithm uses log-space Floyd-Warshall to find maximum product paths,
    then checks for positive log-sum cycles.

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Numerical tolerance for comparisons

    Returns:
        HARPResult with consistency flag and violation details

    Example:
        >>> import numpy as np
        >>> from pyrevealed import ConsumerSession, check_harp
        >>> prices = np.array([[1.0, 2.0], [2.0, 1.0]])
        >>> quantities = np.array([[4.0, 1.0], [1.0, 4.0]])
        >>> session = ConsumerSession(prices=prices, quantities=quantities)
        >>> result = check_harp(session)
        >>> if result.is_homothetic:
        ...     print("Preferences are homothetic")

    References:
        Varian, H. R. (1983). Non-parametric tests of consumer behaviour.
        Review of Economic Studies, 50(1), 99-110.
    """
    start_time = time.perf_counter()

    from pyrevealed.algorithms.garp import check_garp

    E = session.expenditure_matrix  # T x T where E[i,j] = p_i @ x_j
    own_exp = session.own_expenditures  # e_i = E[i,i]
    T = session.num_observations

    # First run GARP for comparison
    garp_result = check_garp(session, tolerance)

    # Compute expenditure ratio matrix: R[i,j] = e_i / E[i,j] = (p_i @ x_i) / (p_i @ x_j)
    # Avoid division by zero
    safe_E = np.where(E > tolerance, E, np.inf)
    ratio_matrix = own_exp[:, np.newaxis] / safe_E

    # Compute log-ratio matrix for numerical stability
    # log_R[i,j] = log(e_i) - log(E[i,j])
    log_own_exp = np.log(np.maximum(own_exp, tolerance))
    log_E = np.log(np.maximum(E, tolerance))
    log_ratio_matrix = log_own_exp[:, np.newaxis] - log_E

    # Set diagonal to 0 (log(1) = 0)
    np.fill_diagonal(log_ratio_matrix, 0.0)

    # Build adjacency matrix: edge i->j exists if ratio >= 1 (i.e., log_ratio >= 0)
    # This means bundle j was affordable when i was chosen
    adjacency = ratio_matrix >= 1.0 - tolerance

    # Modified Floyd-Warshall: track maximum log-sum of paths
    max_log_product = _floyd_warshall_max_log_product(log_ratio_matrix, adjacency)

    # Check for HARP violations:
    # Violation if max_log_product[i,i] > tolerance for any i (cycle with product > 1)
    diagonal_products = np.diag(max_log_product)
    is_consistent = not np.any(diagonal_products > tolerance)

    # Find violating cycles
    violations: list[tuple[Cycle, float]] = []
    max_cycle_product = 1.0

    if not is_consistent:
        violations, max_cycle_product = _find_harp_violations(
            log_ratio_matrix, adjacency, max_log_product, tolerance
        )

    computation_time = (time.perf_counter() - start_time) * 1000

    return HARPResult(
        is_consistent=is_consistent,
        violations=violations,
        max_cycle_product=max_cycle_product,
        expenditure_ratio_matrix=ratio_matrix,
        log_ratio_matrix=log_ratio_matrix,
        garp_result=garp_result,
        computation_time_ms=computation_time,
    )


def _floyd_warshall_max_log_product(
    log_ratios: NDArray[np.float64],
    adjacency: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """
    Modified Floyd-Warshall to find maximum log-product paths.

    Instead of boolean reachability, tracks the maximum sum of log-ratios
    on any path from i to j.

    Args:
        log_ratios: T x T matrix of log(expenditure ratios)
        adjacency: T x T boolean matrix of direct edges

    Returns:
        T x T matrix where result[i,j] = max sum of log_ratios on path i->j
        -inf if no path exists
    """
    T = log_ratios.shape[0]

    # Initialize: direct edges have their log_ratio, no edge = -inf
    max_log = np.where(adjacency, log_ratios, -np.inf)

    # Diagonal: start at 0 (will accumulate cycle costs)
    np.fill_diagonal(max_log, 0.0)

    # Floyd-Warshall: maximize log-sum through intermediate vertices
    for k in range(T):
        # Vectorized update: max_log[i,j] = max(max_log[i,j], max_log[i,k] + max_log[k,j])
        via_k = max_log[:, k:k+1] + max_log[k:k+1, :]
        max_log = np.maximum(max_log, via_k)

    return max_log


def _find_harp_violations(
    log_ratios: NDArray[np.float64],
    adjacency: NDArray[np.bool_],
    max_log_product: NDArray[np.float64],
    tolerance: float,
) -> tuple[list[tuple[Cycle, float]], float]:
    """
    Find cycles that violate HARP (product of ratios > 1).

    Returns:
        Tuple of (list of (cycle, product), maximum product found)
    """
    T = log_ratios.shape[0]
    violations: list[tuple[Cycle, float]] = []
    max_product = 1.0
    seen_cycles: set[frozenset[int]] = set()

    # Find nodes with positive log-product cycles
    for i in range(T):
        if max_log_product[i, i] > tolerance:
            # Reconstruct cycle through i
            cycle = _reconstruct_cycle(adjacency, log_ratios, i, tolerance)
            if cycle is not None:
                cycle_set = frozenset(cycle[:-1])
                if cycle_set not in seen_cycles:
                    seen_cycles.add(cycle_set)

                    # Compute actual product for this cycle
                    product = _compute_cycle_product(cycle, log_ratios)
                    violations.append((tuple(cycle), product))
                    max_product = max(max_product, product)

    return violations, max_product


def _compute_cycle_product(
    cycle: list[int],
    log_ratios: NDArray[np.float64],
) -> float:
    """Compute product of ratios around a cycle."""
    log_sum = 0.0
    for i in range(len(cycle) - 1):
        log_sum += log_ratios[cycle[i], cycle[i + 1]]
    return float(np.exp(log_sum))


def _reconstruct_cycle(
    adjacency: NDArray[np.bool_],
    log_ratios: NDArray[np.float64],
    start: int,
    tolerance: float,
) -> list[int] | None:
    """
    Reconstruct a cycle through the given node using BFS.

    Prioritizes edges with higher log-ratios to find violating cycles.
    """
    T = adjacency.shape[0]

    # Use BFS to find shortest cycle back to start
    visited = {start: [start]}
    queue: deque[int] = deque([start])

    while queue:
        current = queue.popleft()
        path = visited[current]

        # Get neighbors sorted by log_ratio (highest first for better cycles)
        neighbors = np.where(adjacency[current])[0]

        for next_node in neighbors:
            if next_node == start and len(path) > 1:
                # Found cycle back to start
                return path + [start]
            if next_node not in visited:
                visited[next_node] = path + [next_node]
                queue.append(next_node)

    return None


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

# validate_proportional_scaling: Tech-friendly name for check_harp
validate_proportional_scaling = check_harp
"""
Validate that user preferences scale proportionally with budget.

This is the tech-friendly alias for check_harp (HARP = Homothetic Axiom
of Revealed Preference).

Proportional preferences mean the user's relative preferences don't change
with their budget - they just scale up proportionally. This is useful for:
- User segmentation (different budget levels have same relative preferences)
- Demand prediction (can extrapolate to different spending levels)
- Aggregating users across income levels

Example:
    >>> from pyrevealed import BehaviorLog, validate_proportional_scaling
    >>> result = validate_proportional_scaling(user_log)
    >>> if result.is_consistent:
    ...     print("User has proportional preferences")
"""
