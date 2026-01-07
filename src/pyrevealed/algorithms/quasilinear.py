"""Quasilinearity test via cyclic monotonicity."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import QuasilinearityResult
from pyrevealed.core.types import Cycle


def check_quasilinearity(
    session: ConsumerSession,
    tolerance: float = 1e-10,
    max_cycle_length: int = 3,
) -> QuasilinearityResult:
    """
    Test if consumer data is consistent with quasilinear preferences.

    Quasilinear utility has the form U(x, m) = v(x) + m, where m is money
    and v is a concave function. This implies no income effects on goods -
    the marginal utility of money is constant.

    The test checks cyclic monotonicity: for any cycle i_1 -> i_2 -> ... -> i_n -> i_1,
    we must have:
        sum_k p_{i_k} @ (x_{i_{k+1}} - x_{i_k}) >= 0

    This is equivalent to saying the sum of "surplus" around any cycle is non-negative.

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Numerical tolerance for comparisons
        max_cycle_length: Maximum cycle length to check (default: 3)
            Higher values are more thorough but slower

    Returns:
        QuasilinearityResult with is_quasilinear flag and violation details

    Example:
        >>> import numpy as np
        >>> from pyrevealed import ConsumerSession, check_quasilinearity
        >>> prices = np.array([[1.0, 2.0], [2.0, 1.0]])
        >>> quantities = np.array([[4.0, 1.0], [1.0, 4.0]])
        >>> session = ConsumerSession(prices=prices, quantities=quantities)
        >>> result = check_quasilinearity(session)
        >>> if result.is_quasilinear:
        ...     print("No income effects detected")
        >>> else:
        ...     print("Income effects present")

    References:
        Rochet, J. C. (1987). A necessary and sufficient condition for rationalizability
        in a quasi-linear context. Journal of Mathematical Economics, 16(2), 191-200.
    """
    start_time = time.perf_counter()

    T = session.num_observations
    E = session.expenditure_matrix  # T x T where E[i,j] = p_i @ x_j
    own_exp = session.own_expenditures  # e_i = E[i,i]

    # Precompute the "surplus" matrix S where:
    # S[i, j] = p_i @ (x_j - x_i) = E[i,j] - E[i,i]
    # This is the cost difference if we switch from bundle i to bundle j at prices i
    S = E - own_exp[:, np.newaxis]

    violations: list[Cycle] = []
    cycle_sums: dict[Cycle, float] = {}
    worst_violation = 0.0
    num_cycles_tested = 0

    # Check length-2 cycles (pairwise)
    for i in range(T):
        for j in range(i + 1, T):
            # Cycle i -> j -> i
            # Sum = S[i,j] + S[j,i] = (E[i,j] - e_i) + (E[j,i] - e_j)
            # = E[i,j] + E[j,i] - e_i - e_j
            # = (p_i @ x_j + p_j @ x_i) - (p_i @ x_i + p_j @ x_j)
            # = (p_i - p_j) @ (x_j - x_i)
            cycle_sum = float(S[i, j] + S[j, i])

            cycle: Cycle = (i, j, i)
            cycle_sums[cycle] = cycle_sum
            num_cycles_tested += 1

            if cycle_sum < -tolerance:
                violations.append(cycle)
                worst_violation = min(worst_violation, cycle_sum)

    # Check length-3 cycles if requested
    if max_cycle_length >= 3:
        for i in range(T):
            for j in range(T):
                if j == i:
                    continue
                for k in range(T):
                    if k == i or k == j:
                        continue
                    # Cycle i -> j -> k -> i
                    # Sum = S[i,j] + S[j,k] + S[k,i]
                    cycle_sum = float(S[i, j] + S[j, k] + S[k, i])

                    cycle = (i, j, k, i)
                    cycle_sums[cycle] = cycle_sum
                    num_cycles_tested += 1

                    if cycle_sum < -tolerance:
                        violations.append(cycle)
                        worst_violation = min(worst_violation, cycle_sum)

    computation_time = (time.perf_counter() - start_time) * 1000

    return QuasilinearityResult(
        is_quasilinear=len(violations) == 0,
        violations=violations,
        worst_violation_magnitude=worst_violation,
        cycle_sums=cycle_sums,
        num_cycles_tested=num_cycles_tested,
        computation_time_ms=computation_time,
    )


def check_quasilinearity_exhaustive(
    session: ConsumerSession,
    tolerance: float = 1e-10,
) -> QuasilinearityResult:
    """
    Exhaustive quasilinearity test checking all possible cycles.

    This version uses dynamic programming to check cyclic monotonicity
    for cycles of all lengths. More thorough but slower than the default.

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Numerical tolerance for comparisons

    Returns:
        QuasilinearityResult with comprehensive violation analysis
    """
    start_time = time.perf_counter()

    T = session.num_observations
    E = session.expenditure_matrix
    own_exp = session.own_expenditures
    S = E - own_exp[:, np.newaxis]

    # Use Bellman-Ford style detection for negative cycles
    # dist[i] = minimum sum to reach node i from any starting point
    dist = np.zeros(T)
    parent = np.full(T, -1, dtype=int)

    violations: list[Cycle] = []
    cycle_sums: dict[Cycle, float] = {}
    worst_violation = 0.0

    # Relax edges T times
    for _ in range(T):
        for i in range(T):
            for j in range(T):
                if i == j:
                    continue
                if dist[i] + S[i, j] < dist[j] - tolerance:
                    dist[j] = dist[i] + S[i, j]
                    parent[j] = i

    # Check for negative cycles (one more relaxation)
    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            if dist[i] + S[i, j] < dist[j] - tolerance:
                # Negative cycle detected, reconstruct it
                cycle = _reconstruct_negative_cycle(parent, j, S)
                if cycle:
                    cycle_sum = _compute_cycle_sum(cycle, S)
                    if cycle_sum < -tolerance:
                        cycle_tuple: Cycle = tuple(cycle)
                        if cycle_tuple not in cycle_sums:
                            violations.append(cycle_tuple)
                            cycle_sums[cycle_tuple] = cycle_sum
                            worst_violation = min(worst_violation, cycle_sum)

    num_cycles_tested = T * T  # Approximate

    computation_time = (time.perf_counter() - start_time) * 1000

    return QuasilinearityResult(
        is_quasilinear=len(violations) == 0,
        violations=violations,
        worst_violation_magnitude=worst_violation,
        cycle_sums=cycle_sums,
        num_cycles_tested=num_cycles_tested,
        computation_time_ms=computation_time,
    )


def _reconstruct_negative_cycle(
    parent: NDArray[np.int_],
    start: int,
    S: NDArray[np.float64],
) -> list[int]:
    """Reconstruct a negative cycle from parent pointers."""
    T = len(parent)
    visited = set()
    node = start

    # Go back T times to ensure we're in a cycle
    for _ in range(T):
        node = parent[node]
        if node == -1:
            return []

    # Now node is definitely in a cycle, trace it
    cycle_start = node
    cycle = [node]
    node = parent[node]

    while node != cycle_start and node != -1:
        if node in visited:
            break
        visited.add(node)
        cycle.append(node)
        node = parent[node]

    cycle.append(cycle_start)
    cycle.reverse()
    return cycle


def _compute_cycle_sum(cycle: list[int], S: NDArray[np.float64]) -> float:
    """Compute the sum of surplus values around a cycle."""
    total = 0.0
    for i in range(len(cycle) - 1):
        total += S[cycle[i], cycle[i + 1]]
    return total


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

# test_income_invariance: Tech-friendly name for check_quasilinearity
test_income_invariance = check_quasilinearity
"""
Test if user behavior is invariant to income/budget changes.

This is the tech-friendly alias for check_quasilinearity.

Income-invariant behavior means the user's preferences for goods
don't change with their total budget - only relative prices matter.
This is useful for:
- Demand modeling (simpler demand functions)
- Welfare analysis (constant marginal utility of money)
- Price optimization (no need to account for income effects)

Example:
    >>> from pyrevealed import BehaviorLog, test_income_invariance
    >>> result = test_income_invariance(user_log)
    >>> if result.is_quasilinear:
    ...     print("User has constant marginal utility of money")
    >>> else:
    ...     print(f"Income effects detected in {len(result.violations)} cycles")
"""

test_income_invariance_exhaustive = check_quasilinearity_exhaustive
"""
Exhaustive version of test_income_invariance.

Checks all possible cycles, not just short ones.
"""
