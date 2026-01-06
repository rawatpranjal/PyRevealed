"""Money Pump Index (MPI) computation for measuring exploitable inconsistency."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import MPIResult
from pyrevealed.core.types import Cycle


def compute_mpi(
    session: ConsumerSession,
    tolerance: float = 1e-10,
) -> MPIResult:
    """
    Compute Money Pump Index for the consumer data.

    The MPI measures the percentage of total expenditure that could be
    "pumped" from a consumer exhibiting cyclic preferences by an arbitrager.

    For a violation cycle k1 -> k2 -> ... -> kn -> k1:

        MPI = sum(p_ki @ x_ki - p_ki @ x_{ki+1}) / sum(p_ki @ x_ki)

    This represents the fraction of expenditure that is "wasted" due to
    inconsistent choices - money that could be extracted by a clever
    arbitrager exploiting the consumer's cyclic preferences.

    Interpretation:
    - MPI = 0.0: Consistent behavior (no money can be pumped)
    - MPI = 0.10: 10% of budget could be extracted
    - MPI = 1.0: Complete irrational behavior (theoretical maximum)

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Numerical tolerance for GARP detection

    Returns:
        MPIResult with MPI value, worst cycle, and all cycle costs

    Example:
        >>> import numpy as np
        >>> from pyrevealed import ConsumerSession, compute_mpi
        >>> # Data with GARP violation
        >>> prices = np.array([[1.0, 1.0], [1.0, 1.0]])
        >>> quantities = np.array([[3.0, 1.0], [1.0, 3.0]])
        >>> session = ConsumerSession(prices=prices, quantities=quantities)
        >>> result = compute_mpi(session)
        >>> print(f"MPI: {result.mpi_value:.4f}")
    """
    start_time = time.perf_counter()

    # First check GARP to find violations
    from pyrevealed.algorithms.garp import check_garp

    garp_result = check_garp(session, tolerance)

    total_expenditure = float(session.own_expenditures.sum())

    if garp_result.is_consistent:
        computation_time = (time.perf_counter() - start_time) * 1000
        return MPIResult(
            mpi_value=0.0,
            worst_cycle=None,
            cycle_costs=[],
            total_expenditure=total_expenditure,
            computation_time_ms=computation_time,
        )

    # Compute MPI for each violation cycle
    E = session.expenditure_matrix

    cycle_costs: list[tuple[Cycle, float]] = []

    for cycle in garp_result.violations:
        mpi_cycle = _compute_cycle_mpi(cycle, E)
        if mpi_cycle > 0:  # Only include positive MPI cycles
            cycle_costs.append((cycle, mpi_cycle))

    # Find worst (highest MPI) cycle
    if cycle_costs:
        worst_cycle, max_mpi = max(cycle_costs, key=lambda x: x[1])
    else:
        # Fallback: use simple MPI calculation
        max_mpi = _compute_simple_mpi(session, garp_result.violations)
        worst_cycle = garp_result.violations[0] if garp_result.violations else None

    computation_time = (time.perf_counter() - start_time) * 1000

    return MPIResult(
        mpi_value=max_mpi,
        worst_cycle=worst_cycle,
        cycle_costs=cycle_costs,
        total_expenditure=total_expenditure,
        computation_time_ms=computation_time,
    )


def _compute_cycle_mpi(
    cycle: Cycle,
    E: NDArray[np.float64],
) -> float:
    """
    Compute MPI for a single cycle.

    For cycle k1 -> k2 -> ... -> kn -> k1:

        MPI = sum_{i=1}^{n}(E[ki, ki] - E[ki, k_{i+1}]) / sum_{i=1}^{n}(E[ki, ki])

    The numerator is the total "savings" if the consumer had chosen the
    next bundle in the cycle at each step. The denominator is total expenditure
    in the cycle.

    Args:
        cycle: Tuple of observation indices forming the cycle
        E: Expenditure matrix where E[i,j] = p_i @ q_j

    Returns:
        MPI value for this cycle (0 to 1)
    """
    if len(cycle) < 2:
        return 0.0

    numerator = 0.0
    denominator = 0.0

    # cycle is (k1, k2, ..., kn, k1) where last element repeats first
    for i in range(len(cycle) - 1):
        ki = cycle[i]
        ki_next = cycle[i + 1]

        # E[ki, ki] - E[ki, ki_next] = savings from choosing ki_next instead
        savings = E[ki, ki] - E[ki, ki_next]
        numerator += savings

        # E[ki, ki] = expenditure at observation ki
        denominator += E[ki, ki]

    if denominator <= 0:
        return 0.0

    mpi = numerator / denominator

    # MPI should be non-negative; clamp to handle numerical issues
    return max(0.0, mpi)


def _compute_simple_mpi(
    session: ConsumerSession,
    violations: list[Cycle],
) -> float:
    """
    Compute a simple aggregate MPI measure.

    This is a fallback when cycle-based MPI is not well-defined.
    It computes the average "wasted" money across all violation pairs.

    Args:
        session: ConsumerSession
        violations: List of violation cycles

    Returns:
        Simple MPI estimate
    """
    if not violations:
        return 0.0

    E = session.expenditure_matrix
    total_waste = 0.0
    total_spend = 0.0

    for cycle in violations:
        for i in range(len(cycle) - 1):
            ki = cycle[i]
            ki_next = cycle[i + 1]

            waste = E[ki, ki] - E[ki, ki_next]
            if waste > 0:
                total_waste += waste
            total_spend += E[ki, ki]

    if total_spend <= 0:
        return 0.0

    return total_waste / total_spend


def compute_houtman_maks_index(
    session: ConsumerSession,
    tolerance: float = 1e-10,
) -> tuple[float, list[int]]:
    """
    Compute Houtman-Maks index: minimum observations to remove for consistency.

    The Houtman-Maks index is the size of the smallest subset of observations
    that, when removed, makes the remaining data satisfy GARP.

    This is NP-hard in general, so we use a greedy approximation:
    1. Find all violations
    2. Remove the observation involved in most violations
    3. Repeat until consistent

    Args:
        session: ConsumerSession
        tolerance: Numerical tolerance

    Returns:
        Tuple of (index as fraction, list of removed observation indices)

    Note:
        The index is returned as a fraction: num_removed / num_observations.
        A lower value indicates more consistent behavior.
    """
    from pyrevealed.algorithms.garp import check_garp

    T = session.num_observations
    remaining = list(range(T))
    removed: list[int] = []

    while True:
        # Create sub-session with remaining observations
        sub_prices = session.prices[remaining]
        sub_quantities = session.quantities[remaining]

        if len(remaining) < 2:
            break

        sub_session = ConsumerSession(
            prices=sub_prices,
            quantities=sub_quantities,
        )

        result = check_garp(sub_session, tolerance)

        if result.is_consistent:
            break

        # Find observation in most violations
        violation_counts: dict[int, int] = {}
        for cycle in result.violations:
            for idx in cycle:
                if idx < len(remaining):
                    orig_idx = remaining[idx]
                    violation_counts[orig_idx] = violation_counts.get(orig_idx, 0) + 1

        if not violation_counts:
            break

        # Remove observation with most violations
        worst_obs = max(violation_counts.keys(), key=lambda k: violation_counts[k])
        remaining.remove(worst_obs)
        removed.append(worst_obs)

    index = len(removed) / T
    return index, removed
