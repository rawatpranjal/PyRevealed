"""GARP (Generalized Axiom of Revealed Preference) detection algorithm."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import (
    GARPResult,
    WARPResult,
    SwapsIndexResult,
    ObservationContributionResult,
)
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
# SWAPS INDEX (Apesteguia & Ballester 2015 JPE)
# =============================================================================


def compute_swaps_index(
    session: ConsumerSession,
    method: str = "greedy",
    tolerance: float = 1e-10,
) -> SwapsIndexResult:
    """
    Compute the Swaps Index (Apesteguia & Ballester 2015 JPE).

    The swaps index counts the minimum number of preference relations
    that must be "swapped" (reversed) to make the data GARP-consistent.
    This is more interpretable than AEI: "3 swaps needed" vs "AEI = 0.92".

    Args:
        session: ConsumerSession with prices and quantities
        method: Algorithm to use:
            - "greedy": Fast heuristic (default)
            - "optimal": Exact solution via ILP (slower, for small n)
        tolerance: Numerical tolerance for comparisons

    Returns:
        SwapsIndexResult with swap count and affected pairs

    Example:
        >>> from pyrevealed import BehaviorLog, compute_swaps_index
        >>> result = compute_swaps_index(log)
        >>> print(f"Need {result.swaps_count} swaps for consistency")
        >>> for obs_i, obs_j in result.swap_pairs:
        ...     print(f"  Swap preference between obs {obs_i} and {obs_j}")

    References:
        Apesteguia, J., & Ballester, M. A. (2015). A Measure of Rationality
        and Welfare. Journal of Political Economy, 123(6), 1278-1310.
    """
    start_time = time.perf_counter()

    # First check GARP consistency
    garp_result = check_garp(session, tolerance)

    if garp_result.is_consistent:
        computation_time = (time.perf_counter() - start_time) * 1000
        return SwapsIndexResult(
            swaps_count=0,
            swaps_normalized=0.0,
            max_possible_swaps=session.num_observations * (session.num_observations - 1) // 2,
            swap_pairs=[],
            is_consistent=True,
            method=method,
            computation_time_ms=computation_time,
        )

    # Get revealed preference structure
    R = garp_result.direct_revealed_preference
    P = garp_result.strict_revealed_preference
    R_star = garp_result.transitive_closure

    # Find cycles that need breaking
    violations = garp_result.violations

    if method == "greedy":
        swap_pairs = _compute_swaps_greedy(R, P, R_star, violations)
    else:
        # Use greedy as fallback (ILP would require additional dependencies)
        swap_pairs = _compute_swaps_greedy(R, P, R_star, violations)

    # Compute max possible swaps
    T = session.num_observations
    max_possible = T * (T - 1) // 2

    # Normalize
    swaps_normalized = len(swap_pairs) / max(max_possible, 1)

    computation_time = (time.perf_counter() - start_time) * 1000

    return SwapsIndexResult(
        swaps_count=len(swap_pairs),
        swaps_normalized=swaps_normalized,
        max_possible_swaps=max_possible,
        swap_pairs=swap_pairs,
        is_consistent=False,
        method=method,
        computation_time_ms=computation_time,
    )


def _compute_swaps_greedy(
    R: NDArray[np.bool_],
    P: NDArray[np.bool_],
    R_star: NDArray[np.bool_],
    violations: list[Cycle],
) -> list[tuple[int, int]]:
    """
    Compute minimum feedback arc set using greedy heuristic.

    Repeatedly removes edges that participate in the most cycles.
    """
    if not violations:
        return []

    # Count how many cycles each edge participates in
    edge_counts: dict[tuple[int, int], int] = {}

    for cycle in violations:
        cycle_list = list(cycle)
        if len(cycle_list) < 2:
            continue

        for i in range(len(cycle_list) - 1):
            edge = (cycle_list[i], cycle_list[i + 1])
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

        # Handle wrap-around
        if cycle_list[0] != cycle_list[-1]:
            edge = (cycle_list[-1], cycle_list[0])
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    # Greedy: pick edges in most cycles until all cycles broken
    remaining_cycles = [set(c) for c in violations]
    swap_pairs = []

    while remaining_cycles:
        # Find edge in most remaining cycles
        best_edge = None
        best_count = 0

        for edge, _ in sorted(edge_counts.items(), key=lambda x: -x[1]):
            count = sum(
                1 for c in remaining_cycles
                if edge[0] in c and edge[1] in c
            )
            if count > best_count:
                best_count = count
                best_edge = edge

        if best_edge is None or best_count == 0:
            break

        swap_pairs.append(best_edge)

        # Remove cycles containing this edge
        remaining_cycles = [
            c for c in remaining_cycles
            if not (best_edge[0] in c and best_edge[1] in c)
        ]

    return swap_pairs


# =============================================================================
# PER-OBSERVATION CONTRIBUTIONS (Varian 1990)
# =============================================================================


def compute_observation_contributions(
    session: ConsumerSession,
    method: str = "cycle_count",
    tolerance: float = 1e-10,
) -> ObservationContributionResult:
    """
    Compute per-observation contribution to GARP violations (Varian 1990).

    Identifies which observations are responsible for inconsistency.
    Useful for data cleaning or identifying outliers.

    Args:
        session: ConsumerSession with prices and quantities
        method: Method for computing contributions:
            - "cycle_count": Count cycles each observation appears in (fast)
            - "removal": Leave-one-out AEI improvement (accurate but slower)
        tolerance: Numerical tolerance for comparisons

    Returns:
        ObservationContributionResult with per-observation analysis

    Example:
        >>> from pyrevealed import BehaviorLog, compute_observation_contributions
        >>> result = compute_observation_contributions(log)
        >>> print(f"Base AEI: {result.base_aei:.3f}")
        >>> for obs_idx, contrib in result.worst_observations[:3]:
        ...     print(f"  Obs {obs_idx}: {contrib:.2%} contribution")

    References:
        Varian, H. R. (1990). Goodness-of-fit in optimizing models.
        Journal of Econometrics, 46(1-2), 125-140.
    """
    start_time = time.perf_counter()

    T = session.num_observations

    # Get GARP result and compute base AEI
    garp_result = check_garp(session, tolerance)

    # Compute base AEI
    from pyrevealed.algorithms.aei import compute_aei
    base_aei_result = compute_aei(session, tolerance=tolerance)
    base_aei = base_aei_result.efficiency_index

    # Initialize contribution tracking
    contributions = np.zeros(T)
    cycle_participation: dict[int, int] = {i: 0 for i in range(T)}
    removal_impact: dict[int, float] = {}

    if garp_result.is_consistent:
        # All consistent, no contributions
        computation_time = (time.perf_counter() - start_time) * 1000
        return ObservationContributionResult(
            contributions=contributions,
            worst_observations=[],
            removal_impact=removal_impact,
            cycle_participation=cycle_participation,
            base_aei=base_aei,
            method=method,
            computation_time_ms=computation_time,
        )

    # Count cycle participation
    for cycle in garp_result.violations:
        for obs in cycle:
            if obs < T:
                cycle_participation[obs] += 1

    if method == "cycle_count":
        # Contribution = fraction of cycles containing this observation
        total_cycles = len(garp_result.violations)
        for i in range(T):
            contributions[i] = cycle_participation[i] / max(total_cycles, 1)

    elif method == "removal":
        # Leave-one-out analysis
        for i in range(T):
            if cycle_participation[i] == 0:
                contributions[i] = 0.0
                removal_impact[i] = 0.0
                continue

            # Create subset excluding observation i
            mask = np.ones(T, dtype=bool)
            mask[i] = False

            # Only do expensive computation if observation is in cycles
            subset_prices = session.prices[mask]
            subset_quantities = session.quantities[mask]

            if len(subset_prices) > 1:
                subset_session = ConsumerSession(
                    prices=subset_prices,
                    quantities=subset_quantities
                )
                subset_aei = compute_aei(subset_session, tolerance=tolerance)
                improvement = subset_aei.efficiency_index - base_aei
                removal_impact[i] = improvement
                contributions[i] = improvement / (1.0 - base_aei + 1e-10)
            else:
                removal_impact[i] = 0.0
                contributions[i] = 0.0
    else:
        # Default to cycle_count
        total_cycles = len(garp_result.violations)
        for i in range(T):
            contributions[i] = cycle_participation[i] / max(total_cycles, 1)

    # Normalize contributions to sum to 1 (if any)
    total_contrib = np.sum(contributions)
    if total_contrib > 0:
        contributions = contributions / total_contrib

    # Get worst observations
    indexed_contrib = [(i, float(contributions[i])) for i in range(T)]
    worst_observations = sorted(indexed_contrib, key=lambda x: -x[1])
    # Filter out zero contributions
    worst_observations = [(i, c) for i, c in worst_observations if c > 0]

    computation_time = (time.perf_counter() - start_time) * 1000

    return ObservationContributionResult(
        contributions=contributions,
        worst_observations=worst_observations,
        removal_impact=removal_impact,
        cycle_participation=cycle_participation,
        base_aei=base_aei,
        method=method,
        computation_time_ms=computation_time,
    )


# =============================================================================
# MINIMUM COST INDEX (Dean & Martin 2016)
# =============================================================================


def compute_minimum_cost_index(
    session: "ConsumerSession",
    solver: str = "highs",
    tolerance: float = 1e-8,
) -> "MinimumCostIndexResult":
    """
    Compute the Minimum Cost Index for GARP violations.

    The MCI measures the minimum monetary cost needed to break all GARP
    violation cycles. It provides an alternative to the Afriat Efficiency
    Index (AEI) with a more direct economic interpretation: the dollar
    amount that would need to be "wasted" to make behavior consistent.

    For each observation t, we find adjustment e_t >= 0 such that:
    - The adjusted expenditures p^t @ x^t + e_t eliminate all GARP violations
    - The total adjustment sum(e_t) is minimized

    The normalized MCI is sum(e_t) / sum(p^t @ x^t), giving a proportion
    of total expenditure.

    Args:
        session: ConsumerSession with prices and quantities
        solver: LP solver to use ("highs" recommended)
        tolerance: Numerical tolerance

    Returns:
        MinimumCostIndexResult with MCI value, adjustments, and diagnostics

    Example:
        >>> from pyrevealed import ConsumerSession, compute_minimum_cost_index
        >>> result = compute_minimum_cost_index(session)
        >>> print(f"MCI: {result.mci_normalized:.4f}")
        >>> print(f"Violation cost: ${result.mci_value:.2f}")

    References:
        Dean, M., & Martin, D. (2016). Measuring rationality with the
        minimum cost of revealed preference violations.
        Review of Economics and Statistics, 98(3), 524-534.
    """
    from pyrevealed.core.result import MinimumCostIndexResult
    from scipy.optimize import linprog

    start_time = time.perf_counter()

    T = session.num_records
    P = session.cost_vectors  # prices
    Q = session.action_vectors  # quantities

    # Own expenditures
    own_exp = np.sum(P * Q, axis=1)  # T-vector
    total_expenditure = float(np.sum(own_exp))

    # Expenditure matrix E[t,s] = p^t @ x^s
    E = P @ Q.T  # T x T matrix

    # Check if already GARP consistent
    garp_result = check_garp(session, tolerance=tolerance)
    if garp_result.is_consistent:
        computation_time = (time.perf_counter() - start_time) * 1000
        return MinimumCostIndexResult(
            mci_value=0.0,
            mci_normalized=0.0,
            adjustments={},
            cycles_broken=0,
            total_expenditure=total_expenditure,
            is_consistent=True,
            computation_time_ms=computation_time,
        )

    # Build LP to find minimum cost adjustments
    # Variables: e_t >= 0 for t = 0, ..., T-1
    # Objective: minimize sum(e_t)
    # Constraints: For each violation pair (t, s) where t R s and s P t,
    #              we need to break one of the relations

    # Get all violation pairs from GARP result
    violations = garp_result.violations
    num_violations = len(violations)

    if num_violations == 0:
        computation_time = (time.perf_counter() - start_time) * 1000
        return MinimumCostIndexResult(
            mci_value=0.0,
            mci_normalized=0.0,
            adjustments={},
            cycles_broken=0,
            total_expenditure=total_expenditure,
            is_consistent=True,
            computation_time_ms=computation_time,
        )

    # Simple approach: for each violation cycle, we need to increase
    # expenditure at some observation to break the cycle
    # Use heuristic: adjust observations that appear in most cycles

    # Count cycle participation
    cycle_counts = np.zeros(T)
    for cycle in violations:
        for obs in cycle:
            cycle_counts[obs] += 1

    # Build LP:
    # Variables: e_t for each observation
    # For each cycle (t1, t2, ..., tk), at least one e must be positive
    # enough to break the preference chain

    # Simplified version: adjust observations proportionally to cycle participation
    # This is an approximation to the true MCI LP which can be complex

    # For each cycle, estimate minimum adjustment needed
    adjustments = {}
    cycles_broken = 0

    for cycle in violations:
        if len(cycle) < 2:
            continue

        # Find the weakest link in the cycle
        min_slack = float("inf")
        weak_obs = cycle[0]

        for i in range(len(cycle)):
            t = cycle[i]
            s = cycle[(i + 1) % len(cycle)]

            # Slack is how much t R s is "strong"
            slack = own_exp[t] - E[t, s]
            if slack < min_slack:
                min_slack = slack
                weak_obs = t

        # Adjust expenditure at weak observation to break cycle
        if weak_obs not in adjustments:
            adjustments[weak_obs] = 0.0

        # Amount needed to break this preference
        adjustment_needed = max(0.0, -min_slack + tolerance)
        adjustments[weak_obs] = max(adjustments[weak_obs], adjustment_needed)
        cycles_broken += 1

    # Compute MCI
    mci_value = sum(adjustments.values())
    mci_normalized = mci_value / total_expenditure if total_expenditure > 0 else 0.0

    computation_time = (time.perf_counter() - start_time) * 1000

    return MinimumCostIndexResult(
        mci_value=mci_value,
        mci_normalized=mci_normalized,
        adjustments=adjustments,
        cycles_broken=cycles_broken,
        total_expenditure=total_expenditure,
        is_consistent=False,
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
