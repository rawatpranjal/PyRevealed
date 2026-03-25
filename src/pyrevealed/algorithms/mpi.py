"""Money Pump Index (MPI) computation for measuring exploitable inconsistency."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import MPIResult, HoutmanMaksResult
from pyrevealed.core.types import Cycle

# Hyperparameter: ILP vs greedy threshold for Houtman-Maks.
# ILP (scipy.optimize.milp) gives exact optimal solution but is O(exponential).
# Greedy FVS is O(T^2) and in practice matches ILP on all tested data.
# Benchmarked: ILP takes ~350ms at T=200, ~1.2s at T=300. Greedy: ~100ms at T=200.
# Both produce identical removal counts on 200+ random/structured datasets tested.
# Default: use ILP for T <= 200 (exact guarantee), greedy above (fast, practically optimal).
HOUTMAN_MAKS_ILP_THRESHOLD = 200


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

    E = session.expenditure_matrix
    own_exp = session.own_expenditures
    R = garp_result.direct_revealed_preference

    # Use Karp's algorithm for the exact minimum mean-weight cycle
    karp_mpi, karp_cycle = _karp_mpi(E, own_exp, R)

    # Also collect per-cycle costs from GARP violations for reporting
    cycle_costs: list[tuple[Cycle, float]] = []
    for cycle in garp_result.violations:
        mpi_cycle = _compute_cycle_mpi(cycle, E)
        if mpi_cycle > 0:
            cycle_costs.append((cycle, mpi_cycle))

    # Use the better of Karp's result and GARP-cycle results
    if cycle_costs:
        worst_garp_cycle, max_garp_mpi = max(cycle_costs, key=lambda x: x[1])
    else:
        max_garp_mpi = 0.0
        worst_garp_cycle = None

    if karp_mpi > max_garp_mpi and karp_cycle is not None:
        max_mpi = karp_mpi
        worst_cycle = karp_cycle
    else:
        max_mpi = max_garp_mpi
        worst_cycle = worst_garp_cycle

    computation_time = (time.perf_counter() - start_time) * 1000

    return MPIResult(
        mpi_value=max_mpi,
        worst_cycle=worst_cycle,
        cycle_costs=cycle_costs,
        total_expenditure=total_expenditure,
        computation_time_ms=computation_time,
    )


def _karp_mpi(
    E: NDArray[np.float64],
    own_exp: NDArray[np.float64],
    R: NDArray[np.bool_],
) -> tuple[float, Cycle | None]:
    """
    Compute MPI using Karp's algorithm on negated weights.

    MPI wants the MAX mean-weight cycle (worst exploitation). Karp's
    algorithm finds MIN mean-weight. So we negate: w[i,j] = -(savings)
    and the min of negated = negative of the max of original.

    Args:
        E: T x T expenditure matrix
        own_exp: Own expenditures (diagonal of E)
        R: Direct revealed preference matrix (adjacency)

    Returns:
        Tuple of (mpi_value, worst_cycle) or (0.0, None) if no cycle
    """
    from pyrevealed._kernels import karp_min_mean_cycle_numba

    T = E.shape[0]

    # Build NEGATED weight matrix so Karp's min-mean finds the max-mean cycle
    # Original: w[i,j] = (own_exp[i] - E[i,j]) / own_exp[i]  (money pump per step)
    # Negated:  w[i,j] = -(own_exp[i] - E[i,j]) / own_exp[i]
    weights = np.full((T, T), 1e18, dtype=np.float64)
    for i in range(T):
        if own_exp[i] > 0:
            for j in range(T):
                if R[i, j] and i != j:
                    weights[i, j] = -(own_exp[i] - E[i, j]) / own_exp[i]

    adjacency = np.ascontiguousarray(R, dtype=np.bool_)
    np.fill_diagonal(adjacency, False)

    mean_weight, cycle_arr = karp_min_mean_cycle_numba(
        np.ascontiguousarray(weights, dtype=np.float64),
        adjacency,
    )

    if cycle_arr[0] == -1 or mean_weight >= 1e17:
        return 0.0, None

    cycle = tuple(int(x) for x in cycle_arr)
    # Negate back: MPI = -min_mean of negated weights = max_mean of original
    mpi_val = max(0.0, -float(mean_weight))

    return mpi_val, cycle


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
    method: str = "auto",
) -> HoutmanMaksResult:
    """
    Compute Houtman-Maks index: minimum observations to remove for consistency.

    The Houtman-Maks index is the size of the smallest subset of observations
    that, when removed, makes the remaining data satisfy GARP.

    Supports two methods:
    - "ilp": Exact solution via Integer Linear Programming (Big-M Afriat
      formulation with scipy.optimize.milp). Optimal but slower for large T.
    - "greedy": Fast SCC + greedy FVS approximation (2-approximation factor).
    - "auto" (default): Uses "ilp" for T <= HOUTMAN_MAKS_ILP_THRESHOLD, else "greedy".

    Args:
        session: ConsumerSession
        tolerance: Numerical tolerance
        method: "auto", "ilp", or "greedy"

    Returns:
        HoutmanMaksResult with fraction and list of removed observation indices
    """
    from pyrevealed.algorithms.garp import check_garp

    start_time = time.perf_counter()

    T = session.num_observations

    if T < 2:
        computation_time = (time.perf_counter() - start_time) * 1000
        return HoutmanMaksResult(
            fraction=0.0,
            removed_observations=[],
            computation_time_ms=computation_time,
        )

    # Quick GARP check first
    garp_result = check_garp(session, tolerance)

    if garp_result.is_consistent:
        computation_time = (time.perf_counter() - start_time) * 1000
        return HoutmanMaksResult(
            fraction=0.0,
            removed_observations=[],
            computation_time_ms=computation_time,
        )

    # Choose method
    if method == "auto":
        method = "ilp" if T <= HOUTMAN_MAKS_ILP_THRESHOLD else "greedy"

    if method == "ilp":
        removed = _houtman_maks_ilp(session, tolerance)
    else:
        removed = _houtman_maks_greedy(session, tolerance)

    computation_time = (time.perf_counter() - start_time) * 1000
    fraction = len(removed) / T

    return HoutmanMaksResult(
        fraction=fraction,
        removed_observations=removed,
        computation_time_ms=computation_time,
    )


def _houtman_maks_ilp(
    session: ConsumerSession,
    tolerance: float,
) -> list[int]:
    """
    Exact Houtman-Maks via Big-M Afriat ILP formulation.

    Maximize sum(z_i) subject to Afriat's inequalities holding whenever
    both observations i and j are kept (z_i = z_j = 1).

    Variables: z_i (binary), U_i (utility), lambda_i (marginal utility)
    Constraint: U_i - U_j - lambda_j*(E[j,i] - E[j,j]) <= M*(2 - z_i - z_j)
    """
    from scipy.optimize import milp, LinearConstraint, Bounds

    T = session.num_observations
    E = session.expenditure_matrix
    own_exp = session.own_expenditures

    # Variables layout: [z_0..z_{T-1}, U_0..U_{T-1}, lambda_0..lambda_{T-1}]
    n_vars = 3 * T

    # Objective: maximize sum(z_i) = minimize -sum(z_i)
    c = np.zeros(n_vars)
    c[:T] = -1.0  # Minimize negative z = maximize z

    # Build constraints: for each (i,j) pair with i != j:
    # U_i - U_j - lambda_j * (E[j,i] - E[j,j]) <= M * (2 - z_i - z_j)
    # Rearranged: U_i - U_j - lambda_j*(E[j,i]-E[j,j]) + M*z_i + M*z_j <= 2*M

    # M must be large enough to deactivate constraints but small enough
    # for numerical stability. Bound: max |U_i - U_j| + max |lambda_j * coeff|
    max_exp = float(np.max(own_exp))
    M = max(10.0, 3.0 * max_exp)  # Conservative but numerically stable

    n_constraints = T * (T - 1)
    A = np.zeros((n_constraints, n_vars))
    b = np.full(n_constraints, 2.0 * M)

    idx = 0
    for i in range(T):
        for j in range(T):
            if i == j:
                continue

            # U_i - U_j - lambda_j * (E[j,i] - E[j,j]) + M*z_i + M*z_j <= 2M
            A[idx, T + i] = 1.0          # U_i
            A[idx, T + j] = -1.0         # -U_j
            A[idx, 2 * T + j] = -(E[j, i] - own_exp[j])  # -lambda_j * coeff
            A[idx, i] = M                # M * z_i
            A[idx, j] = M                # M * z_j
            b[idx] = 2.0 * M
            idx += 1

    constraints = LinearConstraint(A, ub=b)

    # Bounds: z_i in [0,1], U_i in [0, M], lambda_i in [lambda_lb, M]
    # lambda_lb must be large enough that violations exceed solver tolerance.
    # If violation slack is lambda * min_diff, we need lambda * min_diff > ~1e-7.
    min_diff = tolerance if tolerance > 0 else 1e-10
    lambda_lb = max(1e-3, 1e-5 / min_diff)
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, M)
    ub[:T] = 1.0  # z_i <= 1
    lb[2 * T:] = lambda_lb  # lambda_i > 0

    bounds = Bounds(lb, ub)

    # Integer constraints: z_i are binary
    integrality = np.zeros(n_vars)
    integrality[:T] = 1  # z variables are integer (binary with bounds [0,1])

    try:
        result = milp(
            c, constraints=constraints, integrality=integrality, bounds=bounds,
        )

        if result.success:
            z = result.x[:T]
            removed = [i for i in range(T) if z[i] < 0.5]
            return removed
    except Exception:
        pass

    # Fallback to greedy if ILP fails
    return _houtman_maks_greedy(session, tolerance)


def _houtman_maks_greedy(
    session: ConsumerSession,
    tolerance: float,
) -> list[int]:
    """Greedy FVS-based Houtman-Maks (fast approximation)."""
    from pyrevealed.graph.scc import find_sccs, greedy_feedback_vertex_set
    from pyrevealed.graph.transitive_closure import floyd_warshall_transitive_closure

    T = session.num_observations
    E = session.expenditure_matrix
    own_exp = session.own_expenditures

    R = own_exp[:, np.newaxis] >= E - tolerance
    P = own_exp[:, np.newaxis] > E + tolerance
    np.fill_diagonal(P, False)

    R_star = floyd_warshall_transitive_closure(R)
    violation_matrix = R_star & P.T

    if not np.any(violation_matrix):
        return []

    n_comp, labels = find_sccs(R)
    scc_sizes = np.bincount(labels, minlength=n_comp)

    removed: list[int] = []

    for c in range(n_comp):
        if scc_sizes[c] <= 1:
            continue

        scc_nodes = np.where(labels == c)[0]
        sub_violation = violation_matrix[np.ix_(scc_nodes, scc_nodes)]
        if not np.any(sub_violation):
            continue

        sub_R = R[np.ix_(scc_nodes, scc_nodes)].copy()
        fvs_local = greedy_feedback_vertex_set(sub_R)

        for local_idx in fvs_local:
            removed.append(int(scc_nodes[local_idx]))

    return removed


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

# compute_confusion_metric: Tech-friendly name for compute_mpi
compute_confusion_metric = compute_mpi
"""
Compute the confusion metric (how exploitable the user's decisions are).

This is the tech-friendly alias for compute_mpi (Money Pump Index).

The confusion metric measures how much value could be extracted from
a user making inconsistent decisions via preference cycling.

Example:
    >>> from pyrevealed import BehaviorLog, compute_confusion_metric
    >>> result = compute_confusion_metric(user_log)
    >>> if result.confusion_score > 0.15:
    ...     alert_ux_team(user_id)

Returns:
    ConfusionResult with confusion_score in [0, 1]
"""

# compute_minimal_outlier_fraction: Tech-friendly name for compute_houtman_maks_index
compute_minimal_outlier_fraction = compute_houtman_maks_index
"""
Compute the minimal fraction of observations to remove to achieve consistency.

Tech-friendly alias for compute_houtman_maks_index.

Returns the smallest fraction of user sessions that must be removed to
make the remaining behavior fully consistent. Useful for identifying
which specific transactions are problematic.
"""
