"""Varian's Efficiency Index (VEI) - per-observation efficiency scores."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import VEIResult
from pyrevealed.core.exceptions import SolverError, OptimizationError


def compute_vei(
    session: ConsumerSession,
    tolerance: float = 1e-8,
    efficiency_threshold: float = 0.9,
) -> VEIResult:
    """
    Compute Varian's Efficiency Index - per-observation efficiency scores.

    Unlike AEI which finds a single global efficiency e for all observations,
    VEI finds individual efficiency scores e_i for each observation such that
    the data satisfies GARP with minimal total inefficiency.

    The optimization problem is:
        Minimize: sum(1 - e_i) over all i
        Subject to: e_i * (p_i @ x_i) >= p_i @ x_j  for all i, j where i R* j
                    0 <= e_i <= 1

    This identifies which specific observations are problematic, rather than
    just giving a single aggregate score.

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Numerical tolerance for optimization (default: 1e-8)
        efficiency_threshold: Threshold below which observations are flagged
            as problematic (default: 0.9)

    Returns:
        VEIResult with per-observation efficiency vector and summary statistics

    Example:
        >>> import numpy as np
        >>> from pyrevealed import ConsumerSession, compute_vei
        >>> prices = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
        >>> quantities = np.array([[4.0, 1.0], [1.0, 4.0], [2.0, 2.0]])
        >>> session = ConsumerSession(prices=prices, quantities=quantities)
        >>> result = compute_vei(session)
        >>> print(f"Worst observation: {result.worst_observation}")
        >>> print(f"Min efficiency: {result.min_efficiency:.3f}")

    References:
        Varian, H. R. (1990). Goodness-of-fit in optimizing models.
        Journal of Econometrics, 46(1-2), 125-140.
    """
    start_time = time.perf_counter()

    from pyrevealed.algorithms.garp import check_garp

    T = session.num_observations
    E = session.expenditure_matrix  # T x T
    own_exp = session.own_expenditures  # T

    # First check if data is already consistent
    garp_result = check_garp(session)

    if garp_result.is_consistent:
        computation_time = (time.perf_counter() - start_time) * 1000
        return VEIResult(
            efficiency_vector=np.ones(T),
            mean_efficiency=1.0,
            min_efficiency=1.0,
            worst_observation=0,
            problematic_observations=[],
            total_inefficiency=0.0,
            optimization_success=True,
            optimization_status="Data is perfectly consistent",
            computation_time_ms=computation_time,
        )

    # Get transitive closure to know which constraints must hold
    R_star = garp_result.transitive_closure

    # Build LP: minimize sum(-e_i) (equivalently, maximize sum(e_i))
    # Variables: e_1, ..., e_T
    # Constraints: For each i, j where R_star[i, j] = True and i != j:
    #   e_i * own_exp[i] >= E[i, j]
    #   Rearranged: -e_i <= -E[i, j] / own_exp[i]
    #   Or: e_i >= E[i, j] / own_exp[i]

    # Collect constraints
    A_ub_list = []
    b_ub_list = []

    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            if R_star[i, j]:
                # Constraint: e_i >= E[i,j] / own_exp[i]
                # As inequality: -e_i <= -E[i,j] / own_exp[i]
                row = np.zeros(T)
                row[i] = -1.0
                A_ub_list.append(row)
                b_ub_list.append(-E[i, j] / own_exp[i])

    if A_ub_list:
        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
    else:
        A_ub = np.zeros((0, T))
        b_ub = np.zeros(0)

    # Objective: minimize -sum(e_i) (equivalent to maximizing sum)
    c = -np.ones(T)

    # Bounds: 0 <= e_i <= 1
    bounds = [(0.0, 1.0) for _ in range(T)]

    try:
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )
        if result.success:
            success = True
            status = result.message
            e_vector = result.x
        else:
            raise SolverError(
                f"LP solver failed to compute VEI. Status: {result.status}, "
                f"Message: {result.message}"
            )
    except SolverError:
        raise
    except Exception as e:
        raise SolverError(
            f"LP solver failed during VEI computation. Original error: {e}"
        ) from e

    # Clip to [0, 1] for numerical stability
    e_vector = np.clip(e_vector, 0.0, 1.0)

    # Compute summary statistics
    mean_eff = float(np.mean(e_vector))
    min_eff = float(np.min(e_vector))
    worst_obs = int(np.argmin(e_vector))
    problematic = [i for i in range(T) if e_vector[i] < efficiency_threshold]
    total_ineff = float(np.sum(1.0 - e_vector))

    computation_time = (time.perf_counter() - start_time) * 1000

    return VEIResult(
        efficiency_vector=e_vector,
        mean_efficiency=mean_eff,
        min_efficiency=min_eff,
        worst_observation=worst_obs,
        problematic_observations=problematic,
        total_inefficiency=total_ineff,
        optimization_success=success,
        optimization_status=status,
        computation_time_ms=computation_time,
    )


def compute_vei_l2(
    session: ConsumerSession,
    tolerance: float = 1e-8,
    efficiency_threshold: float = 0.9,
) -> VEIResult:
    """
    Compute VEI using L2 norm (minimize sum of squared deviations).

    This version minimizes sum((1 - e_i)^2) instead of sum(1 - e_i),
    which penalizes large deviations more than small ones.

    Uses scipy.optimize.minimize with SLSQP method.

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Numerical tolerance for optimization
        efficiency_threshold: Threshold for flagging problematic observations

    Returns:
        VEIResult with per-observation efficiency vector
    """
    from scipy.optimize import minimize

    start_time = time.perf_counter()

    from pyrevealed.algorithms.garp import check_garp

    T = session.num_observations
    E = session.expenditure_matrix
    own_exp = session.own_expenditures

    garp_result = check_garp(session)

    if garp_result.is_consistent:
        computation_time = (time.perf_counter() - start_time) * 1000
        return VEIResult(
            efficiency_vector=np.ones(T),
            mean_efficiency=1.0,
            min_efficiency=1.0,
            worst_observation=0,
            problematic_observations=[],
            total_inefficiency=0.0,
            optimization_success=True,
            optimization_status="Data is perfectly consistent",
            computation_time_ms=computation_time,
        )

    R_star = garp_result.transitive_closure

    def objective(e: NDArray) -> float:
        return float(np.sum((1.0 - e) ** 2))

    def grad(e: NDArray) -> NDArray:
        return -2.0 * (1.0 - e)

    # Build constraint functions
    constraints = []
    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            if R_star[i, j]:
                bound_val = E[i, j] / own_exp[i]
                # Constraint: e[i] >= bound_val
                constraints.append(
                    {"type": "ineq", "fun": lambda e, idx=i, bv=bound_val: e[idx] - bv}
                )

    bounds = [(0.0, 1.0) for _ in range(T)]
    e0 = np.ones(T) * 0.9  # Initial guess

    try:
        result = minimize(
            objective,
            e0,
            method="SLSQP",
            jac=grad,
            bounds=bounds,
            constraints=constraints,
            options={"ftol": tolerance, "maxiter": 1000},
        )
        if result.success:
            success = True
            status = result.message
            e_vector = result.x
        else:
            raise OptimizationError(
                f"SLSQP optimization failed for VEI L2 computation. "
                f"Message: {result.message}"
            )
    except OptimizationError:
        raise
    except Exception as e:
        raise OptimizationError(
            f"Optimization failed during VEI L2 computation. Original error: {e}"
        ) from e

    e_vector = np.clip(e_vector, 0.0, 1.0)

    mean_eff = float(np.mean(e_vector))
    min_eff = float(np.min(e_vector))
    worst_obs = int(np.argmin(e_vector))
    problematic = [i for i in range(T) if e_vector[i] < efficiency_threshold]
    total_ineff = float(np.sum(1.0 - e_vector))

    computation_time = (time.perf_counter() - start_time) * 1000

    return VEIResult(
        efficiency_vector=e_vector,
        mean_efficiency=mean_eff,
        min_efficiency=min_eff,
        worst_observation=worst_obs,
        problematic_observations=problematic,
        total_inefficiency=total_ineff,
        optimization_success=success,
        optimization_status=status,
        computation_time_ms=computation_time,
    )


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

# compute_granular_integrity: Tech-friendly name for compute_vei
compute_granular_integrity = compute_vei
"""
Compute integrity scores for each individual observation.

This is the tech-friendly alias for compute_vei (Varian's Efficiency Index).

Unlike compute_integrity_score which gives one global score, this
identifies which specific observations are problematic.

Use this to:
- Find specific transactions to investigate
- Identify when user behavior changed
- Detect specific sessions with issues

Example:
    >>> from pyrevealed import BehaviorLog, compute_granular_integrity
    >>> result = compute_granular_integrity(user_log)
    >>> for obs_idx in result.problematic_observations:
    ...     print(f"Investigate observation {obs_idx}")
"""

compute_granular_integrity_l2 = compute_vei_l2
"""
L2 version of compute_granular_integrity.

Penalizes large deviations more than small ones.
"""
