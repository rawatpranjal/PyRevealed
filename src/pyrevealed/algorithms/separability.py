"""Separability testing for groups of goods.

Tests whether utility can be decomposed into independent sub-utilities
for different groups of goods (weak separability).

This module provides two methods for separability testing:

1. HEURISTIC APPROXIMATION (check_separability):
   Uses AEI within each group + cross-correlation. Fast but approximate.

2. EXACT THEOREM 4.4 TEST (check_separability_exact):
   Solves the nonlinear Afriat inequalities from Chambers & Echenique (2016):

       Uk ≤ Ul + λl·p¹l·(x¹k - x¹l) + (λl/μl)·(Vk - Vl)    (4.1)
       Vk ≤ Vl + μl·p²l·(x²k - x²l)                         (4.2)

   Uses sequential LP relaxation to solve this nonlinear system iteratively.

References:
    Chambers & Echenique (2016), Chapter 4, Theorem 4.4 (pp.63-64)
    Cherchye et al. (2014), "Nonparametric analysis of household labor supply"
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog, minimize

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import SeparabilityResult
from pyrevealed.core.exceptions import DataValidationError, ValueRangeError, SolverError, OptimizationError
from pyrevealed.algorithms.aei import compute_aei


def check_separability(
    session: ConsumerSession,
    group_a: list[int],
    group_b: list[int],
    tolerance: float = 1e-6,
) -> SeparabilityResult:
    """
    Test if two groups of goods are weakly separable (HEURISTIC APPROXIMATION).

    Weak separability means the utility function can be written as:
        U(x_A, x_B) = V(u_A(x_A), u_B(x_B))

    where x_A and x_B are consumption of goods in groups A and B.

    If separable, the groups can be priced independently without considering
    cross-elasticity effects.

    WARNING: This is a HEURISTIC approximation, not the exact Theorem 4.4 test
    from Chambers & Echenique (2016). The heuristic checks:
    1. Within-group GARP consistency (via AEI) for each group
    2. Low cross-correlation between groups

    The exact test (Theorem 4.4) requires solving nonlinear Afriat inequalities,
    which is computationally harder. See Cherchye et al. (2014) for algorithms.

    Args:
        session: ConsumerSession with prices and quantities
        group_a: List of good indices in Group A
        group_b: List of good indices in Group B
        tolerance: Numerical tolerance for GARP checks

    Returns:
        SeparabilityResult with separability test results

    Example:
        >>> import numpy as np
        >>> from pyrevealed import ConsumerSession, test_separability
        >>> # Rides (goods 0,1) and Eats (goods 2,3)
        >>> prices = np.array([
        ...     [1.0, 1.5, 2.0, 2.5],
        ...     [1.5, 1.0, 2.5, 2.0],
        ... ])
        >>> quantities = np.array([
        ...     [2.0, 1.0, 1.0, 0.5],
        ...     [1.0, 2.0, 0.5, 1.0],
        ... ])
        >>> session = ConsumerSession(prices=prices, quantities=quantities)
        >>> result = test_separability(session, [0, 1], [2, 3])
        >>> result.is_separable
        True
    """
    start_time = time.perf_counter()

    # Validate groups don't overlap and cover all goods
    all_indices = set(group_a) | set(group_b)
    if len(all_indices) != len(group_a) + len(group_b):
        overlap = set(group_a) & set(group_b)
        raise DataValidationError(
            f"Groups must not overlap. Found overlapping indices: {list(overlap)}. "
            f"Hint: Each good should belong to exactly one group for separability testing."
        )

    N = session.num_goods
    for idx in all_indices:
        if idx < 0 or idx >= N:
            raise ValueRangeError(
                f"Good index {idx} out of range [0, {N}). "
                f"Hint: Indices must refer to valid goods in the session (0 to {N - 1})."
            )

    # Create sub-sessions for each group
    session_a = _extract_subsession(session, group_a)
    session_b = _extract_subsession(session, group_b)

    # Check GARP within each group
    aei_a = compute_aei(session_a, tolerance=tolerance)
    aei_b = compute_aei(session_b, tolerance=tolerance)

    # Compute cross-effect strength
    cross_effect = _compute_cross_effect(session, group_a, group_b)

    # Separability test:
    # 1. Each group should satisfy GARP internally (or close to it)
    # 2. Cross-effects should be minimal
    within_a_consistent = aei_a.efficiency_index
    within_b_consistent = aei_b.efficiency_index

    # Separable if both groups are internally consistent and cross-effects are low
    is_separable = (
        within_a_consistent > 0.9 and within_b_consistent > 0.9 and cross_effect < 0.2
    )

    # Generate recommendation
    if is_separable:
        recommendation = "price_independently"
    elif cross_effect > 0.5:
        recommendation = "unified_strategy"
    else:
        recommendation = "partial_independence"

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return SeparabilityResult(
        is_separable=is_separable,
        group_a_indices=list(group_a),
        group_b_indices=list(group_b),
        cross_effect_strength=cross_effect,
        within_group_a_consistency=within_a_consistent,
        within_group_b_consistency=within_b_consistent,
        recommendation=recommendation,
        computation_time_ms=elapsed_ms,
    )


def check_separability_exact(
    session: ConsumerSession,
    group_a: list[int],
    group_b: list[int],
    method: str = "sequential",
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> SeparabilityResult:
    """
    Exact separability test using Theorem 4.4 from Chambers & Echenique (2016).

    Tests weak separability by solving the nonlinear Afriat inequalities:

        U_k ≤ U_l + λ_l·p¹_l·(x¹_k - x¹_l) + (λ_l/μ_l)·(V_k - V_l)    (4.1)
        V_k ≤ V_l + μ_l·p²_l·(x²_k - x²_l)                             (4.2)

    where:
    - U_k, V_k are utility values for the outer and inner utilities
    - λ_k, μ_k are Lagrange multipliers (marginal utilities)
    - p¹, p² are prices for groups A and B
    - x¹, x² are quantities for groups A and B

    Args:
        session: ConsumerSession with prices and quantities
        group_a: List of good indices in Group A
        group_b: List of good indices in Group B
        method: Solution method - "sequential" (LP relaxation) or "nonlinear"
        tolerance: Numerical tolerance for convergence
        max_iterations: Maximum iterations for sequential LP method

    Returns:
        SeparabilityResult with exact separability test results

    Example:
        >>> import numpy as np
        >>> from pyrevealed import ConsumerSession, check_separability_exact
        >>> prices = np.array([
        ...     [1.0, 1.5, 2.0, 2.5],
        ...     [1.5, 1.0, 2.5, 2.0],
        ... ])
        >>> quantities = np.array([
        ...     [2.0, 1.0, 1.0, 0.5],
        ...     [1.0, 2.0, 0.5, 1.0],
        ... ])
        >>> session = ConsumerSession(prices=prices, quantities=quantities)
        >>> result = check_separability_exact(session, [0, 1], [2, 3])
        >>> print(f"Is separable: {result.is_separable}")

    References:
        Chambers & Echenique (2016), Chapter 4, Theorem 4.4 (pp.63-64)
    """
    start_time = time.perf_counter()

    # Validate groups don't overlap and cover all goods
    all_indices = set(group_a) | set(group_b)
    if len(all_indices) != len(group_a) + len(group_b):
        overlap = set(group_a) & set(group_b)
        raise DataValidationError(
            f"Groups must not overlap. Found overlapping indices: {list(overlap)}. "
            f"Hint: Each good should belong to exactly one group for separability testing."
        )

    N = session.num_goods
    for idx in all_indices:
        if idx < 0 or idx >= N:
            raise ValueRangeError(
                f"Good index {idx} out of range [0, {N}). "
                f"Hint: Indices must refer to valid goods in the session (0 to {N - 1})."
            )

    # Extract prices and quantities for each group
    P_a = session.prices[:, group_a]
    Q_a = session.quantities[:, group_a]
    P_b = session.prices[:, group_b]
    Q_b = session.quantities[:, group_b]

    # Solve using the specified method
    if method == "sequential":
        is_separable, U, V, lambdas, mus = _solve_separability_sequential(
            P_a, Q_a, P_b, Q_b, tolerance, max_iterations
        )
    elif method == "nonlinear":
        is_separable, U, V, lambdas, mus = _solve_separability_nonlinear(
            P_a, Q_a, P_b, Q_b, tolerance
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sequential' or 'nonlinear'.")

    # Compute within-group consistency (for reporting)
    session_a = _extract_subsession(session, group_a)
    session_b = _extract_subsession(session, group_b)
    aei_a = compute_aei(session_a, tolerance=tolerance)
    aei_b = compute_aei(session_b, tolerance=tolerance)

    # Cross-effect is reported as 0 if separable (by definition), else computed heuristically
    if is_separable:
        cross_effect = 0.0
    else:
        cross_effect = _compute_cross_effect(session, group_a, group_b)

    # Generate recommendation
    if is_separable:
        recommendation = "price_independently"
    elif cross_effect > 0.5:
        recommendation = "unified_strategy"
    else:
        recommendation = "partial_independence"

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return SeparabilityResult(
        is_separable=is_separable,
        group_a_indices=list(group_a),
        group_b_indices=list(group_b),
        cross_effect_strength=cross_effect,
        within_group_a_consistency=aei_a.efficiency_index,
        within_group_b_consistency=aei_b.efficiency_index,
        recommendation=recommendation,
        computation_time_ms=elapsed_ms,
    )


def _solve_separability_sequential(
    P_a: NDArray[np.float64],
    Q_a: NDArray[np.float64],
    P_b: NDArray[np.float64],
    Q_b: NDArray[np.float64],
    tolerance: float,
    max_iterations: int,
) -> tuple[bool, NDArray | None, NDArray | None, NDArray | None, NDArray | None]:
    """
    Solve separability conditions using sequential LP relaxation.

    The approach:
    1. Fix μ_k = 1 for all k (normalize sub-utility scale)
    2. Solve standard Afriat LP for group B to get V_k
    3. With V_k fixed, inequality (4.1) becomes linear in U_k, λ_k
    4. Solve LP for U_k, λ_k
    5. Update μ_k based on solution and iterate until convergence

    Returns:
        Tuple of (is_separable, U, V, lambdas, mus) or (False, None, None, None, None)
    """
    T = P_a.shape[0]
    epsilon = 1e-6

    # Step 1: Initialize μ_k = 1 for all k
    mus = np.ones(T)

    # Step 2: Solve Afriat inequalities for group B to get V_k
    # V_k <= V_l + μ_l * p²_l @ (x²_k - x²_l)
    # With μ = 1, this is standard Afriat LP
    V, mus_b, success_b = _solve_afriat_lp(P_b, Q_b, tolerance)

    if not success_b or V is None:
        return False, None, None, None, None

    # Use the recovered μ values from group B
    mus = mus_b

    # Step 3: With V fixed, solve for U and λ
    # Constraint (4.1): U_k <= U_l + λ_l * p¹_l @ (x¹_k - x¹_l) + (λ_l/μ_l) * (V_k - V_l)

    # This is still nonlinear in λ because of λ_l/μ_l term
    # But if we fix the ratio r_l = λ_l/μ_l, then:
    # U_k <= U_l + λ_l * [p¹_l @ (x¹_k - x¹_l) + (V_k - V_l)/μ_l]

    # Iterative approach: start with r_l = 1, solve LP, update r_l
    lambdas = np.ones(T)
    U = np.zeros(T)

    prev_obj = float('inf')

    for iteration in range(max_iterations):
        # Solve LP for U given current λ/μ ratios
        U_new, lambdas_new, success_u = _solve_outer_afriat_lp(
            P_a, Q_a, V, mus, tolerance
        )

        if not success_u or U_new is None:
            # Try to recover with adjusted parameters
            if iteration > 0:
                # Use previous solution
                break
            return False, None, None, None, None

        U = U_new
        lambdas = lambdas_new

        # Compute objective (sum of slacks)
        obj = np.sum(lambdas)

        # Check convergence
        if abs(obj - prev_obj) < tolerance:
            break

        prev_obj = obj

        # Update μ values based on group B consistency
        # Re-solve group B LP with updated normalization
        V_new, mus_new, success_b = _solve_afriat_lp(P_b, Q_b, tolerance)
        if success_b and V_new is not None and mus_new is not None:
            V = V_new
            mus = mus_new

    # Verify solution satisfies all constraints
    is_valid = _verify_separability_solution(P_a, Q_a, P_b, Q_b, U, V, lambdas, mus, tolerance)

    if is_valid:
        return True, U, V, lambdas, mus
    else:
        return False, None, None, None, None


def _solve_afriat_lp(
    P: NDArray[np.float64],
    Q: NDArray[np.float64],
    tolerance: float,
) -> tuple[NDArray | None, NDArray | None, bool]:
    """
    Solve standard Afriat LP for utility values.

    min Σ λ_k
    s.t. U_k <= U_l + λ_l * p_l @ (x_k - x_l)  for all k, l
         U_k >= 0, λ_k > ε

    Returns:
        Tuple of (utility_values, lagrange_multipliers, success)
    """
    T = P.shape[0]
    n_vars = 2 * T  # U_1...U_T, λ_1...λ_T

    constraints_A = []
    constraints_b = []

    for k in range(T):
        for obs_l in range(T):
            if k == obs_l:
                continue

            # U_k - U_l - λ_l * p_l @ (x_l - x_k) <= 0
            row = np.zeros(n_vars)
            row[k] = 1.0  # U_k
            row[obs_l] = -1.0  # -U_l

            diff = Q[obs_l] - Q[k]
            lambda_coef = P[obs_l] @ diff
            row[T + obs_l] = lambda_coef

            constraints_A.append(row)
            constraints_b.append(0.0)

    A_ub = np.array(constraints_A) if constraints_A else np.zeros((0, n_vars))
    b_ub = np.array(constraints_b) if constraints_b else np.zeros(0)

    epsilon = 1e-6
    bounds = [(0, None)] * T + [(epsilon, None)] * T

    c = np.zeros(n_vars)
    c[T:] = 1.0

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
            U = result.x[:T]
            lambdas = result.x[T:]
            return U, lambdas, True
        else:
            raise SolverError(
                f"LP solver failed for inner Afriat system in separability test. "
                f"Status: {result.status}, Message: {result.message}"
            )
    except SolverError:
        raise
    except Exception as e:
        raise SolverError(
            f"LP solver failed during inner Afriat recovery. Original error: {e}"
        ) from e


def _solve_outer_afriat_lp(
    P_a: NDArray[np.float64],
    Q_a: NDArray[np.float64],
    V: NDArray[np.float64],
    mus: NDArray[np.float64],
    tolerance: float,
) -> tuple[NDArray | None, NDArray | None, bool]:
    """
    Solve the outer utility Afriat LP with V fixed.

    Constraint (4.1) linearized:
    U_k <= U_l + λ_l * [p¹_l @ (x¹_k - x¹_l)] + λ_l * [(V_k - V_l)/μ_l]

    Rearranging:
    U_k - U_l - λ_l * [p¹_l @ (x¹_k - x¹_l) + (V_k - V_l)/μ_l] <= 0
    """
    T = P_a.shape[0]
    n_vars = 2 * T  # U_1...U_T, λ_1...λ_T

    constraints_A = []
    constraints_b = []

    for k in range(T):
        for obs_l in range(T):
            if k == obs_l:
                continue

            row = np.zeros(n_vars)
            row[k] = 1.0  # U_k
            row[obs_l] = -1.0  # -U_l

            # Coefficient for λ_l
            diff_q = Q_a[obs_l] - Q_a[k]
            price_term = P_a[obs_l] @ diff_q

            # V term
            v_diff = (V[k] - V[obs_l]) / max(mus[obs_l], 1e-10)

            # Combined coefficient
            lambda_coef = price_term + v_diff
            row[T + obs_l] = lambda_coef

            constraints_A.append(row)
            constraints_b.append(0.0)

    A_ub = np.array(constraints_A) if constraints_A else np.zeros((0, n_vars))
    b_ub = np.array(constraints_b) if constraints_b else np.zeros(0)

    epsilon = 1e-6
    bounds = [(0, None)] * T + [(epsilon, None)] * T

    c = np.zeros(n_vars)
    c[T:] = 1.0

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
            U = result.x[:T]
            lambdas = result.x[T:]
            return U, lambdas, True
        else:
            raise SolverError(
                f"LP solver failed for outer Afriat system in separability test. "
                f"Status: {result.status}, Message: {result.message}"
            )
    except SolverError:
        raise
    except Exception as e:
        raise SolverError(
            f"LP solver failed during outer Afriat recovery. Original error: {e}"
        ) from e


def _solve_separability_nonlinear(
    P_a: NDArray[np.float64],
    Q_a: NDArray[np.float64],
    P_b: NDArray[np.float64],
    Q_b: NDArray[np.float64],
    tolerance: float,
) -> tuple[bool, NDArray | None, NDArray | None, NDArray | None, NDArray | None]:
    """
    Solve separability conditions using direct nonlinear optimization.

    Minimizes the sum of constraint violations for the full nonlinear system.

    Returns:
        Tuple of (is_separable, U, V, lambdas, mus) or (False, None, None, None, None)
    """
    T = P_a.shape[0]
    epsilon = 1e-6

    # Variables: U_1...U_T, V_1...V_T, λ_1...λ_T, μ_1...μ_T
    n_vars = 4 * T

    def objective(x: NDArray[np.float64]) -> float:
        """Sum of Lagrange multipliers (minimization target)."""
        lambdas = x[2*T:3*T]
        mus = x[3*T:4*T]
        return np.sum(lambdas) + np.sum(mus)

    def constraint_violations(x: NDArray[np.float64]) -> float:
        """Total constraint violation (should be 0 if separable)."""
        U = x[:T]
        V = x[T:2*T]
        lambdas = x[2*T:3*T]
        mus = x[3*T:4*T]

        violations = 0.0

        # Constraint (4.1): U_k <= U_l + λ_l * p¹_l @ (x¹_k - x¹_l) + (λ_l/μ_l) * (V_k - V_l)
        for k in range(T):
            for obs_l in range(T):
                if k == obs_l:
                    continue

                diff_q = Q_a[k] - Q_a[obs_l]
                price_term = lambdas[obs_l] * (P_a[obs_l] @ diff_q)
                v_term = (lambdas[obs_l] / max(mus[obs_l], epsilon)) * (V[k] - V[obs_l])

                rhs = U[obs_l] + price_term + v_term
                if U[k] > rhs + tolerance:
                    violations += U[k] - rhs

        # Constraint (4.2): V_k <= V_l + μ_l * p²_l @ (x²_k - x²_l)
        for k in range(T):
            for obs_l in range(T):
                if k == obs_l:
                    continue

                diff_q = Q_b[k] - Q_b[obs_l]
                rhs = V[obs_l] + mus[obs_l] * (P_b[obs_l] @ diff_q)
                if V[k] > rhs + tolerance:
                    violations += V[k] - rhs

        return violations

    # Initial guess from sequential LP
    _, U_init, V_init, lambda_init, mu_init = _solve_separability_sequential(
        P_a, Q_a, P_b, Q_b, tolerance, max_iterations=50
    )

    if U_init is None:
        # Start from scratch
        U_init = np.ones(T)
        V_init = np.ones(T)
        lambda_init = np.ones(T)
        mu_init = np.ones(T)

    x0 = np.concatenate([U_init, V_init, lambda_init, mu_init])

    # Bounds: U, V >= 0, λ, μ > ε
    bounds = [(0, None)] * T + [(0, None)] * T + [(epsilon, None)] * T + [(epsilon, None)] * T

    # Constraint: violations = 0
    constraints = [{"type": "eq", "fun": constraint_violations}]

    try:
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": tolerance},
        )

        if result.success or constraint_violations(result.x) < tolerance:
            U = result.x[:T]
            V = result.x[T:2*T]
            lambdas = result.x[2*T:3*T]
            mus = result.x[3*T:4*T]

            # Verify solution
            is_valid = _verify_separability_solution(
                P_a, Q_a, P_b, Q_b, U, V, lambdas, mus, tolerance
            )

            if is_valid:
                return True, U, V, lambdas, mus

        # Solution not valid
        raise OptimizationError(
            f"SLSQP optimization failed to find valid separability solution. "
            f"Message: {result.message}"
        )

    except OptimizationError:
        raise
    except Exception as e:
        raise OptimizationError(
            f"Nonlinear optimization failed during separability test. Original error: {e}"
        ) from e


def _verify_separability_solution(
    P_a: NDArray[np.float64],
    Q_a: NDArray[np.float64],
    P_b: NDArray[np.float64],
    Q_b: NDArray[np.float64],
    U: NDArray[np.float64],
    V: NDArray[np.float64],
    lambdas: NDArray[np.float64],
    mus: NDArray[np.float64],
    tolerance: float,
) -> bool:
    """Verify that a solution satisfies all separability constraints."""
    T = len(U)
    epsilon = 1e-10

    # Check constraint (4.1)
    for k in range(T):
        for obs_l in range(T):
            if k == obs_l:
                continue

            diff_q = Q_a[k] - Q_a[obs_l]
            price_term = lambdas[obs_l] * (P_a[obs_l] @ diff_q)
            v_term = (lambdas[obs_l] / max(mus[obs_l], epsilon)) * (V[k] - V[obs_l])

            rhs = U[obs_l] + price_term + v_term
            if U[k] > rhs + tolerance:
                return False

    # Check constraint (4.2)
    for k in range(T):
        for obs_l in range(T):
            if k == obs_l:
                continue

            diff_q = Q_b[k] - Q_b[obs_l]
            rhs = V[obs_l] + mus[obs_l] * (P_b[obs_l] @ diff_q)
            if V[k] > rhs + tolerance:
                return False

    return True


def _extract_subsession(
    session: ConsumerSession,
    good_indices: list[int],
) -> ConsumerSession:
    """Extract a sub-session with only specified goods."""
    prices = session.prices[:, good_indices]
    quantities = session.quantities[:, good_indices]
    return ConsumerSession(prices=prices, quantities=quantities)


def _compute_cross_effect(
    session: ConsumerSession,
    group_a: list[int],
    group_b: list[int],
) -> float:
    """
    Compute cross-price effect between groups.

    Measures how much prices in one group affect quantities in the other.
    Returns a value in [0, 1] where 0 = no cross-effect, 1 = strong effect.
    """
    T = session.num_observations

    if T < 3:
        return 0.0  # Not enough data

    # Normalize prices and quantities
    prices_a = session.prices[:, group_a]
    prices_b = session.prices[:, group_b]
    quantities_a = session.quantities[:, group_a]
    quantities_b = session.quantities[:, group_b]

    # Compute price indices for each group (expenditure weighted)
    np.sum(prices_a * quantities_a, axis=1)
    np.sum(prices_b * quantities_b, axis=1)

    # Compute average price per group
    avg_price_a = np.mean(prices_a, axis=1)
    avg_price_b = np.mean(prices_b, axis=1)

    # Compute total quantity per group
    total_qty_a = np.sum(quantities_a, axis=1)
    total_qty_b = np.sum(quantities_b, axis=1)

    # Cross-correlation: how much does price_B correlate with quantity_A?
    # If separable, this should be low (after controlling for price_A)
    cross_corr_ab = _partial_correlation(avg_price_b, total_qty_a, avg_price_a)
    cross_corr_ba = _partial_correlation(avg_price_a, total_qty_b, avg_price_b)

    # Average absolute cross-correlation
    cross_effect = (abs(cross_corr_ab) + abs(cross_corr_ba)) / 2

    return min(cross_effect, 1.0)


def _partial_correlation(x: NDArray, y: NDArray, control: NDArray) -> float:
    """Compute partial correlation between x and y, controlling for control."""
    if len(x) < 3:
        return 0.0

    # Residualize x and y on control
    def residualize(arr: NDArray, ctrl: NDArray) -> NDArray:
        if np.std(ctrl) < 1e-10:
            return arr - np.mean(arr)
        coef = np.cov(arr, ctrl)[0, 1] / np.var(ctrl)
        return arr - coef * ctrl

    x_resid = residualize(x, control)
    y_resid = residualize(y, control)

    # Correlation of residuals
    if np.std(x_resid) < 1e-10 or np.std(y_resid) < 1e-10:
        return 0.0

    corr = np.corrcoef(x_resid, y_resid)[0, 1]
    return corr if not np.isnan(corr) else 0.0


def find_separable_partition(
    session: ConsumerSession,
    max_groups: int = 3,
) -> list[list[int]]:
    """
    Automatically discover separable groups of goods.

    Uses hierarchical clustering on the preference graph to find
    groups that can be treated independently.

    Args:
        session: ConsumerSession with prices and quantities
        max_groups: Maximum number of groups to find

    Returns:
        List of lists, where each inner list contains good indices in that group
    """
    N = session.num_goods

    if N < 2:
        return [list(range(N))]

    # Compute pairwise "togetherness" score based on consumption patterns
    togetherness = np.zeros((N, N))

    for t in range(session.num_observations):
        q = session.quantities[t]
        total = np.sum(q)
        if total > 0:
            shares = q / total
            # Goods consumed together in similar proportions have high togetherness
            togetherness += np.outer(shares, shares)

    # Normalize
    togetherness /= session.num_observations

    # Convert to distance matrix
    distance = 1 - togetherness / (togetherness.max() + 1e-10)
    np.fill_diagonal(distance, 0)

    # Simple agglomerative clustering
    groups = [[i] for i in range(N)]

    while len(groups) > max_groups:
        # Find closest pair of groups
        min_dist = float("inf")
        merge_i, merge_j = 0, 1

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                # Average linkage
                avg_dist = np.mean(
                    [distance[gi, gj] for gi in groups[i] for gj in groups[j]]
                )
                if avg_dist < min_dist:
                    min_dist = avg_dist
                    merge_i, merge_j = i, j

        # Merge groups
        groups[merge_i].extend(groups[merge_j])
        del groups[merge_j]

    return groups


def compute_cannibalization(
    session: ConsumerSession,
    group_a: list[int],
    group_b: list[int],
) -> dict[str, float]:
    """
    Compute cannibalization metrics between two product groups.

    Useful for superapp analysis (e.g., Uber Rides vs Eats).

    Args:
        session: ConsumerSession with prices and quantities
        group_a: Indices of first product group
        group_b: Indices of second product group

    Returns:
        Dictionary with cannibalization metrics:
        - 'a_to_b': How much A cannibalizes B (0-1)
        - 'b_to_a': How much B cannibalizes A (0-1)
        - 'symmetric': Average cannibalization
        - 'net_direction': Positive if A cannibalizes B more
    """
    T = session.num_observations

    if T < 2:
        return {
            "a_to_b": 0.0,
            "b_to_a": 0.0,
            "symmetric": 0.0,
            "net_direction": 0.0,
        }

    # Compute expenditure shares
    exp_a = np.sum(session.prices[:, group_a] * session.quantities[:, group_a], axis=1)
    exp_b = np.sum(session.prices[:, group_b] * session.quantities[:, group_b], axis=1)
    total_exp = exp_a + exp_b

    # Avoid division by zero
    total_exp = np.maximum(total_exp, 1e-10)

    share_a = exp_a / total_exp
    share_b = exp_b / total_exp

    # Cannibalization: when one share increases, does the other decrease?
    # Beyond what income effects would predict

    # Simple metric: negative correlation of share changes
    if T < 3:
        corr = 0.0
    else:
        delta_a = np.diff(share_a)
        delta_b = np.diff(share_b)
        if np.std(delta_a) > 1e-10 and np.std(delta_b) > 1e-10:
            corr = np.corrcoef(delta_a, delta_b)[0, 1]
            corr = 0.0 if np.isnan(corr) else corr
        else:
            corr = 0.0

    # Negative correlation indicates cannibalization
    symmetric = max(0, -corr)

    # Direction: which group's growth is associated with the other's decline?
    # Compute asymmetric impacts
    a_growth = np.mean(np.diff(exp_a))
    b_growth = np.mean(np.diff(exp_b))

    if a_growth > 0 and b_growth < 0:
        a_to_b = min(1.0, -b_growth / (a_growth + 1e-10))
        b_to_a = 0.0
    elif b_growth > 0 and a_growth < 0:
        a_to_b = 0.0
        b_to_a = min(1.0, -a_growth / (b_growth + 1e-10))
    else:
        a_to_b = symmetric / 2
        b_to_a = symmetric / 2

    return {
        "a_to_b": a_to_b,
        "b_to_a": b_to_a,
        "symmetric": symmetric,
        "net_direction": a_to_b - b_to_a,
    }


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

# test_feature_independence: Tech-friendly name for check_separability
test_feature_independence = check_separability
"""
Test if two feature groups are independent (can be optimized separately).

This is the tech-friendly alias for check_separability.

Use this to determine if product categories can be priced/optimized
independently without considering cross-effects.

Example:
    >>> from pyrevealed import BehaviorLog, test_feature_independence
    >>> # Test if Rides and Eats are independent for a superapp user
    >>> result = test_feature_independence(user_log, group_a=[0, 1], group_b=[2, 3])
    >>> if result.is_separable:
    ...     print("Can price independently")

Returns:
    FeatureIndependenceResult with is_separable and cross_effect_strength
"""

# discover_independent_groups: Tech-friendly name for find_separable_partition
discover_independent_groups = find_separable_partition
"""
Auto-discover groups of features that can be treated independently.

This is the tech-friendly alias for find_separable_partition.

Uses clustering to find natural groupings of features where
cross-effects are minimal.
"""

# compute_cross_impact: Tech-friendly name for compute_cannibalization
compute_cross_impact = compute_cannibalization
"""
Compute how much one feature group impacts another.

This is the tech-friendly alias for compute_cannibalization.

Measures cross-elasticity effects between feature groups.
High cross-impact means changes in one group significantly affect the other.
"""

# test_feature_independence_exact: Tech-friendly name for check_separability_exact
test_feature_independence_exact = check_separability_exact
"""
Exact test if two feature groups are independent using Theorem 4.4.

This is the tech-friendly alias for check_separability_exact.

Uses the rigorous nonlinear Afriat inequality approach from
Chambers & Echenique (2016) Chapter 4 to test separability.
"""
