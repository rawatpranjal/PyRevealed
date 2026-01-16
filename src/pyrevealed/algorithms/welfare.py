"""Welfare analysis for consumer behavior.

Measures consumer welfare changes from policy or price changes using
compensating variation (CV) and equivalent variation (EV).
Based on Chapters 7.3-7.4 of Chambers & Echenique (2016).

This module provides multiple methods for CV/EV computation:
1. Exact methods using Afriat utility recovery and numerical optimization
2. Vartia path integral approximation
3. Bounds (Laspeyres/Paasche) as fallbacks

Tech-Friendly Names (Primary):
    - compute_compensating_variation(): CV measure (uses exact method)
    - compute_equivalent_variation(): EV measure (uses exact method)
    - compute_cv_exact(): Exact CV via constrained optimization
    - compute_ev_exact(): Exact EV via constrained optimization
    - compute_cv_vartia(): CV via Vartia path integral
    - compute_ev_vartia(): EV via Vartia path integral
    - compute_cv_bounds(): Laspeyres bound for CV
    - compute_ev_bounds(): Paasche bound for EV
    - analyze_welfare_change(): Compare two scenarios
    - recover_expenditure_function(): Proper expenditure function estimation

Economics Names (Legacy Aliases):
    - compute_cv() -> compute_compensating_variation()
    - compute_ev() -> compute_equivalent_variation()

References:
    Chambers & Echenique (2016), Chapter 7.3-7.4
    Vartia (1983), "Efficient Methods of Measuring Welfare Change"
    Afriat (1967), "The Construction of Utility Functions from Expenditure Data"
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, linprog
from scipy.integrate import quad_vec

from pyrevealed.core.result import WelfareResult

if TYPE_CHECKING:
    from pyrevealed.core.session import BehaviorLog


# =============================================================================
# AFRIAT UTILITY RECOVERY FOR WELFARE ANALYSIS
# =============================================================================


def _recover_afriat_utility(
    log: "BehaviorLog",
    tolerance: float = 1e-8,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, bool]:
    """
    Recover utility values and multipliers satisfying Afriat's inequalities.

    Solves the LP:
        min Σ lambda_k
        s.t. U_k <= U_l + lambda_l * p_l @ (x_k - x_l)  for all k, l
             U_k >= 0, lambda_k > epsilon

    Args:
        log: BehaviorLog with prices and quantities
        tolerance: Numerical tolerance for LP solver

    Returns:
        Tuple of (utility_values, lagrange_multipliers, success)
    """
    T = log.num_records
    P = log.cost_vectors
    Q = log.action_vectors

    # Number of variables: U_1, ..., U_T, lambda_1, ..., lambda_T
    n_vars = 2 * T

    # Build inequality constraints: A_ub @ x <= b_ub
    constraints_A = []
    constraints_b = []

    for k in range(T):
        for obs_l in range(T):
            if k == obs_l:
                continue

            # Constraint: U_k - U_l - lambda_l * [p_l @ (x_k - x_l)] <= 0
            row = np.zeros(n_vars)
            row[k] = 1.0  # Coefficient for U_k
            row[obs_l] = -1.0  # Coefficient for U_l

            # Coefficient for lambda_l: -p_l @ (x_k - x_l) = p_l @ (x_l - x_k)
            diff = Q[obs_l] - Q[k]  # x_l - x_k
            lambda_coef = P[obs_l] @ diff  # p_l @ (x_l - x_k)
            row[T + obs_l] = lambda_coef

            constraints_A.append(row)
            constraints_b.append(0.0)

    A_ub = np.array(constraints_A) if constraints_A else np.zeros((0, n_vars))
    b_ub = np.array(constraints_b) if constraints_b else np.zeros(0)

    # Bounds: U_k >= 0, lambda_k > epsilon
    epsilon = 1e-6
    bounds = [(0, None)] * T + [(epsilon, None)] * T

    # Objective: minimize sum of lambdas
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
    except Exception:
        pass

    return None, None, False


def _construct_afriat_utility_function(
    P: NDArray[np.float64],
    Q: NDArray[np.float64],
    U: NDArray[np.float64],
    lambdas: NDArray[np.float64],
) -> Callable[[NDArray[np.float64]], float]:
    """
    Construct the Afriat utility function from recovered parameters.

    u(x) = min_k { U_k + lambda_k * p_k @ (x - x_k) }

    Args:
        P: Price matrix (T x N)
        Q: Quantity matrix (T x N)
        U: Recovered utility values (T,)
        lambdas: Recovered Lagrange multipliers (T,)

    Returns:
        Callable function u(x) that takes a bundle and returns utility
    """
    T = len(U)

    def afriat_utility(x: NDArray[np.float64]) -> float:
        x = np.asarray(x, dtype=np.float64)
        values = np.zeros(T)
        for k in range(T):
            values[k] = U[k] + lambdas[k] * (P[k] @ (x - Q[k]))
        return float(np.min(values))

    return afriat_utility


# =============================================================================
# EXACT CV/EV COMPUTATION
# =============================================================================


def compute_cv_exact(
    baseline_log: "BehaviorLog",
    policy_log: "BehaviorLog",
    tolerance: float = 1e-6,
) -> tuple[float, bool]:
    """
    Compute exact compensating variation using Afriat utility recovery.

    CV = e(p_new, U_baseline) - m_new

    where e(p, u) is the expenditure function and U_baseline is the utility
    achieved at baseline prices. We solve:

        CV = min{p_new @ x : U(x) >= U_baseline} - m_new

    Args:
        baseline_log: BehaviorLog at baseline prices
        policy_log: BehaviorLog at policy prices
        tolerance: Numerical tolerance

    Returns:
        Tuple of (cv_value, success_flag)

    Note:
        If utility recovery fails, falls back to Laspeyres bound.
    """
    # Recover utility from baseline data
    U, lambdas, success = _recover_afriat_utility(baseline_log, tolerance)

    if not success or U is None or lambdas is None:
        # Fall back to Laspeyres bound
        cv_bound = compute_cv_bounds(baseline_log, policy_log)
        return cv_bound, False

    # Construct utility function
    utility_fn = _construct_afriat_utility_function(
        baseline_log.cost_vectors,
        baseline_log.action_vectors,
        U,
        lambdas,
    )

    # Get baseline and policy quantities
    baseline_quantities = np.mean(baseline_log.action_vectors, axis=0)
    policy_prices = np.mean(policy_log.cost_vectors, axis=0)
    policy_quantities = np.mean(policy_log.action_vectors, axis=0)

    # Baseline utility level
    U_baseline = utility_fn(baseline_quantities)

    # Policy expenditure
    m_policy = policy_prices @ policy_quantities

    N = len(baseline_quantities)

    # Solve: min{p_new @ x : U(x) >= U_baseline, x >= 0}
    def objective(x: NDArray[np.float64]) -> float:
        return float(policy_prices @ x)

    def utility_constraint(x: NDArray[np.float64]) -> float:
        return utility_fn(x) - U_baseline

    # Initial guess: policy quantities
    x0 = policy_quantities.copy()

    # Bounds: x >= 0
    bounds = [(0, None)] * N

    # Constraints
    constraints = [{"type": "ineq", "fun": utility_constraint}]

    try:
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": tolerance},
        )

        if result.success:
            # e(p_new, U_baseline) - actual expenditure at new prices
            expenditure_at_baseline_utility = result.fun
            cv = expenditure_at_baseline_utility - m_policy
            return cv, True
    except Exception:
        pass

    # Fall back to Laspeyres bound
    cv_bound = compute_cv_bounds(baseline_log, policy_log)
    return cv_bound, False


def compute_ev_exact(
    baseline_log: "BehaviorLog",
    policy_log: "BehaviorLog",
    tolerance: float = 1e-6,
) -> tuple[float, bool]:
    """
    Compute exact equivalent variation using Afriat utility recovery.

    EV = m_baseline - e(p_baseline, U_policy)

    where e(p, u) is the expenditure function and U_policy is the utility
    achieved at policy prices. We solve:

        EV = m_baseline - min{p_baseline @ x : U(x) >= U_policy}

    Args:
        baseline_log: BehaviorLog at baseline prices
        policy_log: BehaviorLog at policy prices
        tolerance: Numerical tolerance

    Returns:
        Tuple of (ev_value, success_flag)

    Note:
        If utility recovery fails, falls back to Paasche bound.
    """
    # Recover utility from combined data for better coverage
    combined_P = np.vstack([baseline_log.cost_vectors, policy_log.cost_vectors])
    combined_Q = np.vstack([baseline_log.action_vectors, policy_log.action_vectors])

    # Create combined log for utility recovery
    from pyrevealed.core.session import BehaviorLog

    try:
        combined_log = BehaviorLog(cost_vectors=combined_P, action_vectors=combined_Q)
        U, lambdas, success = _recover_afriat_utility(combined_log, tolerance)
    except Exception:
        success = False
        U, lambdas = None, None

    if not success or U is None or lambdas is None:
        # Fall back to Paasche bound
        ev_bound = compute_ev_bounds(baseline_log, policy_log)
        return ev_bound, False

    # Construct utility function
    utility_fn = _construct_afriat_utility_function(combined_P, combined_Q, U, lambdas)

    # Get baseline and policy quantities
    baseline_prices = np.mean(baseline_log.cost_vectors, axis=0)
    baseline_quantities = np.mean(baseline_log.action_vectors, axis=0)
    policy_quantities = np.mean(policy_log.action_vectors, axis=0)

    # Policy utility level
    U_policy = utility_fn(policy_quantities)

    # Baseline expenditure
    m_baseline = baseline_prices @ baseline_quantities

    N = len(baseline_quantities)

    # Solve: min{p_baseline @ x : U(x) >= U_policy, x >= 0}
    def objective(x: NDArray[np.float64]) -> float:
        return float(baseline_prices @ x)

    def utility_constraint(x: NDArray[np.float64]) -> float:
        return utility_fn(x) - U_policy

    # Initial guess: baseline quantities
    x0 = baseline_quantities.copy()

    # Bounds: x >= 0
    bounds = [(0, None)] * N

    # Constraints
    constraints = [{"type": "ineq", "fun": utility_constraint}]

    try:
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": tolerance},
        )

        if result.success:
            # baseline expenditure - e(p_baseline, U_policy)
            expenditure_at_policy_utility = result.fun
            ev = m_baseline - expenditure_at_policy_utility
            return ev, True
    except Exception:
        pass

    # Fall back to Paasche bound
    ev_bound = compute_ev_bounds(baseline_log, policy_log)
    return ev_bound, False


# =============================================================================
# VARTIA PATH INTEGRAL METHODS
# =============================================================================


def compute_cv_vartia(
    baseline_log: "BehaviorLog",
    policy_log: "BehaviorLog",
    n_steps: int = 100,
) -> float:
    """
    Compute CV using Vartia (1983) path integral approximation.

    CV = ∫ x(p, U_0) · dp along path from p0 to p1

    This integrates Hicksian demand along a price path, which gives
    exact CV for smooth preferences. We approximate Hicksian demand
    using Stone-Geary functional form estimation.

    Args:
        baseline_log: BehaviorLog at baseline prices
        policy_log: BehaviorLog at policy prices
        n_steps: Number of integration steps

    Returns:
        Compensating variation via path integral

    References:
        Vartia (1983), "Efficient Methods of Measuring Welfare Change"
    """
    # Get average prices and quantities
    p0 = np.mean(baseline_log.cost_vectors, axis=0)
    x0 = np.mean(baseline_log.action_vectors, axis=0)
    p1 = np.mean(policy_log.cost_vectors, axis=0)

    N = len(p0)

    # Estimate Stone-Geary parameters from baseline data
    # Demand: x_i = gamma_i + beta_i * (m - p @ gamma) / p_i
    # where m = p @ x is expenditure, gamma is subsistence, beta is share

    # Simple estimation: assume gamma = 0 (Cobb-Douglas approximation)
    m0 = p0 @ x0
    budget_shares = (p0 * x0) / m0
    beta = budget_shares  # Estimated budget shares

    # Hicksian demand approximation at utility level U0
    # For Cobb-Douglas: h_i(p, U) = beta_i * U^(1/sum(beta)) * prod(p_j^(-beta_j/sum(beta)))
    # Simplified: h_i(p, U) ≈ x0_i * (p0_i / p_i)^(1) for i-th good (substitution effect only)

    def hicksian_demand(p: NDArray[np.float64]) -> NDArray[np.float64]:
        """Approximate Hicksian demand at baseline utility."""
        # Use Slutsky-compensated demand approximation
        # h(p, U0) ≈ x0 + S @ (p - p0) where S is Slutsky matrix
        # For Cobb-Douglas, own-price effect dominates

        # Simple approximation: constant utility means adjusting for price change
        # to maintain same utility level
        h = np.zeros(N)
        for i in range(N):
            # Own-price Slutsky substitution effect (always negative for normal goods)
            # h_i = x0_i * (p0_i / p_i)^sigma where sigma ≈ 1 for Cobb-Douglas
            if p[i] > 1e-10:
                h[i] = x0[i] * (p0[i] / p[i]) ** beta[i]
            else:
                h[i] = x0[i]
        return h

    # Integrate along linear price path from p0 to p1
    # CV = ∫_0^1 h(p(t), U0) @ dp/dt dt where p(t) = p0 + t*(p1-p0)
    dp = p1 - p0

    def integrand(t: float) -> NDArray[np.float64]:
        p_t = p0 + t * dp
        h_t = hicksian_demand(p_t)
        return h_t * dp

    # Numerical integration using trapezoidal rule
    cv = 0.0
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t0 = i * dt
        t1 = (i + 1) * dt
        cv += 0.5 * (np.sum(integrand(t0)) + np.sum(integrand(t1))) * dt

    return cv


def compute_ev_vartia(
    baseline_log: "BehaviorLog",
    policy_log: "BehaviorLog",
    n_steps: int = 100,
) -> float:
    """
    Compute EV using Vartia (1983) path integral approximation.

    EV = -∫ x(p, U_1) · dp along path from p0 to p1

    This integrates Hicksian demand at policy utility along a price path.

    Args:
        baseline_log: BehaviorLog at baseline prices
        policy_log: BehaviorLog at policy prices
        n_steps: Number of integration steps

    Returns:
        Equivalent variation via path integral

    References:
        Vartia (1983), "Efficient Methods of Measuring Welfare Change"
    """
    # Get average prices and quantities
    p0 = np.mean(baseline_log.cost_vectors, axis=0)
    p1 = np.mean(policy_log.cost_vectors, axis=0)
    x1 = np.mean(policy_log.action_vectors, axis=0)

    N = len(p0)

    # Estimate from policy data
    m1 = p1 @ x1
    budget_shares = (p1 * x1) / m1
    beta = budget_shares

    def hicksian_demand_at_u1(p: NDArray[np.float64]) -> NDArray[np.float64]:
        """Approximate Hicksian demand at policy utility."""
        h = np.zeros(N)
        for i in range(N):
            if p[i] > 1e-10:
                h[i] = x1[i] * (p1[i] / p[i]) ** beta[i]
            else:
                h[i] = x1[i]
        return h

    # Integrate along linear price path from p0 to p1
    dp = p1 - p0

    def integrand(t: float) -> NDArray[np.float64]:
        p_t = p0 + t * dp
        h_t = hicksian_demand_at_u1(p_t)
        return h_t * dp

    # Numerical integration
    ev = 0.0
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t0 = i * dt
        t1 = (i + 1) * dt
        ev += 0.5 * (np.sum(integrand(t0)) + np.sum(integrand(t1))) * dt

    # EV is negative of the integral (going from p0 to p1)
    return -ev


# =============================================================================
# BOUNDS (LASPEYRES/PAASCHE)
# =============================================================================


def compute_cv_bounds(
    baseline_log: "BehaviorLog",
    policy_log: "BehaviorLog",
) -> float:
    """
    Compute Laspeyres bound for compensating variation.

    CV_bound = p1 @ x0 - p1 @ x1

    This is an upper bound for CV when welfare improved (prices fell)
    and a lower bound when welfare worsened (prices rose).

    Args:
        baseline_log: BehaviorLog at baseline prices
        policy_log: BehaviorLog at policy prices

    Returns:
        Laspeyres bound for CV
    """
    baseline_quantities = np.mean(baseline_log.action_vectors, axis=0)
    policy_prices = np.mean(policy_log.cost_vectors, axis=0)
    policy_quantities = np.mean(policy_log.action_vectors, axis=0)

    cost_baseline_at_policy = np.dot(policy_prices, baseline_quantities)
    e_policy = np.dot(policy_prices, policy_quantities)

    return cost_baseline_at_policy - e_policy


def compute_ev_bounds(
    baseline_log: "BehaviorLog",
    policy_log: "BehaviorLog",
) -> float:
    """
    Compute Paasche bound for equivalent variation.

    EV_bound = p0 @ x1 - p0 @ x0

    This is a lower bound for EV when welfare improved
    and an upper bound when welfare worsened.

    Args:
        baseline_log: BehaviorLog at baseline prices
        policy_log: BehaviorLog at policy prices

    Returns:
        Paasche bound for EV
    """
    baseline_prices = np.mean(baseline_log.cost_vectors, axis=0)
    baseline_quantities = np.mean(baseline_log.action_vectors, axis=0)
    policy_quantities = np.mean(policy_log.action_vectors, axis=0)

    cost_policy_at_baseline = np.dot(baseline_prices, policy_quantities)
    e_baseline = np.dot(baseline_prices, baseline_quantities)

    return cost_policy_at_baseline - e_baseline


# =============================================================================
# EXPENDITURE FUNCTION RECOVERY
# =============================================================================


def recover_expenditure_function(
    log: "BehaviorLog",
    tolerance: float = 1e-8,
) -> dict:
    """
    Recover the expenditure function e(p, u) from revealed preference data.

    Uses Afriat's approach to construct a piecewise linear expenditure function
    that is consistent with the observed behavior.

    For any price vector p and utility level u:
        e(p, u) = min{p @ x : U(x) >= u}

    where U(x) is the Afriat utility function.

    Args:
        log: BehaviorLog with prices and quantities
        tolerance: Numerical tolerance

    Returns:
        Dictionary containing:
        - 'success': Whether utility recovery succeeded
        - 'utility_function': Callable that evaluates U(x)
        - 'expenditure_function': Callable that evaluates e(p, u)
        - 'utility_values': Recovered U_k values
        - 'lagrange_multipliers': Recovered lambda_k values
        - 'observation_utilities': Utility at each observed bundle
        - 'observation_expenditures': Expenditure at each observation
    """
    T = log.num_records
    P = log.cost_vectors
    Q = log.action_vectors

    # Recover Afriat utility
    U, lambdas, success = _recover_afriat_utility(log, tolerance)

    if not success or U is None or lambdas is None:
        # Return fallback with simple utility approximation
        expenditures = log.total_spend
        utilities = np.sum(np.log(Q + 1e-10), axis=1)

        return {
            "success": False,
            "utility_function": None,
            "expenditure_function": None,
            "utility_values": None,
            "lagrange_multipliers": None,
            "observation_utilities": utilities,
            "observation_expenditures": expenditures,
        }

    # Construct utility function
    utility_fn = _construct_afriat_utility_function(P, Q, U, lambdas)

    # Construct expenditure function
    def expenditure_function(
        p: NDArray[np.float64], u: float
    ) -> tuple[float, NDArray[np.float64] | None]:
        """
        Compute e(p, u) = min{p @ x : U(x) >= u, x >= 0}.

        Args:
            p: Price vector
            u: Target utility level

        Returns:
            Tuple of (expenditure, optimal_bundle) or (inf, None) if infeasible
        """
        p = np.asarray(p, dtype=np.float64)
        N = len(p)

        def objective(x: NDArray[np.float64]) -> float:
            return float(p @ x)

        def utility_constraint(x: NDArray[np.float64]) -> float:
            return utility_fn(x) - u

        # Initial guess: use observed bundle with similar utility
        utilities_obs = np.array([utility_fn(Q[k]) for k in range(T)])
        closest_idx = np.argmin(np.abs(utilities_obs - u))
        x0 = Q[closest_idx].copy()

        bounds = [(0, None)] * N
        constraints = [{"type": "ineq", "fun": utility_constraint}]

        try:
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": tolerance},
            )
            if result.success:
                return result.fun, result.x
        except Exception:
            pass

        return float("inf"), None

    # Compute utilities at each observation
    observation_utilities = np.array([utility_fn(Q[k]) for k in range(T)])
    observation_expenditures = log.total_spend

    return {
        "success": True,
        "utility_function": utility_fn,
        "expenditure_function": expenditure_function,
        "utility_values": U,
        "lagrange_multipliers": lambdas,
        "observation_utilities": observation_utilities,
        "observation_expenditures": observation_expenditures,
    }


# =============================================================================
# MAIN WELFARE ANALYSIS FUNCTIONS
# =============================================================================


def analyze_welfare_change(
    baseline_log: "BehaviorLog",
    policy_log: "BehaviorLog",
    tolerance: float = 1e-6,
    method: str = "exact",
) -> WelfareResult:
    """
    Analyze welfare change between two scenarios.

    Computes both compensating variation (CV) and equivalent variation (EV)
    to measure the welfare impact of a policy or price change.

    CV: Amount of money to give consumer after the change to restore
        original utility level.
    EV: Amount of money equivalent to the utility change at original prices.

    This function uses theoretically rigorous methods:
    - "exact": Uses Afriat utility recovery and constrained optimization
    - "vartia": Uses Vartia (1983) path integral approximation
    - "bounds": Uses Laspeyres/Paasche bounds (fastest but least accurate)

    Args:
        baseline_log: BehaviorLog at baseline (pre-policy) prices
        policy_log: BehaviorLog at policy (post-change) prices
        tolerance: Numerical tolerance
        method: Computation method - "exact", "vartia", or "bounds"

    Returns:
        WelfareResult with CV, EV, and welfare direction

    Example:
        >>> from pyrevealed import BehaviorLog, analyze_welfare_change
        >>> result = analyze_welfare_change(baseline_data, policy_data)
        >>> print(f"Welfare direction: {result.welfare_direction}")
        >>> print(f"CV: ${result.compensating_variation:.2f}")
        >>> print(f"EV: ${result.equivalent_variation:.2f}")

    References:
        Chambers & Echenique (2016), Chapter 7.3-7.4
        Afriat (1967), "The Construction of Utility Functions"
        Vartia (1983), "Efficient Methods of Measuring Welfare Change"
    """
    start_time = time.perf_counter()

    # Compute baseline and policy utilities using Afriat recovery
    baseline_expenditure = np.mean(baseline_log.total_spend)
    policy_expenditure = np.mean(policy_log.total_spend)

    # Try to recover utility from combined data for accurate utility comparison
    combined_P = np.vstack([baseline_log.cost_vectors, policy_log.cost_vectors])
    combined_Q = np.vstack([baseline_log.action_vectors, policy_log.action_vectors])

    try:
        from pyrevealed.core.session import BehaviorLog

        combined_log = BehaviorLog(cost_vectors=combined_P, action_vectors=combined_Q)
        U, lambdas, success = _recover_afriat_utility(combined_log, tolerance)
    except Exception:
        success = False
        U, lambdas = None, None

    if success and U is not None and lambdas is not None:
        utility_fn = _construct_afriat_utility_function(combined_P, combined_Q, U, lambdas)
        baseline_quantities = np.mean(baseline_log.action_vectors, axis=0)
        policy_quantities = np.mean(policy_log.action_vectors, axis=0)
        baseline_utility = utility_fn(baseline_quantities)
        policy_utility = utility_fn(policy_quantities)
    else:
        # Fall back to simple utility index
        baseline_utility = _estimate_utility_index(baseline_log)
        policy_utility = _estimate_utility_index(policy_log)

    # Compute CV and EV using the specified method
    if method == "exact":
        cv, cv_success = compute_cv_exact(baseline_log, policy_log, tolerance)
        ev, ev_success = compute_ev_exact(baseline_log, policy_log, tolerance)
        if not cv_success or not ev_success:
            # Fall back to Vartia if exact method fails
            cv = compute_cv_vartia(baseline_log, policy_log)
            ev = compute_ev_vartia(baseline_log, policy_log)
    elif method == "vartia":
        cv = compute_cv_vartia(baseline_log, policy_log)
        ev = compute_ev_vartia(baseline_log, policy_log)
    elif method == "bounds":
        cv = compute_cv_bounds(baseline_log, policy_log)
        ev = compute_ev_bounds(baseline_log, policy_log)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'exact', 'vartia', or 'bounds'.")

    # Determine welfare direction based on utility comparison
    if policy_utility > baseline_utility + tolerance:
        welfare_direction = "improved"
    elif policy_utility < baseline_utility - tolerance:
        welfare_direction = "worsened"
    else:
        welfare_direction = "ambiguous"

    # Compute Hicksian consumer surplus as average of CV and EV
    hicksian_surplus = (cv + ev) / 2

    computation_time = (time.perf_counter() - start_time) * 1000

    return WelfareResult(
        compensating_variation=cv,
        equivalent_variation=ev,
        welfare_direction=welfare_direction,
        baseline_utility=baseline_utility,
        policy_utility=policy_utility,
        baseline_expenditure=baseline_expenditure,
        policy_expenditure=policy_expenditure,
        hicksian_surplus=hicksian_surplus,
        computation_time_ms=computation_time,
    )


def compute_compensating_variation(
    baseline_log: "BehaviorLog",
    policy_log: "BehaviorLog",
    target_utility: float | None = None,
    tolerance: float = 1e-6,
    method: str = "exact",
) -> float:
    """
    Compute compensating variation (CV).

    CV measures how much additional money the consumer would need
    at the new prices to achieve the old utility level.

    CV > 0: Consumer needs compensation (welfare worsened)
    CV < 0: Consumer can afford to pay (welfare improved)

    This function uses theoretically rigorous methods by default:
    - "exact": Uses Afriat utility recovery and solves min{p_new @ x : U(x) >= U_baseline}
    - "vartia": Uses Vartia (1983) path integral approximation
    - "bounds": Uses Laspeyres bound (fastest but least accurate)

    Args:
        baseline_log: BehaviorLog at baseline prices
        policy_log: BehaviorLog at policy prices
        target_utility: Deprecated parameter (kept for backward compatibility)
        tolerance: Numerical tolerance
        method: Computation method - "exact", "vartia", or "bounds"

    Returns:
        Compensating variation amount

    References:
        Chambers & Echenique (2016), Chapter 7.3-7.4
        Afriat (1967), "The Construction of Utility Functions"
    """
    if method == "exact":
        cv, success = compute_cv_exact(baseline_log, policy_log, tolerance)
        if not success:
            # Fall back to Vartia if exact fails
            cv = compute_cv_vartia(baseline_log, policy_log)
        return cv
    elif method == "vartia":
        return compute_cv_vartia(baseline_log, policy_log)
    elif method == "bounds":
        return compute_cv_bounds(baseline_log, policy_log)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'exact', 'vartia', or 'bounds'.")


def compute_equivalent_variation(
    baseline_log: "BehaviorLog",
    policy_log: "BehaviorLog",
    target_utility: float | None = None,
    tolerance: float = 1e-6,
    method: str = "exact",
) -> float:
    """
    Compute equivalent variation (EV).

    EV measures how much money at baseline prices would be equivalent
    to the utility change caused by the policy.

    EV > 0: Policy improved welfare by this amount
    EV < 0: Policy worsened welfare by this amount

    This function uses theoretically rigorous methods by default:
    - "exact": Uses Afriat utility recovery and solves for expenditure function
    - "vartia": Uses Vartia (1983) path integral approximation
    - "bounds": Uses Paasche bound (fastest but least accurate)

    Args:
        baseline_log: BehaviorLog at baseline prices
        policy_log: BehaviorLog at policy prices
        target_utility: Deprecated parameter (kept for backward compatibility)
        tolerance: Numerical tolerance
        method: Computation method - "exact", "vartia", or "bounds"

    Returns:
        Equivalent variation amount

    References:
        Chambers & Echenique (2016), Chapter 7.3-7.4
        Afriat (1967), "The Construction of Utility Functions"
    """
    if method == "exact":
        ev, success = compute_ev_exact(baseline_log, policy_log, tolerance)
        if not success:
            # Fall back to Vartia if exact fails
            ev = compute_ev_vartia(baseline_log, policy_log)
        return ev
    elif method == "vartia":
        return compute_ev_vartia(baseline_log, policy_log)
    elif method == "bounds":
        return compute_ev_bounds(baseline_log, policy_log)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'exact', 'vartia', or 'bounds'.")


def recover_cost_function(
    log: "BehaviorLog",
    target_utility: float | None = None,
    tolerance: float = 1e-8,
) -> dict:
    """
    Estimate the expenditure (cost) function from revealed preference data.

    The expenditure function e(p, u) gives the minimum cost to achieve
    utility level u at prices p.

    This is a wrapper around recover_expenditure_function() that provides
    a backward-compatible interface while using the rigorous Afriat approach.

    Args:
        log: BehaviorLog with prices and quantities
        target_utility: Optional utility level to estimate cost for (deprecated)
        tolerance: Numerical tolerance for LP solver

    Returns:
        Dictionary with cost function estimates including:
        - 'success': Whether Afriat utility recovery succeeded
        - 'utility_function': Callable to evaluate utility at any bundle
        - 'expenditure_function': Callable to compute e(p, u)
        - 'observation_utilities': Utility at each observed bundle
        - 'observation_expenditures': Expenditure at each observation
    """
    return recover_expenditure_function(log, tolerance)


def compute_consumer_surplus(
    log: "BehaviorLog",
    good_index: int,
    price_change: float,
) -> float:
    """
    Compute consumer surplus change for a price change in one good.

    Uses the area under the demand curve approximation.

    Args:
        log: BehaviorLog with prices and quantities
        good_index: Index of the good with price change
        price_change: Change in price (positive = increase)

    Returns:
        Consumer surplus change (negative if price increased)
    """
    # Get average quantity demanded
    avg_quantity = np.mean(log.action_vectors[:, good_index])

    # Simple linear approximation: CS change ≈ -q * dp
    cs_change = -avg_quantity * price_change

    return cs_change


def compute_deadweight_loss(
    baseline_log: "BehaviorLog",
    policy_log: "BehaviorLog",
    method: str = "exact",
) -> float:
    """
    Estimate deadweight loss from a policy intervention.

    Deadweight loss is the welfare loss not captured by transfers.
    It represents the economic inefficiency caused by market distortions.

    For a welfare-improving policy: DWL = CV - EV (EV > CV implies inefficiency)
    For a welfare-worsening policy: DWL = EV - CV (CV > EV implies inefficiency)

    The Harberger approximation uses: DWL ≈ |CV - EV| / 2

    This function uses theoretically rigorous CV/EV computation methods.

    Args:
        baseline_log: BehaviorLog at efficient prices
        policy_log: BehaviorLog at distorted prices
        method: CV/EV computation method - "exact", "vartia", or "bounds"

    Returns:
        Estimated deadweight loss (always non-negative)

    References:
        Harberger (1964), "The Measurement of Waste"
        Chambers & Echenique (2016), Chapter 7.4
    """
    # Compute CV and EV using the specified method
    cv = compute_compensating_variation(baseline_log, policy_log, method=method)
    ev = compute_equivalent_variation(baseline_log, policy_log, method=method)

    # Deadweight loss is the difference between CV and EV
    # For small changes, DWL ≈ |CV - EV| / 2 (Harberger triangle approximation)
    # For larger changes, the exact DWL depends on the curvature of indifference curves
    dwl = abs(cv - ev) / 2

    return dwl


def _estimate_utility_index(log: "BehaviorLog") -> float:
    """
    Estimate a utility index from behavior log.

    Uses a simple additive utility approximation based on
    expenditure-weighted quantities.
    """
    total_expenditure = np.sum(log.total_spend)
    if total_expenditure < 1e-10:
        return 0.0

    # Use expenditure share weighted average of log quantities
    Q = log.action_vectors
    shares = log.total_spend / total_expenditure

    # Cobb-Douglas style utility index
    log_quantities = np.log(Q + 1e-10)
    weighted_log_q = np.sum(shares[:, np.newaxis] * log_quantities, axis=0)
    utility_index = np.sum(weighted_log_q)

    return utility_index


def _estimate_observation_utility(quantities: NDArray[np.float64]) -> float:
    """
    Estimate utility for a single observation.

    Uses log-linear (Cobb-Douglas style) utility.
    """
    log_q = np.log(quantities + 1e-10)
    return np.sum(log_q)


# =============================================================================
# LEGACY ALIASES
# =============================================================================

compute_cv = compute_compensating_variation
"""Legacy alias: use compute_compensating_variation instead."""

compute_ev = compute_equivalent_variation
"""Legacy alias: use compute_equivalent_variation instead."""

compute_welfare_change = analyze_welfare_change
"""Legacy alias: use analyze_welfare_change instead."""
