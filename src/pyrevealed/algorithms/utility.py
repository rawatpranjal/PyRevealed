"""Utility recovery via linear programming (Afriat's inequalities)."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import UtilityRecoveryResult
from pyrevealed.core.exceptions import OptimizationError


def recover_utility(
    session: ConsumerSession,
    tolerance: float = 1e-8,
) -> UtilityRecoveryResult:
    """
    Recover utility values satisfying Afriat's inequalities using LP.

    If the data satisfies GARP, we can recover utility values U_k and
    Lagrange multipliers (marginal utility of money) lambda_k such that
    Afriat's inequalities hold:

        U_k <= U_l + lambda_l * p_l @ (x_k - x_l)  for all k, l

    This is solved as a linear programming feasibility problem:
    - Variables: U_1, ..., U_T, lambda_1, ..., lambda_T (2T variables)
    - Constraints: Afriat inequalities for all k, l pairs
    - Objective: Minimize sum of variables (for a centered solution)

    The recovered utility function is piecewise linear and concave.

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Numerical tolerance for LP solver

    Returns:
        UtilityRecoveryResult with utility values and multipliers

    Example:
        >>> import numpy as np
        >>> from pyrevealed import ConsumerSession, recover_utility
        >>> prices = np.array([[1.0, 2.0], [2.0, 1.0]])
        >>> quantities = np.array([[4.0, 1.0], [1.0, 4.0]])
        >>> session = ConsumerSession(prices=prices, quantities=quantities)
        >>> result = recover_utility(session)
        >>> if result.success:
        ...     print(f"Utility values: {result.utility_values}")
    """
    start_time = time.perf_counter()

    T = session.num_observations
    P = session.prices
    Q = session.quantities

    # Number of variables: U_1, ..., U_T, lambda_1, ..., lambda_T
    n_vars = 2 * T

    # Build inequality constraints: A_ub @ x <= b_ub
    # Afriat inequality: U_k - U_l - lambda_l * p_l @ (x_k - x_l) <= 0
    # Rearranged for LP format

    constraints_A = []
    constraints_b = []

    for k in range(T):
        for obs_l in range(T):
            if k == obs_l:
                continue

            # Constraint: U_k - U_l - lambda_l * [p_l @ (x_k - x_l)] <= 0
            # Variables are [U_0, ..., U_{T-1}, lambda_0, ..., lambda_{T-1}]

            row = np.zeros(n_vars)
            row[k] = 1.0  # Coefficient for U_k
            row[obs_l] = -1.0  # Coefficient for U_l

            # Coefficient for lambda_l: -p_l @ (x_k - x_l) = p_l @ (x_l - x_k)
            diff = Q[obs_l] - Q[k]  # x_l - x_k
            lambda_coef = P[obs_l] @ diff  # p_l @ (x_l - x_k)
            row[T + obs_l] = lambda_coef

            constraints_A.append(row)
            constraints_b.append(0.0)

    A_ub = np.array(constraints_A)
    b_ub = np.array(constraints_b)

    # Bounds:
    # - U_k >= 0 (utility can be normalized to be non-negative)
    # - lambda_k > 0 (strictly positive marginal utility of money)
    epsilon = 1e-6
    bounds = [(0, None)] * T + [(epsilon, None)] * T  # U >= 0, lambda > 0

    # Objective: minimize sum of lambdas (utility values don't need minimizing)
    # This finds a solution with minimal marginal utilities of money
    c = np.zeros(n_vars)
    c[T:] = 1.0  # Only minimize lambdas

    # Solve LP using HiGHS solver
    try:
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
            options={"presolve": True, "tol": tolerance},
        )
    except Exception as e:
        computation_time = (time.perf_counter() - start_time) * 1000
        return UtilityRecoveryResult(
            success=False,
            utility_values=None,
            lagrange_multipliers=None,
            lp_status=f"LP solver error: {str(e)}",
            residuals=None,
            computation_time_ms=computation_time,
        )

    computation_time = (time.perf_counter() - start_time) * 1000

    if result.success:
        U = result.x[:T]
        lambdas = result.x[T:]

        # Compute residuals for verification
        residuals = _compute_afriat_residuals(U, lambdas, P, Q)

        return UtilityRecoveryResult(
            success=True,
            utility_values=U,
            lagrange_multipliers=lambdas,
            lp_status=result.message,
            residuals=residuals,
            computation_time_ms=computation_time,
        )
    else:
        return UtilityRecoveryResult(
            success=False,
            utility_values=None,
            lagrange_multipliers=None,
            lp_status=result.message,
            residuals=None,
            computation_time_ms=computation_time,
        )


def _compute_afriat_residuals(
    U: NDArray[np.float64],
    lambdas: NDArray[np.float64],
    P: NDArray[np.float64],
    Q: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute residuals of Afriat inequalities for verification.

    Residual[k,l] = U_l + lambda_l * p_l @ (x_k - x_l) - U_k

    If constraints are satisfied, all residuals should be >= 0.

    Args:
        U: Utility values
        lambdas: Lagrange multipliers
        P: Price matrix
        Q: Quantity matrix

    Returns:
        T x T matrix of residuals
    """
    T = len(U)
    residuals = np.zeros((T, T))

    for k in range(T):
        for obs_l in range(T):
            # Afriat inequality: U_k <= U_l + lambda_l * p_l @ (x_k - x_l)
            # Residual = RHS - LHS = U_l + lambda_l * p_l @ (x_k - x_l) - U_k
            diff = Q[k] - Q[obs_l]
            residuals[k, obs_l] = U[obs_l] + lambdas[obs_l] * (P[obs_l] @ diff) - U[k]

    return residuals


def construct_afriat_utility(
    session: ConsumerSession,
    utility_result: UtilityRecoveryResult,
) -> Callable[[NDArray[np.float64]], float]:
    """
    Construct the Afriat utility function from recovered parameters.

    The Afriat utility function is piecewise linear and concave:

        u(x) = min_k { U_k + lambda_k * p_k @ (x - x_k) }

    This function rationalizes the observed data: for each observation k,
    the bundle x_k maximizes u(x) subject to the budget constraint p_k @ x <= p_k @ x_k.

    Args:
        session: ConsumerSession with the choice data
        utility_result: Result from recover_utility

    Returns:
        Callable function u(x) that takes a bundle and returns utility

    Raises:
        ValueError: If utility recovery was not successful

    Example:
        >>> result = recover_utility(session)
        >>> if result.success:
        ...     u = construct_afriat_utility(session, result)
        ...     # Evaluate utility at a new bundle
        ...     new_bundle = np.array([2.0, 3.0])
        ...     utility = u(new_bundle)
    """
    if not utility_result.success:
        raise OptimizationError(
            "Cannot construct utility from failed recovery. "
            f"LP status: {utility_result.lp_status}. "
            "Hint: Check data consistency with compute_integrity_score() first. "
            "If integrity is low, the behavior may be too inconsistent for utility recovery."
        )

    if (
        utility_result.utility_values is None
        or utility_result.lagrange_multipliers is None
    ):
        raise OptimizationError(
            "Utility values or multipliers are None despite successful LP. "
            "This may indicate a numerical issue. Try adjusting tolerance."
        )

    U = utility_result.utility_values
    lambdas = utility_result.lagrange_multipliers
    P = session.prices
    Q = session.quantities
    T = len(U)

    def afriat_utility(x: NDArray[np.float64]) -> float:
        """
        Evaluate Afriat utility at bundle x.

        u(x) = min_k { U_k + lambda_k * p_k @ (x - x_k) }

        Args:
            x: Bundle (array of quantities for each good)

        Returns:
            Utility value
        """
        x = np.asarray(x, dtype=np.float64)
        values = []

        for k in range(T):
            val = U[k] + lambdas[k] * (P[k] @ (x - Q[k]))
            values.append(val)

        return min(values)

    return afriat_utility


def predict_demand(
    session: ConsumerSession,
    utility_result: UtilityRecoveryResult,
    new_prices: NDArray[np.float64],
    budget: float,
    n_goods: int | None = None,
) -> NDArray[np.float64] | None:
    """
    Predict demand at new prices using the recovered utility function.

    Uses the Afriat utility function to find the utility-maximizing bundle
    at the given prices and budget.

    This is a simple grid search implementation suitable for small problems.
    For production use with many goods, consider using scipy.optimize.

    Args:
        session: ConsumerSession with choice data
        utility_result: Result from recover_utility
        new_prices: Price vector for the new budget set
        budget: Total budget available
        n_goods: Number of goods (inferred from prices if not provided)

    Returns:
        Predicted demand bundle, or None if optimization fails
    """
    if not utility_result.success:
        return None

    u = construct_afriat_utility(session, utility_result)
    new_prices = np.asarray(new_prices, dtype=np.float64)

    if n_goods is None:
        n_goods = len(new_prices)

    if n_goods > 3:
        # For more than 3 goods, use scipy optimization
        from scipy.optimize import minimize

        def neg_utility(x: NDArray[np.float64]) -> float:
            return -u(x)

        # Budget constraint: p @ x <= budget
        constraints = {"type": "ineq", "fun": lambda x: budget - new_prices @ x}
        bounds = [(0, budget / p) for p in new_prices]

        # Start from equal allocation
        x0 = np.full(n_goods, budget / (n_goods * np.mean(new_prices)))

        result = minimize(
            neg_utility,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            return result.x
        return None

    else:
        # For 2-3 goods, use grid search for simplicity
        best_x = None
        best_utility = float("-inf")

        # Create grid
        n_points = 50
        max_quantities = budget / new_prices

        if n_goods == 2:
            for q0 in np.linspace(0, max_quantities[0], n_points):
                remaining = budget - new_prices[0] * q0
                if remaining >= 0:
                    q1 = remaining / new_prices[1]
                    x = np.array([q0, q1])
                    util = u(x)
                    if util > best_utility:
                        best_utility = util
                        best_x = x

        return best_x


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

# fit_latent_values: Tech-friendly name for recover_utility
fit_latent_values = recover_utility
"""
Fit latent preference values from user behavior.

This is the tech-friendly alias for recover_utility.

Extracts latent preference values that can be used as:
- Features for ML models
- Inputs to counterfactual simulations
- User embeddings for personalization

Example:
    >>> from pyrevealed import BehaviorLog, fit_latent_values
    >>> result = fit_latent_values(user_log)
    >>> if result.converged:
    ...     features = result.latent_values  # Use as ML features

Returns:
    LatentValueResult with latent_values array
"""

# build_value_function: Tech-friendly name for construct_afriat_utility
build_value_function = construct_afriat_utility
"""
Build a callable value function from fitted latent values.

This is the tech-friendly alias for construct_afriat_utility.

Returns a function that estimates the value/utility of any action vector.

Example:
    >>> from pyrevealed import BehaviorLog, fit_latent_values, build_value_function
    >>> result = fit_latent_values(user_log)
    >>> value_fn = build_value_function(user_log, result)
    >>> estimated_value = value_fn(new_action_vector)
"""

# predict_choice: Tech-friendly name for predict_demand
predict_choice = predict_demand
"""
Predict what action a user will take given new costs and a resource limit.

This is the tech-friendly alias for predict_demand.

Uses the fitted latent value model to predict user behavior under
new conditions (counterfactual analysis).

Example:
    >>> from pyrevealed import BehaviorLog, fit_latent_values, predict_choice
    >>> result = fit_latent_values(user_log)
    >>> predicted_action = predict_choice(user_log, result, new_costs, budget)
"""
