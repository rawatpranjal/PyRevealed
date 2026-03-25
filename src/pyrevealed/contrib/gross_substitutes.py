"""Gross substitutes test and compensated demand analysis.

Tests for gross substitutes/complements relationships and implements
Slutsky decomposition of price effects. Based on Chapter 10.3 of
Chambers & Echenique (2016) "Revealed Preference Theory".

Tech-Friendly Names (Primary):
    - test_cross_price_effect(): Test substitute/complement relationship
    - decompose_price_effects(): Slutsky decomposition
    - compute_hicksian_demand(): Compensated demand estimation

Economics Names (Legacy Aliases):
    - check_gross_substitutes() -> test_cross_price_effect()
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import (
    GrossSubstitutesResult,
    SubstitutionMatrixResult,
    CompensatedDemandResult,
)
from pyrevealed.core.exceptions import ValueRangeError, DataValidationError


def check_gross_substitutes(
    session: ConsumerSession,
    good_g: int,
    good_h: int,
    price_change_threshold: float = 0.05,
    tolerance: float = 1e-10,
) -> GrossSubstitutesResult:
    """
    Test if two goods are gross substitutes based on revealed preference data.

    Gross substitutes: when the price of good g increases (other prices constant)
    and quantity of g decreases, we should see quantity of h increase.

    Gross complements: when p_g increases and x_g decreases, x_h also decreases.

    The algorithm:
    1. Finds observation pairs where price of g changed significantly
    2. Checks if other prices stayed relatively constant
    3. Analyzes the direction of quantity changes
    4. Classifies the relationship based on majority of informative pairs

    Args:
        session: ConsumerSession with prices and quantities
        good_g: Index of first good
        good_h: Index of second good (potential substitute)
        price_change_threshold: Minimum relative price change to consider (default 5%)
        tolerance: Numerical tolerance

    Returns:
        GrossSubstitutesResult with relationship classification and confidence

    Example:
        >>> import numpy as np
        >>> from pyrevealed import ConsumerSession, check_gross_substitutes
        >>> # Prices for goods 0 and 1 over 3 observations
        >>> prices = np.array([[1.0, 2.0], [2.0, 2.0], [1.0, 1.0]])
        >>> quantities = np.array([[4.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
        >>> session = ConsumerSession(prices=prices, quantities=quantities)
        >>> result = check_gross_substitutes(session, good_g=0, good_h=1)
        >>> print(f"Relationship: {result.relationship}")

    References:
        Hicks, J. R. (1939). Value and Capital. Oxford University Press.
    """
    start_time = time.perf_counter()

    T = session.num_observations
    N = session.num_goods

    if good_g < 0 or good_g >= N or good_h < 0 or good_h >= N:
        raise ValueRangeError(
            f"Good indices must be in [0, {N}). "
            f"Got good_g={good_g}, good_h={good_h}. "
            f"Hint: Use valid indices from 0 to {N - 1}."
        )
    if good_g == good_h:
        raise DataValidationError(
            f"good_g and good_h must be different. Got {good_g} for both. "
            f"Hint: Gross substitutes analysis requires two distinct goods."
        )

    P = session.prices  # T x N
    Q = session.quantities  # T x N

    substitutes_pairs: list[tuple[int, int]] = []
    complements_pairs: list[tuple[int, int]] = []

    # Compare all pairs of observations
    for i in range(T):
        for j in range(i + 1, T):
            # Check if price of g changed significantly while others ~constant
            pg_i, pg_j = P[i, good_g], P[j, good_g]
            ph_i, ph_j = P[i, good_h], P[j, good_h]

            # Skip if prices are near zero
            if pg_i < tolerance or pg_j < tolerance:
                continue

            # Relative price change for good g
            rel_change_g = abs(pg_j - pg_i) / pg_i

            # Check if price of h stayed relatively constant
            rel_change_h = abs(ph_j - ph_i) / max(ph_i, tolerance)

            # We want: significant change in p_g, small change in p_h
            if rel_change_g < price_change_threshold:
                continue  # Not enough price movement in g
            if rel_change_h > rel_change_g * 0.5:
                continue  # Too much change in h relative to g

            # Check other prices didn't change too much
            other_goods = [k for k in range(N) if k != good_g and k != good_h]
            if other_goods:
                other_changes = [
                    abs(P[j, k] - P[i, k]) / max(P[i, k], tolerance)
                    for k in other_goods
                ]
                if max(other_changes) > rel_change_g * 0.5:
                    continue  # Other prices changed too much

            # Get quantity changes
            xg_i, xg_j = Q[i, good_g], Q[j, good_g]
            xh_i, xh_j = Q[i, good_h], Q[j, good_h]

            # Direction of price change for g
            pg_increased = pg_j > pg_i + tolerance
            pg_decreased = pg_j < pg_i - tolerance

            # Direction of quantity changes
            xg_increased = xg_j > xg_i + tolerance
            xg_decreased = xg_j < xg_i - tolerance
            xh_increased = xh_j > xh_i + tolerance
            xh_decreased = xh_j < xh_i - tolerance

            # Gross substitutes pattern:
            # p_g up, x_g down => x_h up (or p_g down, x_g up => x_h down)
            if pg_increased and xg_decreased:
                if xh_increased:
                    substitutes_pairs.append((i, j))
                elif xh_decreased:
                    complements_pairs.append((i, j))
            elif pg_decreased and xg_increased:
                if xh_decreased:
                    substitutes_pairs.append((i, j))
                elif xh_increased:
                    complements_pairs.append((i, j))

    # Determine relationship
    n_subs = len(substitutes_pairs)
    n_comp = len(complements_pairs)
    informative_pairs = n_subs + n_comp

    if informative_pairs == 0:
        relationship = "inconclusive"
        are_substitutes = False
        are_complements = False
        confidence = 0.0
        supporting = []
        violating = []
    elif n_subs > n_comp:
        relationship = "substitutes"
        are_substitutes = True
        are_complements = False
        confidence = n_subs / informative_pairs
        supporting = substitutes_pairs
        violating = complements_pairs
    elif n_comp > n_subs:
        relationship = "complements"
        are_substitutes = False
        are_complements = True
        confidence = n_comp / informative_pairs
        supporting = complements_pairs
        violating = substitutes_pairs
    else:
        relationship = "independent"
        are_substitutes = False
        are_complements = False
        confidence = 0.5
        supporting = []
        violating = []

    computation_time = (time.perf_counter() - start_time) * 1000

    return GrossSubstitutesResult(
        are_substitutes=are_substitutes,
        are_complements=are_complements,
        relationship=relationship,
        supporting_pairs=supporting,
        violating_pairs=violating,
        confidence_score=confidence,
        good_g_index=good_g,
        good_h_index=good_h,
        computation_time_ms=computation_time,
    )


# Relationship codes for efficient storage
_RELATIONSHIP_CODES = {
    "self": 0,
    "substitutes": 1,
    "complements": 2,
    "independent": 3,
    "inconclusive": 4,
}
_CODE_TO_RELATIONSHIP = {v: k for k, v in _RELATIONSHIP_CODES.items()}


def compute_substitution_matrix(
    session: ConsumerSession,
    price_change_threshold: float = 0.05,
) -> SubstitutionMatrixResult:
    """
    Compute pairwise substitution relationships for all goods.

    Returns an N x N matrix where entry [g, h] indicates the relationship
    between goods g and h.

    Args:
        session: ConsumerSession
        price_change_threshold: Minimum price change to consider

    Returns:
        SubstitutionMatrixResult with relationship matrix

    Example:
        >>> from pyrevealed import ConsumerSession, compute_substitution_matrix
        >>> result = compute_substitution_matrix(session)
        >>> print(f"Substitute pairs: {result.substitute_pairs}")
        >>> print(f"Complement pairs: {result.complement_pairs}")
    """
    start_time = time.perf_counter()

    N = session.num_goods
    # Use int8 codes for memory efficiency instead of object dtype strings
    relationship_codes = np.zeros((N, N), dtype=np.int8)
    confidence_matrix = np.zeros((N, N))

    for g in range(N):
        for h in range(N):
            if g == h:
                relationship_codes[g, h] = _RELATIONSHIP_CODES["self"]
                confidence_matrix[g, h] = 1.0
            elif g < h:
                result = check_gross_substitutes(session, g, h, price_change_threshold)
                code = _RELATIONSHIP_CODES.get(result.relationship, _RELATIONSHIP_CODES["inconclusive"])
                relationship_codes[g, h] = code
                relationship_codes[h, g] = code
                confidence_matrix[g, h] = result.confidence_score
                confidence_matrix[h, g] = result.confidence_score

    # Convert codes back to strings for the result object (for API compatibility)
    relationship_matrix = np.empty((N, N), dtype=object)
    for g in range(N):
        for h in range(N):
            relationship_matrix[g, h] = _CODE_TO_RELATIONSHIP[relationship_codes[g, h]]

    computation_time = (time.perf_counter() - start_time) * 1000

    return SubstitutionMatrixResult(
        relationship_matrix=relationship_matrix,
        confidence_matrix=confidence_matrix,
        num_goods=N,
        computation_time_ms=computation_time,
    )


def check_law_of_demand(
    session: ConsumerSession,
    good: int,
    price_change_threshold: float = 0.05,
    tolerance: float = 1e-10,
) -> dict:
    """
    Check if a good satisfies the law of demand (own-price effect is negative).

    The law of demand states that when price increases, quantity demanded
    decreases (holding other factors constant).

    Args:
        session: ConsumerSession
        good: Index of the good to test
        price_change_threshold: Minimum price change to consider
        tolerance: Numerical tolerance

    Returns:
        Dictionary with:
        - satisfies_law: True if law of demand holds
        - supporting_pairs: Pairs where law holds
        - violating_pairs: Pairs where law is violated (Giffen good behavior)
        - confidence: Fraction of pairs supporting the law
    """
    T = session.num_observations
    N = session.num_goods
    P = session.prices
    Q = session.quantities

    supporting_pairs: list[tuple[int, int]] = []
    violating_pairs: list[tuple[int, int]] = []

    for i in range(T):
        for j in range(i + 1, T):
            pg_i, pg_j = P[i, good], P[j, good]

            if pg_i < tolerance or pg_j < tolerance:
                continue

            rel_change = abs(pg_j - pg_i) / pg_i
            if rel_change < price_change_threshold:
                continue

            # Check other prices (vectorized)
            mask = np.ones(N, dtype=bool)
            mask[good] = False
            if np.any(mask):
                other_changes = np.abs(P[j, mask] - P[i, mask]) / np.maximum(P[i, mask], tolerance)
                if np.max(other_changes) > rel_change * 0.5:
                    continue

            xg_i, xg_j = Q[i, good], Q[j, good]

            # Law of demand: price up => quantity down
            price_up = pg_j > pg_i + tolerance
            price_down = pg_j < pg_i - tolerance
            qty_up = xg_j > xg_i + tolerance
            qty_down = xg_j < xg_i - tolerance

            if (price_up and qty_down) or (price_down and qty_up):
                supporting_pairs.append((i, j))
            elif (price_up and qty_up) or (price_down and qty_down):
                violating_pairs.append((i, j))

    total = len(supporting_pairs) + len(violating_pairs)
    confidence = len(supporting_pairs) / total if total > 0 else 0.5

    return {
        "satisfies_law": len(violating_pairs) == 0 and len(supporting_pairs) > 0,
        "supporting_pairs": supporting_pairs,
        "violating_pairs": violating_pairs,
        "confidence": confidence,
        "num_informative_pairs": total,
    }


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

# test_cross_price_effect: Tech-friendly name for check_gross_substitutes
test_cross_price_effect = check_gross_substitutes
"""
Test how changes in one item's price affect demand for another item.

This is the tech-friendly alias for check_gross_substitutes.

Use this to understand cross-price relationships between products:
- Substitutes: Price of A up → Demand for B up (users switch)
- Complements: Price of A up → Demand for B down (bought together)
- Independent: No clear relationship

Example:
    >>> from pyrevealed import BehaviorLog, test_cross_price_effect
    >>> result = test_cross_price_effect(user_log, good_g=0, good_h=1)
    >>> if result.are_substitutes:
    ...     print("Users treat these as substitutes")
"""

compute_cross_price_matrix = compute_substitution_matrix
"""
Compute all pairwise cross-price relationships.

Returns an N x N matrix of relationships between all goods.
"""


# =============================================================================
# COMPENSATED DEMAND (Chapter 10.3)
# =============================================================================


def decompose_price_effects(
    session: ConsumerSession,
    price_change_threshold: float = 0.05,
    tolerance: float = 1e-10,
) -> CompensatedDemandResult:
    """
    Decompose price effects into substitution and income effects.

    The Slutsky equation states:
    Total effect = Substitution effect + Income effect
    dx_i/dp_j = (∂x_i/∂p_j)|_u + x_j * (∂x_i/∂m)

    The substitution effect measures how demand changes when price changes
    but utility is held constant (compensated demand).

    The income effect measures how the change in purchasing power
    affects demand.

    Args:
        session: ConsumerSession with prices and quantities
        price_change_threshold: Minimum price change to consider
        tolerance: Numerical tolerance

    Returns:
        CompensatedDemandResult with Slutsky decomposition

    Example:
        >>> from pyrevealed import ConsumerSession, decompose_price_effects
        >>> result = decompose_price_effects(user_session)
        >>> print(f"Substitution effect for good 0 from price 1: {result.substitution_effects[0,1]:.3f}")
        >>> print(f"Income effect: {result.income_effects[0,1]:.3f}")
        >>> print(f"Satisfies compensated law: {result.satisfies_compensated_law}")

    References:
        Chambers & Echenique (2016), Chapter 10.3
        Slutsky, E. (1915). "On the Theory of the Budget of the Consumer"
    """
    start_time = time.perf_counter()

    T = session.num_observations
    N = session.num_goods
    P = session.prices
    Q = session.quantities

    # Compute substitution and income effects matrices
    substitution_effects = np.zeros((N, N), dtype=np.float64)
    income_effects = np.zeros((N, N), dtype=np.float64)
    total_effects = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            s_ij, ie_ij, te_ij = _estimate_slutsky_components(
                P, Q, i, j, price_change_threshold, tolerance
            )
            substitution_effects[i, j] = s_ij
            income_effects[i, j] = ie_ij
            total_effects[i, j] = te_ij

    # Compute own-price elasticities
    own_price_elasticities = {}
    for i in range(N):
        avg_price = np.mean(P[:, i])
        avg_qty = np.mean(Q[:, i])
        if avg_qty > tolerance:
            own_price_elasticities[i] = total_effects[i, i] * avg_price / avg_qty
        else:
            own_price_elasticities[i] = 0.0

    # Compute cross-price elasticities
    cross_price_elasticities = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            avg_price_j = np.mean(P[:, j])
            avg_qty_i = np.mean(Q[:, i])
            if avg_qty_i > tolerance:
                cross_price_elasticities[i, j] = total_effects[i, j] * avg_price_j / avg_qty_i

    # Check compensated law of demand
    # Substitution effects should be negative semi-definite
    # Own-price substitution effects should be negative (or zero)
    violations = []
    for i in range(N):
        if substitution_effects[i, i] > tolerance:
            violations.append((i, i))

    satisfies_compensated_law = len(violations) == 0

    computation_time = (time.perf_counter() - start_time) * 1000

    return CompensatedDemandResult(
        substitution_effects=substitution_effects,
        income_effects=income_effects,
        satisfies_compensated_law=satisfies_compensated_law,
        own_price_elasticities=own_price_elasticities,
        cross_price_elasticities=cross_price_elasticities,
        violations=violations,
        computation_time_ms=computation_time,
    )


def compute_hicksian_demand(
    session: ConsumerSession,
    target_utility: float | None = None,
    method: str = "exact",
) -> dict:
    """
    Compute Hicksian (compensated) demand via expenditure minimization.

    Hicksian demand h(p, u) solves:
        min_x  p @ x
        s.t.   U(x) >= u
               x >= 0

    This implementation recovers the Afriat utility function U(x) from the data
    and uses constrained optimization to solve for Hicksian demand at any
    price vector and utility level.

    Args:
        session: ConsumerSession with prices and quantities
        target_utility: Utility level for computing derivatives (default: median)
        method: Computation method:
            - "exact": Full Afriat recovery + constrained optimization
            - "approximation": Legacy finite differences (faster but approximate)

    Returns:
        Dictionary containing:
        - 'success': Whether Afriat utility recovery succeeded
        - 'hicksian_demand_fn': Callable (prices, utility) -> quantities
        - 'hicksian_derivatives': N x N matrix of ∂h_i/∂p_j at target utility
        - 'target_utility': Utility level used for derivatives
        - 'utility_function': Callable to evaluate utility at any bundle
        - 'observation_utilities': Utility at each observed bundle
        - 'observations_used': Number of observations (for backward compatibility)

    Example:
        >>> from pyrevealed import ConsumerSession, compute_hicksian_demand
        >>> import numpy as np
        >>> P = np.array([[1.0, 2.0], [1.5, 1.5], [2.0, 1.0]])
        >>> Q = np.array([[2.0, 1.0], [1.5, 1.5], [1.0, 2.0]])
        >>> session = ConsumerSession(prices=P, quantities=Q)
        >>> result = compute_hicksian_demand(session, method='exact')
        >>> if result['success']:
        ...     h = result['hicksian_demand_fn']
        ...     print(f"h([1,1], 0.5) = {h([1, 1], 0.5)}")

    References:
        Chambers & Echenique (2016), Chapter 10.3
        Afriat (1967), "The Construction of Utility Functions"
    """
    if method == "approximation":
        return _compute_hicksian_demand_approximation(session, target_utility)

    T = session.num_observations
    N = session.num_goods
    P = session.prices
    Q = session.quantities

    # Recover Afriat utility function
    utility_fn, U, lambdas, success = _recover_afriat_utility_for_hicksian(P, Q)

    if not success or utility_fn is None:
        # Fall back to approximation method
        result = _compute_hicksian_demand_approximation(session, target_utility)
        result['success'] = False
        result['utility_function'] = None
        result['hicksian_demand_fn'] = None
        return result

    # Compute utility at each observation
    observation_utilities = np.array([utility_fn(Q[t]) for t in range(T)])

    if target_utility is None:
        target_utility = np.median(observation_utilities)

    # Create Hicksian demand function
    def hicksian_demand_fn(
        prices: NDArray[np.float64],
        utility_level: float,
    ) -> NDArray[np.float64] | None:
        """
        Compute Hicksian demand h(p, u).

        Args:
            prices: N-dimensional price vector
            utility_level: Target utility level

        Returns:
            N-dimensional quantity vector or None if optimization fails
        """
        return _solve_hicksian_at_point(
            utility_fn, np.asarray(prices), utility_level, Q, tolerance=1e-8
        )

    # Compute Hicksian derivatives at target utility level
    # ∂h_i/∂p_j ≈ [h_i(p + ε*e_j, u) - h_i(p - ε*e_j, u)] / (2ε)
    hicksian_derivatives = np.zeros((N, N), dtype=np.float64)

    # Use mean prices as evaluation point
    p_mean = np.mean(P, axis=0)
    epsilon = 0.01 * np.mean(p_mean)  # 1% perturbation

    # Compute baseline Hicksian demand
    h_base = hicksian_demand_fn(p_mean, target_utility)

    if h_base is not None:
        for j in range(N):
            p_plus = p_mean.copy()
            p_plus[j] += epsilon
            p_minus = p_mean.copy()
            p_minus[j] -= epsilon

            h_plus = hicksian_demand_fn(p_plus, target_utility)
            h_minus = hicksian_demand_fn(p_minus, target_utility)

            if h_plus is not None and h_minus is not None:
                for i in range(N):
                    hicksian_derivatives[i, j] = (h_plus[i] - h_minus[i]) / (2 * epsilon)

    return {
        "success": True,
        "hicksian_demand_fn": hicksian_demand_fn,
        "hicksian_derivatives": hicksian_derivatives,
        "target_utility": target_utility,
        "utility_function": utility_fn,
        "observation_utilities": observation_utilities,
        "observations_used": T,
    }


def _recover_afriat_utility_for_hicksian(
    P: NDArray[np.float64],
    Q: NDArray[np.float64],
    tolerance: float = 1e-8,
) -> tuple[callable | None, NDArray | None, NDArray | None, bool]:
    """
    Recover Afriat utility function from price-quantity data.

    Solves the LP:
        min Σ λ_k
        s.t. U_k <= U_l + λ_l * p_l @ (x_k - x_l)  for all k, l
             U_k >= 0, λ_k > ε

    Then constructs:
        u(x) = min_k { U_k + λ_k * p_k @ (x - x_k) }

    Returns:
        Tuple of (utility_function, U_values, lambda_values, success)
    """
    from scipy.optimize import linprog

    T = P.shape[0]
    n_vars = 2 * T

    constraints_A = []
    constraints_b = []

    for k in range(T):
        for obs_l in range(T):
            if k == obs_l:
                continue

            row = np.zeros(n_vars)
            row[k] = 1.0
            row[obs_l] = -1.0

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

            # Construct the Afriat utility function
            def utility_fn(x: NDArray[np.float64]) -> float:
                x = np.asarray(x, dtype=np.float64)
                values = np.zeros(T)
                for k in range(T):
                    values[k] = U[k] + lambdas[k] * (P[k] @ (x - Q[k]))
                return float(np.min(values))

            return utility_fn, U, lambdas, True
    except Exception as e:
        from pyrevealed.core.exceptions import OptimizationError

        raise OptimizationError(
            f"Afriat utility recovery failed for Hicksian demand. Original error: {e}"
        ) from e

    return None, None, None, False


def _solve_hicksian_at_point(
    utility_fn: callable,
    prices: NDArray[np.float64],
    target_utility: float,
    observed_Q: NDArray[np.float64],
    tolerance: float = 1e-8,
) -> NDArray[np.float64] | None:
    """
    Solve the Hicksian demand problem at a specific price-utility point.

    min_x  p @ x
    s.t.   U(x) >= u
           x >= 0

    Args:
        utility_fn: Afriat utility function
        prices: N-dimensional price vector
        target_utility: Target utility level u
        observed_Q: Observed quantity matrix (for initial guess)
        tolerance: Numerical tolerance

    Returns:
        Optimal quantity vector or None if optimization fails
    """
    from scipy.optimize import minimize

    N = len(prices)

    def objective(x: NDArray[np.float64]) -> float:
        return float(prices @ x)

    def utility_constraint(x: NDArray[np.float64]) -> float:
        return utility_fn(x) - target_utility

    # Find initial guess: observed bundle closest to target utility
    T = observed_Q.shape[0]
    obs_utilities = np.array([utility_fn(observed_Q[t]) for t in range(T)])
    closest_idx = np.argmin(np.abs(obs_utilities - target_utility))
    x0 = observed_Q[closest_idx].copy()

    # Ensure x0 is positive
    x0 = np.maximum(x0, 1e-6)

    bounds = [(1e-10, None)] * N
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
            return result.x
    except Exception as e:
        from pyrevealed.core.exceptions import OptimizationError

        raise OptimizationError(
            f"Hicksian demand optimization failed. Original error: {e}"
        ) from e

    return None


def _compute_hicksian_demand_approximation(
    session: ConsumerSession,
    target_utility: float | None = None,
) -> dict:
    """
    Legacy approximation of Hicksian demand using finite differences.

    This is the original implementation kept for backward compatibility
    and as a fallback when Afriat recovery fails.
    """
    T = session.num_observations
    N = session.num_goods
    P = session.prices
    Q = session.quantities

    # Estimate utility for each observation
    utilities = np.zeros(T)
    for t in range(T):
        utilities[t] = np.sum(np.log(Q[t] + 1e-10))

    if target_utility is None:
        target_utility = np.median(utilities)

    # Find observations near target utility
    utility_diffs = np.abs(utilities - target_utility)
    near_target = utility_diffs < np.std(utilities) * 0.5

    if np.sum(near_target) < 2:
        near_target = np.ones(T, dtype=bool)

    # Estimate Hicksian demand derivatives
    hicksian_derivatives = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            effects = []

            for t1 in range(T):
                if not near_target[t1]:
                    continue
                for t2 in range(t1 + 1, T):
                    if not near_target[t2]:
                        continue

                    dp_j = P[t2, j] - P[t1, j]
                    if abs(dp_j) < 1e-10:
                        continue

                    dq_i = Q[t2, i] - Q[t1, i]
                    effects.append(dq_i / dp_j)

            if effects:
                hicksian_derivatives[i, j] = np.median(effects)

    return {
        "success": False,
        "hicksian_demand_fn": None,
        "hicksian_derivatives": hicksian_derivatives,
        "target_utility": target_utility,
        "utility_function": None,
        "observation_utilities": utilities,
        "observations_used": int(np.sum(near_target)),
    }


def check_compensated_law_of_demand(
    session: ConsumerSession,
    tolerance: float = 1e-6,
) -> dict:
    """
    Check if data satisfies the compensated law of demand.

    The compensated law states that substitution effects are negative
    for own-price changes: when price increases (holding utility constant),
    quantity demanded decreases.

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Numerical tolerance

    Returns:
        Dictionary with test results
    """
    result = decompose_price_effects(session)

    # Check own-price substitution effects
    N = session.num_goods
    violations = []

    for i in range(N):
        if result.substitution_effects[i, i] > tolerance:
            violations.append(i)

    return {
        "satisfies_law": len(violations) == 0,
        "violations": violations,
        "substitution_effects_diagonal": np.diag(result.substitution_effects),
    }


def _estimate_slutsky_components(
    P: NDArray[np.float64],
    Q: NDArray[np.float64],
    good_i: int,
    good_j: int,
    price_threshold: float,
    tolerance: float,
) -> tuple[float, float, float]:
    """
    Estimate Slutsky decomposition components for a pair of goods.

    Returns:
        Tuple of (substitution_effect, income_effect, total_effect)
    """
    T = P.shape[0]
    N = P.shape[1]

    # Precompute expenditures (vectorized)
    expenditures = np.sum(P * Q, axis=1)

    total_effects = []
    income_effects_raw = []

    for t1 in range(T):
        for t2 in range(t1 + 1, T):
            # Price change in good j
            dp_j = P[t2, good_j] - P[t1, good_j]
            if abs(dp_j) / max(P[t1, good_j], tolerance) < price_threshold:
                continue

            # Check other prices are relatively stable (vectorized)
            mask = np.ones(N, dtype=bool)
            mask[good_j] = False
            other_change = np.sum(
                np.abs(P[t2, mask] - P[t1, mask]) / np.maximum(P[t1, mask], tolerance)
            )

            if other_change > price_threshold * (N - 1) * 0.5:
                continue

            # Quantity change in good i
            dq_i = Q[t2, good_i] - Q[t1, good_i]

            # Total effect (Marshallian)
            total_effect = dq_i / dp_j
            total_effects.append(total_effect)

            # Estimate income effect
            # Income effect = x_j * (dq_i/dm)
            dm = expenditures[t2] - expenditures[t1]

            if abs(dm) > tolerance:
                dq_dm = dq_i / dm
                x_j_avg = (Q[t1, good_j] + Q[t2, good_j]) / 2
                income_effect = x_j_avg * dq_dm
                income_effects_raw.append(income_effect)

    if not total_effects:
        return 0.0, 0.0, 0.0

    total_effect = np.median(total_effects)
    income_effect = np.median(income_effects_raw) if income_effects_raw else 0.0
    substitution_effect = total_effect - income_effect

    return substitution_effect, income_effect, total_effect


# =============================================================================
# ADDITIONAL TECH-FRIENDLY ALIASES
# =============================================================================

compute_slutsky_decomposition = decompose_price_effects
"""
Compute Slutsky decomposition of price effects.

Alias for decompose_price_effects.
"""

estimate_compensated_demand = compute_hicksian_demand
"""
Estimate compensated (Hicksian) demand.

Alias for compute_hicksian_demand.
"""
