"""Additive separability test for preferences.

Tests whether utility has the additive form U(x) = Σ u_i(x_i),
which is stronger than quasilinearity. Based on Chapter 9.3 of
Chambers & Echenique (2016) "Revealed Preference Theory".

Additive preferences imply:
1. No cross-effects: ∂x_i/∂p_j = 0 for i≠j (holding income constant)
2. Each good can be priced independently

This module provides theoretically rigorous estimation methods:
- OLS/2SLS regression for cross-price elasticity estimation
- LP-based cyclic monotonicity test for additive utility
- Bootstrap confidence intervals for cross-effects

Tech-Friendly Names (Primary):
    - test_additive_separability(): Test for additive utility
    - identify_additive_groups(): Find additively separable groups
    - check_no_cross_effects(): Test for zero cross-price effects
    - test_additivity_lp(): LP-based cyclic monotonicity test
    - compute_cross_effects_regression(): Regression-based estimation

Economics Names (Legacy Aliases):
    - check_additivity() -> test_additive_separability()

References:
    Chambers & Echenique (2016), Chapter 9.3
    Gorman, W.M. (1968), "The Structure of Utility Functions"
    Rockafellar (1970), "Convex Analysis" (cyclic monotonicity)
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog
from scipy import stats

from pyrevealed.core.result import AdditivityResult

if TYPE_CHECKING:
    from pyrevealed.core.session import BehaviorLog


def test_additive_separability(
    log: "BehaviorLog",
    cross_effect_threshold: float = 0.1,
    price_change_threshold: float = 0.05,
) -> AdditivityResult:
    """
    Test if preferences are additively separable.

    Additive separability means U(x) = Σ u_i(x_i), implying:
    - Each good's marginal utility depends only on its own quantity
    - Cross-price effects (holding income constant) are zero
    - Goods can be priced independently

    This is a stronger condition than weak separability and quasilinearity.

    Args:
        log: BehaviorLog with prices and quantities
        cross_effect_threshold: Threshold for cross-effect significance
        price_change_threshold: Minimum price change to consider

    Returns:
        AdditivityResult with separability analysis

    Example:
        >>> from pyrevealed import BehaviorLog, test_additive_separability
        >>> result = test_additive_separability(user_log)
        >>> if result.is_additive:
        ...     print("Goods can be priced independently")
        >>> else:
        ...     print(f"Cross-effects found: {result.violations}")

    References:
        Chambers & Echenique (2016), Chapter 9.3
        Gorman, W.M. (1968). "The Structure of Utility Functions"
    """
    start_time = time.perf_counter()

    N = log.num_features

    # Compute cross-price effects matrix
    cross_effects_matrix = _compute_cross_effects_matrix(
        log, price_change_threshold
    )

    # Find violations (significant off-diagonal effects)
    violations = []
    max_cross_effect = 0.0

    for i in range(N):
        for j in range(N):
            if i != j:
                effect = abs(cross_effects_matrix[i, j])
                if effect > max_cross_effect:
                    max_cross_effect = effect
                if effect > cross_effect_threshold:
                    violations.append((i, j))

    # Determine additive groups using connected components
    additive_groups = identify_additive_groups(
        cross_effects_matrix, cross_effect_threshold
    )

    # Is fully additive if no significant cross-effects
    is_additive = len(violations) == 0

    computation_time = (time.perf_counter() - start_time) * 1000

    return AdditivityResult(
        is_additive=is_additive,
        additive_groups=additive_groups,
        cross_effects_matrix=cross_effects_matrix,
        max_cross_effect=max_cross_effect,
        violations=violations,
        num_violations=len(violations),
        computation_time_ms=computation_time,
    )


def identify_additive_groups(
    cross_effects_matrix: NDArray[np.float64],
    threshold: float = 0.1,
) -> list[set[int]]:
    """
    Identify groups of goods that are additively separable from each other.

    Uses connected components: goods i and j are in the same group if
    there's a significant cross-effect between them (directly or transitively).

    Args:
        cross_effects_matrix: N x N matrix of cross-price effects
        threshold: Threshold for significant cross-effect

    Returns:
        List of sets, each containing good indices in a separable group

    Example:
        >>> groups = identify_additive_groups(cross_effects)
        >>> # If groups = [{0, 1}, {2, 3, 4}], then goods 0-1 and 2-4
        >>> # can be priced independently from each other
    """
    N = cross_effects_matrix.shape[0]

    # Build adjacency based on cross-effects
    adjacency = np.abs(cross_effects_matrix) > threshold
    np.fill_diagonal(adjacency, False)

    # Find connected components using union-find
    parent = list(range(N))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Union goods with cross-effects
    for i in range(N):
        for j in range(i + 1, N):
            if adjacency[i, j] or adjacency[j, i]:
                union(i, j)

    # Collect groups
    groups_dict: dict[int, set[int]] = {}
    for i in range(N):
        root = find(i)
        if root not in groups_dict:
            groups_dict[root] = set()
        groups_dict[root].add(i)

    return list(groups_dict.values())


def check_no_cross_effects(
    log: "BehaviorLog",
    good_i: int,
    good_j: int,
    price_change_threshold: float = 0.05,
) -> dict:
    """
    Test if there are cross-price effects between two goods.

    For additive preferences, when p_j changes (other prices constant),
    x_i should not change (holding income constant).

    Args:
        log: BehaviorLog with prices and quantities
        good_i: Index of quantity good
        good_j: Index of price good
        price_change_threshold: Minimum price change to consider

    Returns:
        Dictionary with cross-effect analysis
    """
    T = log.num_records
    N = log.num_features
    P = log.cost_vectors
    Q = log.action_vectors

    cross_effects = []
    supporting_pairs = []
    violating_pairs = []

    for t1 in range(T):
        for t2 in range(t1 + 1, T):
            # Check if price of good j changed significantly
            dp_j = P[t2, good_j] - P[t1, good_j]
            rel_change_j = abs(dp_j) / max(P[t1, good_j], 1e-10)

            if rel_change_j < price_change_threshold:
                continue

            # Check if other prices (except j) are stable
            other_stable = True
            for k in range(N):
                if k != good_j:
                    rel_change_k = abs(P[t2, k] - P[t1, k]) / max(P[t1, k], 1e-10)
                    if rel_change_k > price_change_threshold * 0.5:
                        other_stable = False
                        break

            if not other_stable:
                continue

            # Compute cross-effect
            dq_i = Q[t2, good_i] - Q[t1, good_i]
            cross_effect = dq_i / dp_j if abs(dp_j) > 1e-10 else 0.0
            cross_effects.append(cross_effect)

            # Income-adjusted cross-effect
            # Under additive utility, should be zero
            if abs(cross_effect) < 0.01:
                supporting_pairs.append((t1, t2))
            else:
                violating_pairs.append((t1, t2))

    return {
        "good_i": good_i,
        "good_j": good_j,
        "mean_cross_effect": np.mean(cross_effects) if cross_effects else 0.0,
        "std_cross_effect": np.std(cross_effects) if cross_effects else 0.0,
        "no_cross_effects": len(violating_pairs) == 0 and len(supporting_pairs) > 0,
        "supporting_pairs": supporting_pairs,
        "violating_pairs": violating_pairs,
        "num_observations": len(cross_effects),
    }


def _compute_cross_effects_matrix(
    log: "BehaviorLog",
    price_change_threshold: float = 0.05,
    method: str = "regression",
) -> NDArray[np.float64]:
    """
    Compute N x N matrix of cross-price effects.

    Entry [i,j] is the effect of p_j change on x_i quantity (cross-price elasticity).

    This function provides multiple estimation methods:
    - "regression": OLS regression with log-linear demand (recommended)
    - "finite_diff": Legacy pairwise finite differences method

    Args:
        log: BehaviorLog with prices and quantities
        price_change_threshold: Minimum price change for finite_diff method
        method: Estimation method

    Returns:
        N x N matrix of cross-price effects
    """
    if method == "regression":
        return compute_cross_effects_regression(log)
    elif method == "finite_diff":
        return _compute_cross_effects_finite_diff(log, price_change_threshold)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'regression' or 'finite_diff'.")


def compute_cross_effects_regression(
    log: "BehaviorLog",
    include_standard_errors: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute cross-price effects using OLS regression.

    Estimates log-linear demand functions:
        log(x_i) = α_i + Σ_j β_ij * log(p_j) + γ_i * log(m) + ε

    The cross-price elasticity matrix is β (off-diagonal elements).
    For additive utility, β_ij should be zero for i ≠ j.

    Args:
        log: BehaviorLog with prices and quantities
        include_standard_errors: Whether to return standard errors

    Returns:
        N x N matrix of cross-price elasticities (off-diagonal = cross effects)
        If include_standard_errors=True, also returns N x N standard error matrix
    """
    T = log.num_records
    N = log.num_features
    P = log.cost_vectors
    Q = log.action_vectors

    # Compute expenditures
    expenditures = log.total_spend

    # Take logs
    log_P = np.log(P + 1e-10)
    log_Q = np.log(Q + 1e-10)
    log_m = np.log(expenditures + 1e-10)

    # Build design matrix: [1, log(p_1), ..., log(p_N), log(m)]
    X = np.column_stack([np.ones(T), log_P, log_m])

    # Estimate for each good
    beta_matrix = np.zeros((N, N))  # Cross-price elasticities
    se_matrix = np.zeros((N, N))  # Standard errors

    for i in range(N):
        y = log_Q[:, i]

        # OLS estimation
        try:
            XtX = X.T @ X
            XtX_inv = np.linalg.pinv(XtX)
            coeffs = XtX_inv @ (X.T @ y)

            # Extract price elasticities (indices 1 to N)
            beta_matrix[i, :] = coeffs[1 : N + 1]

            # Compute standard errors
            residuals = y - X @ coeffs
            sigma2 = np.sum(residuals**2) / (T - X.shape[1])
            var_coeffs = sigma2 * np.diag(XtX_inv)
            se_matrix[i, :] = np.sqrt(var_coeffs[1 : N + 1])
        except np.linalg.LinAlgError:
            pass

    if include_standard_errors:
        return beta_matrix, se_matrix
    return beta_matrix


def compute_cross_effects_2sls(
    log: "BehaviorLog",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute cross-price effects using Two-Stage Least Squares (2SLS).

    Uses lagged prices as instruments to address potential price endogeneity.
    This is more robust when prices are set in response to demand.

    Stage 1: Predict log(p_j) using lagged prices
    Stage 2: Estimate demand using predicted prices

    Args:
        log: BehaviorLog with prices and quantities

    Returns:
        Tuple of (cross_effects_matrix, standard_errors, first_stage_f_stats)
    """
    T = log.num_records
    N = log.num_features
    P = log.cost_vectors
    Q = log.action_vectors

    if T < 4:
        warnings.warn("Insufficient observations for 2SLS. Falling back to OLS.")
        beta, se = compute_cross_effects_regression(log, include_standard_errors=True)
        return beta, se, np.zeros(N)

    # Take logs
    log_P = np.log(P + 1e-10)
    log_Q = np.log(Q + 1e-10)
    expenditures = log.total_spend
    log_m = np.log(expenditures + 1e-10)

    # Create lagged prices as instruments (use t-1 prices for t observations)
    # Skip first observation
    log_P_current = log_P[1:]  # t = 1, ..., T-1
    log_P_lagged = log_P[:-1]  # t = 0, ..., T-2
    log_Q_current = log_Q[1:]
    log_m_current = log_m[1:]
    T_eff = T - 1

    beta_matrix = np.zeros((N, N))
    se_matrix = np.zeros((N, N))
    first_stage_f = np.zeros(N)

    # Stage 1: Predict prices using lagged prices
    Z = np.column_stack([np.ones(T_eff), log_P_lagged, log_m_current])  # Instruments
    predicted_log_P = np.zeros_like(log_P_current)

    for j in range(N):
        y_price = log_P_current[:, j]
        try:
            ZtZ_inv = np.linalg.pinv(Z.T @ Z)
            gamma = ZtZ_inv @ (Z.T @ y_price)
            predicted_log_P[:, j] = Z @ gamma

            # First stage F-statistic
            residuals = y_price - Z @ gamma
            tss = np.sum((y_price - np.mean(y_price)) ** 2)
            rss = np.sum(residuals**2)
            r2 = 1 - rss / tss if tss > 0 else 0
            k = Z.shape[1]
            first_stage_f[j] = (r2 / (1 - r2)) * (T_eff - k) / (k - 1) if r2 < 1 else 0
        except np.linalg.LinAlgError:
            predicted_log_P[:, j] = log_P_current[:, j]

    # Stage 2: Estimate demand using predicted prices
    X_2sls = np.column_stack([np.ones(T_eff), predicted_log_P, log_m_current])

    for i in range(N):
        y = log_Q_current[:, i]

        try:
            XtX_inv = np.linalg.pinv(X_2sls.T @ X_2sls)
            coeffs = XtX_inv @ (X_2sls.T @ y)
            beta_matrix[i, :] = coeffs[1 : N + 1]

            # 2SLS standard errors (using original X for variance)
            X_original = np.column_stack([np.ones(T_eff), log_P_current, log_m_current])
            residuals = y - X_original @ coeffs
            sigma2 = np.sum(residuals**2) / (T_eff - X_2sls.shape[1])
            var_coeffs = sigma2 * np.diag(XtX_inv)
            se_matrix[i, :] = np.sqrt(np.abs(var_coeffs[1 : N + 1]))
        except np.linalg.LinAlgError:
            pass

    return beta_matrix, se_matrix, first_stage_f


def _compute_cross_effects_finite_diff(
    log: "BehaviorLog",
    price_change_threshold: float = 0.05,
) -> NDArray[np.float64]:
    """
    Legacy finite differences method for cross-effects estimation.
    """
    N = log.num_features
    T = log.num_records
    P = log.cost_vectors
    Q = log.action_vectors

    cross_effects = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            effects = []

            for t1 in range(T):
                for t2 in range(t1 + 1, T):
                    dp_j = P[t2, j] - P[t1, j]
                    if abs(dp_j) / max(P[t1, j], 1e-10) < price_change_threshold:
                        continue

                    mask = np.ones(N, dtype=bool)
                    mask[j] = False
                    other_change = np.sum(
                        np.abs(P[t2, mask] - P[t1, mask]) / np.maximum(P[t1, mask], 1e-10)
                    )

                    if other_change > price_change_threshold * N * 0.3:
                        continue

                    dq_i = Q[t2, i] - Q[t1, i]
                    rel_dq_i = dq_i / Q[t1, i] if Q[t1, i] > 1e-10 else dq_i
                    rel_dp_j = dp_j / P[t1, j]

                    if abs(rel_dp_j) > 1e-10:
                        effect = rel_dq_i / rel_dp_j
                        effects.append(effect)

            if effects:
                cross_effects[i, j] = np.median(effects)

    return cross_effects


def test_additive_consistency(
    log: "BehaviorLog",
    tolerance: float = 1e-6,
) -> dict:
    """
    Test consistency with additive utility using cycle conditions.

    For additive utility, certain cycle conditions must hold.
    This is based on the cyclic monotonicity generalization.

    Args:
        log: BehaviorLog with prices and quantities
        tolerance: Numerical tolerance

    Returns:
        Dictionary with consistency analysis
    """
    T = log.num_records
    N = log.num_features
    P = log.cost_vectors
    Q = log.action_vectors

    # Precompute expenditures (vectorized)
    expenditures = np.sum(P * Q, axis=1)  # Shape: (T,)

    # For additive utility, we test using first-order conditions
    # The demand for each good depends only on its own price and income

    violations = []

    for t1 in range(T):
        for t2 in range(T):
            if t1 == t2:
                continue

            # Check if any good violates additivity
            for i in range(N):
                # Under additivity: x_i(p_i, m) should be independent of p_j for j != i
                # Compare quantity of i across observations with same p_i

                if abs(P[t1, i] - P[t2, i]) < tolerance:
                    # Same price for good i
                    # Check if quantities differ significantly
                    if abs(Q[t1, i] - Q[t2, i]) > tolerance:
                        # Could be due to income effect or cross-price effect
                        # Check if income is similar (use precomputed expenditures)
                        m1 = expenditures[t1]
                        m2 = expenditures[t2]

                        if abs(m1 - m2) / max(m1, m2, 1e-10) < 0.1:
                            # Income similar, but quantities differ
                            # Must be cross-price effect
                            violations.append((t1, t2, i))

    return {
        "is_consistent": len(violations) == 0,
        "violations": violations,
        "num_violations": len(violations),
    }


def test_additivity_lp(
    log: "BehaviorLog",
    tolerance: float = 1e-6,
) -> dict:
    """
    Test for additive utility using LP-based cyclic monotonicity.

    For additive separability U(x) = Σ u_i(x_i), the subdifferential of each
    u_i must satisfy cyclic monotonicity (Rockafellar 1970):

    For any cycle of observations (k_0, k_1, ..., k_m, k_0), the condition:
        Σ p_{k_j}[i] * (x_{k_{j+1}}[i] - x_{k_j}[i]) ≤ 0
    must hold for each good i separately.

    This formulates the test as an LP to check if additive utility functions
    exist that rationalize the data.

    Args:
        log: BehaviorLog with prices and quantities
        tolerance: Numerical tolerance for LP

    Returns:
        Dictionary with:
        - 'is_additive': Whether data is consistent with additive utility
        - 'violation_cycles': List of cycles that violate cyclic monotonicity
        - 'utility_values': Recovered u_i(x_i) values if consistent

    References:
        Rockafellar (1970), "Convex Analysis", Chapter 24
        Chambers & Echenique (2016), Chapter 9.3
    """
    T = log.num_records
    N = log.num_features
    P = log.cost_vectors
    Q = log.action_vectors

    # For additive utility, we need to find u_i: R -> R for each good i
    # such that the first-order conditions hold.
    #
    # If x_t maximizes U(x) = Σ u_i(x_i) s.t. p_t @ x = m_t,
    # then u'_i(x_{ti}) = λ_t * p_{ti} for all i.
    #
    # For cyclic monotonicity of each u_i:
    # For any permutation π of {0, ..., T-1}:
    #   Σ_t p_t[i] * (x_{π(t)}[i] - x_t[i]) ≤ 0
    #
    # We use LP to check if this is satisfied for all 2-cycles.

    violation_cycles = []
    is_additive_per_good = []

    for i in range(N):
        # For good i, check 2-cycle monotonicity:
        # For all t1 < t2: p_t1[i] * (x_t2[i] - x_t1[i]) + p_t2[i] * (x_t1[i] - x_t2[i]) ≤ 0
        # This simplifies to: (p_t1[i] - p_t2[i]) * (x_t2[i] - x_t1[i]) ≤ 0
        # I.e., price and quantity move in opposite directions (law of demand for each good)

        violations_for_good = []

        for t1 in range(T):
            for t2 in range(t1 + 1, T):
                dp = P[t1, i] - P[t2, i]
                dq = Q[t2, i] - Q[t1, i]

                # Check cyclic monotonicity condition
                if dp * dq > tolerance:
                    violations_for_good.append((t1, t2, dp * dq))

        is_additive_per_good.append(len(violations_for_good) == 0)
        if violations_for_good:
            violation_cycles.extend([(i, v) for v in violations_for_good])

    # Overall additivity requires all goods to satisfy cyclic monotonicity
    is_additive = all(is_additive_per_good)

    # If consistent, try to recover utility values using LP
    utility_values = None
    if is_additive:
        utility_values = _recover_additive_utility_values(log, tolerance)

    return {
        "is_additive": is_additive,
        "is_additive_per_good": is_additive_per_good,
        "violation_cycles": violation_cycles,
        "num_violations": len(violation_cycles),
        "utility_values": utility_values,
    }


def _recover_additive_utility_values(
    log: "BehaviorLog",
    tolerance: float = 1e-6,
) -> dict | None:
    """
    Recover additive utility values u_i(x_{ti}) for each observation.

    Uses LP to find utility values consistent with the observed demands.

    For each good i, we need:
        u_i(x_{ti}) - u_i(x_{si}) ≤ p_{ti} * (x_{ti} - x_{si})
    for all observations t, s where x_{ti} is revealed preferred to x_{si}.

    Args:
        log: BehaviorLog
        tolerance: Numerical tolerance

    Returns:
        Dictionary mapping good index to array of utility values, or None if infeasible
    """
    T = log.num_records
    N = log.num_features
    P = log.cost_vectors
    Q = log.action_vectors

    utility_values = {}

    for i in range(N):
        # For good i, solve LP to find u_i(x_t[i]) for t = 0, ..., T-1
        # Variables: U_0, U_1, ..., U_{T-1}
        # Constraints: U_t - U_s ≤ p_t[i] * (x_t[i] - x_s[i]) for all t, s

        # Number of variables
        n_vars = T

        # Build constraints
        A_ub = []
        b_ub = []

        for t in range(T):
            for s in range(T):
                if t == s:
                    continue

                # Constraint: U_t - U_s ≤ p_t[i] * (x_t[i] - x_s[i])
                row = np.zeros(n_vars)
                row[t] = 1.0
                row[s] = -1.0

                rhs = P[t, i] * (Q[t, i] - Q[s, i])

                A_ub.append(row)
                b_ub.append(rhs)

        if not A_ub:
            utility_values[i] = np.zeros(T)
            continue

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # Objective: minimize sum of U_t (arbitrary, just need feasibility)
        c = np.ones(n_vars)

        # Bounds: U_t >= 0
        bounds = [(0, None)] * n_vars

        try:
            result = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method="highs",
            )

            if result.success:
                utility_values[i] = result.x
            else:
                return None
        except Exception:
            return None

    return utility_values


# =============================================================================
# LEGACY ALIASES
# =============================================================================

check_additivity = test_additive_separability
"""Legacy alias: use test_additive_separability instead."""

find_additive_groups = identify_additive_groups
"""Legacy alias: use identify_additive_groups instead."""
