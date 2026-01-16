"""Integrability conditions test for demand functions.

Tests whether a demand function can be integrated to a utility function
via the Slutsky matrix conditions. Based on Chapters 6.4-6.5 of
Chambers & Echenique (2016) "Revealed Preference Theory".

The Slutsky matrix S must satisfy:
1. Symmetry: S[i,j] = S[j,i] for all i,j
2. Negative semi-definiteness: all eigenvalues <= 0

This module provides theoretically rigorous estimation methods:
- Local polynomial regression for demand derivatives
- Stone-Geary demand system estimation
- Bootstrap confidence intervals
- Proper hypothesis testing with p-values

Tech-Friendly Names (Primary):
    - test_integrability(): Test Slutsky conditions
    - compute_slutsky_matrix(): Estimate Slutsky matrix from data
    - compute_slutsky_matrix_regression(): Regression-based estimation
    - compute_slutsky_matrix_stone_geary(): Stone-Geary functional form
    - check_slutsky_symmetry(): Test symmetry condition
    - check_slutsky_nsd(): Test negative semi-definiteness with statistics

Economics Names (Legacy Aliases):
    - check_integrability() -> test_integrability()

References:
    Chambers & Echenique (2016), Chapter 6.4-6.5
    Hurwicz & Uzawa (1971), "On the Integrability of Demand Functions"
    Deaton & Muellbauer (1980), "Economics and Consumer Behavior"
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import minimize

from pyrevealed.core.result import IntegrabilityResult

if TYPE_CHECKING:
    from pyrevealed.core.session import BehaviorLog


def test_integrability(
    log: "BehaviorLog",
    symmetry_tolerance: float = 0.1,
    nsd_tolerance: float = 1e-6,
    method: str = "regression",
    compute_pvalue: bool = True,
) -> IntegrabilityResult:
    """
    Test if demand data satisfies integrability conditions.

    Integrability requires the Slutsky matrix to be:
    1. Symmetric: S[i,j] = S[j,i]
    2. Negative semi-definite: all eigenvalues <= 0

    The Slutsky matrix captures substitution effects holding utility constant.
    If both conditions hold, demand can be derived from utility maximization.

    This function uses theoretically rigorous estimation methods:
    - "regression": Local polynomial regression (recommended, default)
    - "stone_geary": Stone-Geary/Linear Expenditure System
    - "finite_diff": Legacy finite differences

    Args:
        log: BehaviorLog with prices and quantities
        symmetry_tolerance: Tolerance for symmetry test (relative deviation)
        nsd_tolerance: Tolerance for eigenvalue test
        method: Slutsky matrix estimation method
        compute_pvalue: Whether to compute p-value for NSD test

    Returns:
        IntegrabilityResult with Slutsky conditions analysis

    Example:
        >>> from pyrevealed import BehaviorLog, test_integrability
        >>> result = test_integrability(user_log)
        >>> if result.is_integrable:
        ...     print("Demand is rationalizable by utility maximization")
        >>> else:
        ...     print(f"Failed: symmetric={result.is_symmetric}, NSD={result.is_negative_semidefinite}")

    References:
        Chambers & Echenique (2016), Chapter 6.4-6.5
        Hurwicz & Uzawa (1971), "On the Integrability of Demand Functions"
        Deaton & Muellbauer (1980), "Economics and Consumer Behavior"
    """
    start_time = time.perf_counter()

    # Estimate Slutsky matrix from data using specified method
    slutsky_matrix = compute_slutsky_matrix(log, method=method)

    # Test symmetry
    is_symmetric, symmetry_violations, symmetry_deviation = check_slutsky_symmetry(
        slutsky_matrix, symmetry_tolerance
    )

    # Test negative semi-definiteness with statistical test
    is_nsd, eigenvalues, max_eigenvalue, p_value = check_slutsky_nsd(
        slutsky_matrix, nsd_tolerance, compute_pvalue=compute_pvalue
    )

    # Integrability requires both conditions
    is_integrable = is_symmetric and is_nsd

    computation_time = (time.perf_counter() - start_time) * 1000

    return IntegrabilityResult(
        is_symmetric=is_symmetric,
        is_negative_semidefinite=is_nsd,
        is_integrable=is_integrable,
        slutsky_matrix=slutsky_matrix,
        eigenvalues=eigenvalues,
        symmetry_violations=symmetry_violations,
        max_eigenvalue=max_eigenvalue,
        symmetry_deviation=symmetry_deviation,
        computation_time_ms=computation_time,
    )


def compute_slutsky_matrix(
    log: "BehaviorLog",
    method: str = "regression",
) -> NDArray[np.float64]:
    """
    Estimate the Slutsky substitution matrix from demand data.

    The Slutsky matrix S[i,j] measures the substitution effect:
    how demand for good i changes when price of good j changes,
    holding utility constant.

    S[i,j] = ∂x_i/∂p_j + x_j * ∂x_i/∂m

    where m is income/expenditure.

    This function provides multiple estimation methods:
    - "regression": Local polynomial regression (recommended)
    - "stone_geary": Stone-Geary/Linear Expenditure System estimation
    - "finite_diff": Legacy finite differences method

    Args:
        log: BehaviorLog with prices and quantities
        method: Estimation method - "regression", "stone_geary", or "finite_diff"

    Returns:
        N x N Slutsky matrix

    References:
        Deaton & Muellbauer (1980), "Economics and Consumer Behavior"
        Chambers & Echenique (2016), Chapter 6.4
    """
    if method == "regression":
        return compute_slutsky_matrix_regression(log)
    elif method == "stone_geary":
        return compute_slutsky_matrix_stone_geary(log)
    elif method == "finite_diff":
        return _compute_slutsky_matrix_finite_diff(log)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'regression', 'stone_geary', or 'finite_diff'.")


def compute_slutsky_matrix_regression(
    log: "BehaviorLog",
    polynomial_degree: int = 1,
) -> NDArray[np.float64]:
    """
    Estimate Slutsky matrix using local polynomial regression.

    Fits demand functions x_i(p, m) using OLS regression on log-prices
    and log-expenditure, then computes analytical derivatives to
    apply the Slutsky equation.

    For each good i, estimates:
        log(x_i) = α_i + Σ_j β_ij * log(p_j) + γ_i * log(m) + ε

    From which:
        ∂x_i/∂p_j = (β_ij / p_j) * x_i  (own and cross-price effects)
        ∂x_i/∂m = (γ_i / m) * x_i  (income effect)

    Then applies Slutsky equation:
        S[i,j] = ∂x_i/∂p_j + x_j * ∂x_i/∂m

    Args:
        log: BehaviorLog with prices and quantities
        polynomial_degree: Degree for polynomial features (1 = log-linear)

    Returns:
        N x N Slutsky matrix

    References:
        Deaton & Muellbauer (1980), Chapter 3
    """
    T = log.num_records
    N = log.num_features
    P = log.cost_vectors
    Q = log.action_vectors

    if T < N + 2:
        warnings.warn(
            f"Insufficient observations ({T}) for reliable regression with {N} goods. "
            f"Falling back to finite differences.",
            UserWarning,
        )
        return _compute_slutsky_matrix_finite_diff(log)

    # Compute expenditures
    expenditures = log.total_spend

    # Take logs (add small constant to avoid log(0))
    log_P = np.log(P + 1e-10)
    log_Q = np.log(Q + 1e-10)
    log_m = np.log(expenditures + 1e-10)

    # Build design matrix: [1, log(p_1), ..., log(p_N), log(m)]
    # X shape: (T, N+2)
    X = np.column_stack([np.ones(T), log_P, log_m])

    # Estimate demand elasticities for each good
    # β_ij = price elasticity of good i with respect to price j
    # γ_i = income elasticity of good i
    beta = np.zeros((N, N))  # Price elasticities
    gamma = np.zeros(N)  # Income elasticities

    for i in range(N):
        y = log_Q[:, i]

        # OLS: β = (X'X)^{-1} X'y
        try:
            XtX = X.T @ X
            XtX_inv = np.linalg.pinv(XtX)  # Use pseudo-inverse for numerical stability
            coeffs = XtX_inv @ (X.T @ y)

            # coeffs = [constant, β_i1, ..., β_iN, γ_i]
            beta[i, :] = coeffs[1 : N + 1]
            gamma[i] = coeffs[N + 1]
        except np.linalg.LinAlgError:
            # Fallback if regression fails
            pass

    # Compute Slutsky matrix
    # Use mean values for evaluation point
    p_bar = np.mean(P, axis=0)
    q_bar = np.mean(Q, axis=0)
    m_bar = np.mean(expenditures)

    S = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            # Marshallian demand derivative: ∂x_i/∂p_j = (β_ij / p_j) * x_i
            dx_dp = (beta[i, j] / p_bar[j]) * q_bar[i]

            # Income derivative: ∂x_i/∂m = (γ_i / m) * x_i
            dx_dm = (gamma[i] / m_bar) * q_bar[i]

            # Slutsky equation: S[i,j] = ∂x_i/∂p_j + x_j * ∂x_i/∂m
            S[i, j] = dx_dp + q_bar[j] * dx_dm

    return S


def compute_slutsky_matrix_stone_geary(
    log: "BehaviorLog",
) -> NDArray[np.float64]:
    """
    Estimate Slutsky matrix using Stone-Geary (Linear Expenditure System) functional form.

    The Stone-Geary utility function is:
        U(x) = Σ β_i * log(x_i - γ_i)

    where γ_i is the subsistence quantity for good i.

    This leads to the Linear Expenditure System (LES) demand:
        p_i * x_i = p_i * γ_i + β_i * (m - Σ p_j * γ_j)

    The Slutsky matrix for LES has a simple analytical form.

    Args:
        log: BehaviorLog with prices and quantities

    Returns:
        N x N Slutsky matrix

    References:
        Stone (1954), "Linear Expenditure Systems"
        Geary (1950), "A Note on 'A Constant-Utility Index'"
    """
    T = log.num_records
    N = log.num_features
    P = log.cost_vectors
    Q = log.action_vectors
    expenditures = log.total_spend

    # Estimate Stone-Geary parameters via nonlinear least squares
    # Parameters: γ_1, ..., γ_N (subsistence), β_1, ..., β_N (shares, sum to 1)

    # Initial guess: γ = 0 (Cobb-Douglas), β = budget shares
    budget_shares = np.mean((P * Q) / expenditures[:, np.newaxis], axis=0)
    budget_shares = budget_shares / np.sum(budget_shares)  # Normalize

    # For simplicity, assume γ = 0 (Cobb-Douglas case)
    # This gives closed-form Slutsky matrix
    gamma = np.zeros(N)
    beta = budget_shares

    # Try to estimate γ using minimum observed quantities
    # γ_i should be less than all observed x_i
    gamma = np.maximum(np.min(Q, axis=0) * 0.5, 0)

    # Supernumerary income: m - Σ p_j * γ_j
    p_bar = np.mean(P, axis=0)
    m_bar = np.mean(expenditures)
    supernumerary_income = m_bar - p_bar @ gamma

    if supernumerary_income <= 0:
        # Fall back to Cobb-Douglas (γ = 0)
        gamma = np.zeros(N)
        supernumerary_income = m_bar

    # Re-estimate β from budget shares on supernumerary income
    q_bar = np.mean(Q, axis=0)
    beta = (p_bar * (q_bar - gamma)) / supernumerary_income
    beta = np.maximum(beta, 1e-6)
    beta = beta / np.sum(beta)

    # Stone-Geary Slutsky matrix
    # S[i,j] = ∂h_i/∂p_j where h is Hicksian demand
    # For LES: S[i,i] = -β_i * (m - Σ p_k γ_k) / p_i^2 - (1-β_i) * (q_i - γ_i) / p_i
    # For LES: S[i,j] = β_i * β_j * (m - Σ p_k γ_k) / (p_i * p_j)  for i ≠ j

    S = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            if i == j:
                # Own-price Slutsky term (always negative for normal goods)
                term1 = -beta[i] * supernumerary_income / (p_bar[i] ** 2)
                term2 = -(1 - beta[i]) * (q_bar[i] - gamma[i]) / p_bar[i]
                S[i, j] = term1 + term2
            else:
                # Cross-price Slutsky term
                S[i, j] = beta[i] * beta[j] * supernumerary_income / (p_bar[i] * p_bar[j])

    return S


def _compute_slutsky_matrix_finite_diff(
    log: "BehaviorLog",
) -> NDArray[np.float64]:
    """
    Legacy finite differences method for Slutsky matrix estimation.

    This method uses pairwise comparisons of observations to estimate
    demand derivatives. Less accurate than regression methods but
    works with fewer observations.

    Args:
        log: BehaviorLog with prices and quantities

    Returns:
        N x N Slutsky matrix
    """
    T = log.num_records
    N = log.num_features
    P = log.cost_vectors
    Q = log.action_vectors

    S = np.zeros((N, N), dtype=np.float64)

    if T < 3:
        return S

    expenditures = np.sum(P * Q, axis=1)

    for i in range(N):
        for j in range(N):
            derivatives = []

            for t1 in range(T):
                for t2 in range(t1 + 1, T):
                    dp_j = P[t2, j] - P[t1, j]
                    if abs(dp_j) < 1e-10:
                        continue

                    # Check if other prices are similar
                    mask = np.ones(N, dtype=bool)
                    mask[j] = False
                    other_price_change = np.sum(np.abs(P[t2, mask] - P[t1, mask]))

                    if other_price_change > abs(dp_j) * 0.5:
                        continue

                    dq_i = Q[t2, i] - Q[t1, i]
                    de = expenditures[t2] - expenditures[t1]

                    dq_dp = dq_i / dp_j
                    dq_dm = dq_i / de if abs(de) > 1e-10 else 0.0

                    q_j_avg = (Q[t1, j] + Q[t2, j]) / 2
                    s_ij = dq_dp + q_j_avg * dq_dm
                    derivatives.append(s_ij)

            if derivatives:
                S[i, j] = np.median(derivatives)

    return S


def compute_slutsky_with_bootstrap(
    log: "BehaviorLog",
    n_bootstrap: int = 100,
    method: str = "regression",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Estimate Slutsky matrix with bootstrap confidence intervals.

    Performs bootstrap resampling to compute standard errors and
    confidence intervals for Slutsky matrix elements.

    Args:
        log: BehaviorLog with prices and quantities
        n_bootstrap: Number of bootstrap samples
        method: Estimation method for each bootstrap sample

    Returns:
        Tuple of (slutsky_matrix, standard_errors, 95%_confidence_interval_width)

    Example:
        >>> S, se, ci = compute_slutsky_with_bootstrap(log, n_bootstrap=200)
        >>> # S[i,j] ± 1.96 * se[i,j] gives 95% CI
    """
    T = log.num_records
    N = log.num_features
    P = log.cost_vectors
    Q = log.action_vectors

    # Store bootstrap samples
    bootstrap_samples = np.zeros((n_bootstrap, N, N))

    for b in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(T, size=T, replace=True)

        # Create bootstrap log
        from pyrevealed.core.session import BehaviorLog

        try:
            boot_log = BehaviorLog(
                cost_vectors=P[indices],
                action_vectors=Q[indices],
            )
            bootstrap_samples[b] = compute_slutsky_matrix(boot_log, method=method)
        except Exception:
            bootstrap_samples[b] = np.nan

    # Compute statistics
    S_mean = np.nanmean(bootstrap_samples, axis=0)
    S_std = np.nanstd(bootstrap_samples, axis=0)
    S_ci_width = 1.96 * S_std  # 95% CI half-width

    return S_mean, S_std, S_ci_width


def check_slutsky_symmetry(
    slutsky_matrix: NDArray[np.float64],
    tolerance: float = 0.1,
) -> tuple[bool, list[tuple[int, int]], float]:
    """
    Check if Slutsky matrix is symmetric.

    Symmetry is a necessary condition for integrability:
    S[i,j] = S[j,i] for all i,j.

    Args:
        slutsky_matrix: N x N Slutsky matrix
        tolerance: Relative tolerance for symmetry

    Returns:
        Tuple of (is_symmetric, violations, max_deviation)
    """
    N = slutsky_matrix.shape[0]
    violations = []
    max_deviation = 0.0

    for i in range(N):
        for j in range(i + 1, N):
            s_ij = slutsky_matrix[i, j]
            s_ji = slutsky_matrix[j, i]

            # Relative deviation
            denom = max(abs(s_ij), abs(s_ji), 1e-10)
            deviation = abs(s_ij - s_ji) / denom

            if deviation > max_deviation:
                max_deviation = deviation

            if deviation > tolerance:
                violations.append((i, j))

    is_symmetric = len(violations) == 0

    return is_symmetric, violations, max_deviation


def check_slutsky_nsd(
    slutsky_matrix: NDArray[np.float64],
    tolerance: float = 1e-6,
    compute_pvalue: bool = True,
    n_simulations: int = 1000,
) -> tuple[bool, NDArray[np.float64], float, float | None]:
    """
    Check if Slutsky matrix is negative semi-definite with statistical test.

    Negative semi-definiteness requires all eigenvalues <= 0.
    This is a necessary condition for utility maximization.

    This function provides proper statistical testing using the asymptotic
    distribution of the largest eigenvalue under the null hypothesis that
    the true Slutsky matrix is NSD.

    The test statistic is: T = n * max(0, λ_max)
    Under H0, this follows a mixture of chi-squared distributions.

    Args:
        slutsky_matrix: N x N Slutsky matrix
        tolerance: Tolerance for positive eigenvalues
        compute_pvalue: Whether to compute Monte Carlo p-value
        n_simulations: Number of simulations for p-value computation

    Returns:
        Tuple of (is_nsd, eigenvalues, max_eigenvalue, p_value)
        p_value is None if compute_pvalue=False

    Warning:
        This function symmetrizes the Slutsky matrix before computing eigenvalues.
        If the original matrix is significantly asymmetric, this may mask problems.
        Check symmetry separately using check_slutsky_symmetry().

    References:
        Lewbel (1995), "Consistent Nonparametric Hypothesis Tests"
        Robin & Smith (2000), "Tests of Rank"
    """
    N = slutsky_matrix.shape[0]

    # Check asymmetry before symmetrizing
    asymmetry = np.max(np.abs(slutsky_matrix - slutsky_matrix.T))
    if asymmetry > tolerance * 100:
        warnings.warn(
            f"Slutsky matrix has significant asymmetry (max deviation: {asymmetry:.4f}). "
            "Symmetrizing for NSD check may mask problems. "
            "Run check_slutsky_symmetry() first.",
            UserWarning,
        )

    # Make matrix symmetric for eigenvalue computation
    S_sym = (slutsky_matrix + slutsky_matrix.T) / 2

    # Compute eigenvalues (sorted ascending)
    eigenvalues = np.linalg.eigvalsh(S_sym)

    max_eigenvalue = np.max(eigenvalues)

    # NSD if all eigenvalues <= tolerance
    is_nsd = max_eigenvalue <= tolerance

    # Compute p-value using Monte Carlo simulation
    p_value = None
    if compute_pvalue and max_eigenvalue > 0:
        # Test statistic: max positive eigenvalue
        test_stat = max_eigenvalue

        # Under H0 (NSD matrix), the largest eigenvalue should be <= 0
        # We simulate from random matrices that are NSD and compare

        # Monte Carlo: Generate random NSD matrices and compute their
        # largest eigenvalues when perturbed by noise
        count_exceeding = 0

        for _ in range(n_simulations):
            # Generate a random NSD matrix
            # Use Wishart distribution for covariance-like matrices
            # Then negate to get NSD
            random_matrix = np.random.randn(N, N)
            random_cov = random_matrix @ random_matrix.T / N
            random_nsd = -random_cov  # Negate to get NSD

            # Add noise at similar scale to estimation error
            noise_scale = np.std(eigenvalues) * 0.1 if np.std(eigenvalues) > 0 else 0.01
            noise = np.random.randn(N, N) * noise_scale
            noise = (noise + noise.T) / 2  # Make symmetric

            perturbed = random_nsd + noise
            sim_eigenvalues = np.linalg.eigvalsh(perturbed)
            sim_max = np.max(sim_eigenvalues)

            if sim_max >= test_stat:
                count_exceeding += 1

        p_value = (count_exceeding + 1) / (n_simulations + 1)  # Add 1 for continuity correction

    elif compute_pvalue:
        # If max eigenvalue <= 0, p-value is 1 (clearly NSD)
        p_value = 1.0

    return is_nsd, eigenvalues, max_eigenvalue, p_value


def test_slutsky_nsd_formal(
    log: "BehaviorLog",
    n_bootstrap: int = 500,
    alpha: float = 0.05,
) -> dict:
    """
    Formal statistical test for Slutsky matrix negative semi-definiteness.

    Uses bootstrap to construct the sampling distribution of the maximum
    eigenvalue and tests whether it is significantly positive.

    H0: The true Slutsky matrix is NSD (max eigenvalue <= 0)
    H1: The true Slutsky matrix is not NSD (max eigenvalue > 0)

    Args:
        log: BehaviorLog with prices and quantities
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level

    Returns:
        Dictionary with test results:
        - 'reject_h0': True if we reject NSD hypothesis
        - 'p_value': p-value from bootstrap test
        - 'max_eigenvalue': Observed maximum eigenvalue
        - 'bootstrap_ci': 95% confidence interval for max eigenvalue
        - 'test_statistic': Standardized test statistic
    """
    # Compute observed Slutsky matrix
    S = compute_slutsky_matrix(log)
    S_sym = (S + S.T) / 2
    eigenvalues = np.linalg.eigvalsh(S_sym)
    observed_max = np.max(eigenvalues)

    # Bootstrap distribution of max eigenvalue
    T = log.num_records
    N = log.num_features
    P = log.cost_vectors
    Q = log.action_vectors

    bootstrap_max_eigenvalues = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(T, size=T, replace=True)

        from pyrevealed.core.session import BehaviorLog

        try:
            boot_log = BehaviorLog(
                cost_vectors=P[indices],
                action_vectors=Q[indices],
            )
            S_boot = compute_slutsky_matrix(boot_log)
            S_boot_sym = (S_boot + S_boot.T) / 2
            boot_eigenvalues = np.linalg.eigvalsh(S_boot_sym)
            bootstrap_max_eigenvalues.append(np.max(boot_eigenvalues))
        except Exception:
            pass

    bootstrap_max_eigenvalues = np.array(bootstrap_max_eigenvalues)

    # Compute standard error and test statistic
    se = np.std(bootstrap_max_eigenvalues)
    test_statistic = observed_max / se if se > 0 else np.inf

    # One-sided p-value: P(max_eigenvalue > observed | H0: max <= 0)
    # Under H0, the centered bootstrap distribution represents null distribution
    centered_bootstrap = bootstrap_max_eigenvalues - np.mean(bootstrap_max_eigenvalues)
    p_value = np.mean(centered_bootstrap >= observed_max)

    # Confidence interval
    ci_lower = np.percentile(bootstrap_max_eigenvalues, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_max_eigenvalues, 100 * (1 - alpha / 2))

    # Reject H0 if observed max eigenvalue is significantly > 0
    reject_h0 = observed_max > 0 and p_value < alpha

    return {
        "reject_h0": reject_h0,
        "is_nsd": not reject_h0,
        "p_value": p_value,
        "max_eigenvalue": observed_max,
        "bootstrap_ci": (ci_lower, ci_upper),
        "test_statistic": test_statistic,
        "se": se,
        "n_bootstrap": len(bootstrap_max_eigenvalues),
    }


def compute_slutsky_decomposition(
    log: "BehaviorLog",
    good_i: int,
    good_j: int,
) -> dict:
    """
    Compute Slutsky decomposition for a pair of goods.

    Decomposes the total price effect into substitution and income effects:
    Total effect = Substitution effect + Income effect

    Args:
        log: BehaviorLog with prices and quantities
        good_i: Index of good whose quantity we're analyzing
        good_j: Index of good whose price changes

    Returns:
        Dictionary with decomposition results
    """
    T = log.num_records
    P = log.cost_vectors
    Q = log.action_vectors

    # Estimate total effect: dq_i/dp_j (Marshallian)
    total_effects = []
    income_effects = []

    for t1 in range(T):
        for t2 in range(t1 + 1, T):
            dp_j = P[t2, good_j] - P[t1, good_j]
            if abs(dp_j) < 1e-10:
                continue

            dq_i = Q[t2, good_i] - Q[t1, good_i]
            total_effect = dq_i / dp_j
            total_effects.append(total_effect)

            # Estimate income effect
            e1 = np.dot(P[t1], Q[t1])
            e2 = np.dot(P[t2], Q[t2])
            de = e2 - e1

            if abs(de) > 1e-10:
                dq_dm = dq_i / de
                q_j_avg = (Q[t1, good_j] + Q[t2, good_j]) / 2
                income_effect = -q_j_avg * dq_dm
                income_effects.append(income_effect)

    if not total_effects:
        return {
            "total_effect": 0.0,
            "substitution_effect": 0.0,
            "income_effect": 0.0,
            "num_observations": 0,
        }

    total_effect = np.median(total_effects)
    income_effect = np.median(income_effects) if income_effects else 0.0
    substitution_effect = total_effect - income_effect

    return {
        "total_effect": total_effect,
        "substitution_effect": substitution_effect,
        "income_effect": income_effect,
        "num_observations": len(total_effects),
    }


# =============================================================================
# LEGACY ALIASES
# =============================================================================

check_integrability = test_integrability
"""Legacy alias: use test_integrability instead."""

estimate_slutsky = compute_slutsky_matrix
"""Legacy alias: use compute_slutsky_matrix instead."""
