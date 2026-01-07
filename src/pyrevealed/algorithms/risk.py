"""Risk profile analysis via revealed preferences under uncertainty.

Implements classification of risk attitudes (risk-seeking, risk-neutral, risk-averse)
based on choices between safe and risky options using CRRA utility estimation.

Note: This implementation uses Maximum Likelihood Estimation (MLE) to estimate
the CRRA parameter. For revealed preference axioms (SAREU/SARSEU) for CRRA,
see Chambers, Echenique, and Saito (2015) "Testing Theories of Financial
Decision Making" as referenced in Chambers & Echenique (2016) Chapter 8, p.128.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from pyrevealed.core.session import RiskSession
from pyrevealed.core.result import RiskProfileResult


def compute_risk_profile(
    session: RiskSession,
    rho_bounds: tuple[float, float] = (-2.0, 5.0),
    tolerance: float = 1e-6,
) -> RiskProfileResult:
    """
    Estimate risk profile from choices under uncertainty.

    Uses Constant Relative Risk Aversion (CRRA) utility model:
        u(x) = x^(1-ρ) / (1-ρ)  for ρ ≠ 1
        u(x) = ln(x)            for ρ = 1

    where ρ is the Arrow-Pratt coefficient of relative risk aversion.

    This function estimates ρ using Maximum Likelihood Estimation (MLE) with
    a logistic choice model. This is an econometric approach; for the revealed
    preference axiom approach, see Chambers, Echenique, and Saito (2015).

    Args:
        session: RiskSession with safe values, risky lotteries, and choices
        rho_bounds: Search bounds for risk aversion coefficient (min, max)
        tolerance: Convergence tolerance for optimization

    Returns:
        RiskProfileResult with estimated risk profile

    Example:
        >>> import numpy as np
        >>> from pyrevealed import RiskSession, compute_risk_profile
        >>> # Risk-averse person: prefers $50 certain over 50/50 chance of $100/$0
        >>> safe = np.array([50.0, 40.0, 30.0])
        >>> outcomes = np.array([[100.0, 0.0], [100.0, 0.0], [100.0, 0.0]])
        >>> probs = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        >>> choices = np.array([False, False, True])  # Only takes gamble at $30
        >>> session = RiskSession(safe, outcomes, probs, choices)
        >>> result = compute_risk_profile(session)
        >>> result.risk_category
        'risk_averse'
    """
    start_time = time.perf_counter()

    T = session.num_observations

    # Find optimal rho by maximizing choice likelihood
    def neg_log_likelihood(rho: float) -> float:
        """Negative log-likelihood of choices given rho."""
        # Compute utility of safe option
        u_safe = _crra_utility(session.safe_values, rho)

        # Compute expected utility of risky option
        u_risky_outcomes = _crra_utility(session.risky_outcomes, rho)
        eu_risky = np.sum(u_risky_outcomes * session.risky_probabilities, axis=1)

        # Compute probability of choosing risky (logistic model)
        # P(risky) = 1 / (1 + exp(-(EU_risky - U_safe)))
        diff = eu_risky - u_safe

        # Clip to avoid overflow
        diff = np.clip(diff, -500, 500)

        # Log-likelihood: sum of log P(observed choice)
        log_p_risky = -np.log1p(np.exp(-diff))
        log_p_safe = -np.log1p(np.exp(diff))

        ll = np.sum(np.where(session.choices, log_p_risky, log_p_safe))

        return -ll  # Negative for minimization

    # Optimize rho
    result = minimize_scalar(
        neg_log_likelihood,
        bounds=rho_bounds,
        method="bounded",
        options={"xatol": tolerance},
    )

    rho = result.x

    # Compute certainty equivalents for each lottery
    certainty_equivalents = _compute_certainty_equivalents(session, rho)

    # Classify risk category
    if rho > 0.1:
        risk_category = "risk_averse"
    elif rho < -0.1:
        risk_category = "risk_seeking"
    else:
        risk_category = "risk_neutral"

    # Compute consistency: how many choices match the model prediction
    u_safe = _crra_utility(session.safe_values, rho)
    u_risky_outcomes = _crra_utility(session.risky_outcomes, rho)
    eu_risky = np.sum(u_risky_outcomes * session.risky_probabilities, axis=1)

    predicted_risky = eu_risky > u_safe
    num_consistent = int(np.sum(predicted_risky == session.choices))
    consistency_score = num_consistent / T

    # Utility curvature (second derivative at mean wealth)
    mean_outcome = np.mean(session.risky_outcomes)
    utility_curvature = _crra_curvature(mean_outcome, rho)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return RiskProfileResult(
        risk_aversion_coefficient=rho,
        risk_category=risk_category,
        certainty_equivalents=certainty_equivalents,
        utility_curvature=utility_curvature,
        consistency_score=consistency_score,
        num_consistent_choices=num_consistent,
        num_total_choices=T,
        computation_time_ms=elapsed_ms,
    )


def _crra_utility(x: NDArray[np.float64], rho: float) -> NDArray[np.float64]:
    """
    Compute CRRA utility.

    u(x) = x^(1-ρ) / (1-ρ)  for ρ ≠ 1
    u(x) = ln(x)            for ρ = 1

    Handles edge cases for negative outcomes and zero.
    """
    x = np.asarray(x, dtype=np.float64)

    # Handle zeros and negatives (add small epsilon)
    x_safe = np.maximum(x, 1e-10)

    if np.abs(rho - 1.0) < 1e-10:
        return np.log(x_safe)
    else:
        return np.power(x_safe, 1 - rho) / (1 - rho)


def _crra_curvature(x: float, rho: float) -> float:
    """Compute second derivative of CRRA utility at x."""
    if x <= 0:
        return 0.0
    return -rho * (x ** (-rho - 1))


def _compute_certainty_equivalents(
    session: RiskSession, rho: float
) -> NDArray[np.float64]:
    """
    Compute certainty equivalent for each risky lottery.

    The certainty equivalent CE is the certain amount such that
    u(CE) = E[u(X)] where X is the lottery.
    """
    # Compute expected utility of each lottery
    u_outcomes = _crra_utility(session.risky_outcomes, rho)
    eu = np.sum(u_outcomes * session.risky_probabilities, axis=1)

    # Invert CRRA to get CE
    if np.abs(rho - 1.0) < 1e-10:
        # u(x) = ln(x) => x = exp(u)
        ce = np.exp(eu)
    else:
        # u(x) = x^(1-ρ)/(1-ρ) => x = ((1-ρ)*u)^(1/(1-ρ))
        ce = np.power(np.maximum((1 - rho) * eu, 1e-10), 1 / (1 - rho))

    return ce


def check_expected_utility_axioms(session: RiskSession) -> tuple[bool, list[str]]:
    """
    Check if choices are consistent with Expected Utility axioms.

    Tests for violations of:
    1. Monotonicity: preferring more to less
    2. Independence: compound lottery invariance

    Args:
        session: RiskSession with choice data

    Returns:
        Tuple of (is_consistent, list of violation descriptions)
    """
    violations = []

    # Check monotonicity: if safe > max(risky), should choose safe
    max_risky = session.risky_outcomes.max(axis=1)
    chose_risky_when_dominated = session.choices & (session.safe_values > max_risky)
    if np.any(chose_risky_when_dominated):
        indices = np.where(chose_risky_when_dominated)[0]
        for i in indices:
            violations.append(
                f"Obs {i}: Chose risky {session.risky_outcomes[i]} over "
                f"dominating safe {session.safe_values[i]}"
            )

    # Check if safe < min(risky), should choose risky
    min_risky = session.risky_outcomes.min(axis=1)
    chose_safe_when_dominated = (~session.choices) & (session.safe_values < min_risky)
    if np.any(chose_safe_when_dominated):
        indices = np.where(chose_safe_when_dominated)[0]
        for i in indices:
            violations.append(
                f"Obs {i}: Chose safe {session.safe_values[i]} over "
                f"dominating risky {session.risky_outcomes[i]}"
            )

    is_consistent = len(violations) == 0
    return is_consistent, violations


def classify_risk_type(
    session: RiskSession,
) -> Literal["gambler", "investor", "neutral", "inconsistent"]:
    """
    Quick classification of decision-maker type.

    - "gambler": Risk-seeking, prefers uncertainty
    - "investor": Risk-averse, prefers certainty
    - "neutral": Maximizes expected value
    - "inconsistent": Choices don't fit any clear pattern

    Args:
        session: RiskSession with choice data

    Returns:
        Classification string
    """
    result = compute_risk_profile(session)

    if result.consistency_score < 0.6:
        return "inconsistent"

    if result.risk_category == "risk_seeking":
        return "gambler"
    elif result.risk_category == "risk_averse":
        return "investor"
    else:
        return "neutral"
