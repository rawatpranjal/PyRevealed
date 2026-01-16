"""Stochastic choice and random utility models.

Implements probabilistic choice models including logit, Luce model,
and random utility maximization (RUM). Based on Chapter 13 of
Chambers & Echenique (2016) "Revealed Preference Theory".

Tech-Friendly Names (Primary):
    - fit_random_utility_model(): Fit RUM to stochastic choice data
    - test_mcfadden_axioms(): Test IIA and regularity conditions
    - estimate_choice_probabilities(): Predict choice probabilities
    - check_independence_irrelevant_alternatives(): Test IIA

Economics Names (Legacy Aliases):
    - fit_rum() -> fit_random_utility_model()
    - check_iia() -> check_independence_irrelevant_alternatives()
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from pyrevealed.core.result import StochasticChoiceResult

if TYPE_CHECKING:
    from pyrevealed.core.session import StochasticChoiceLog, MenuChoiceLog


def fit_random_utility_model(
    log: "StochasticChoiceLog",
    model_type: str = "logit",
    max_iterations: int = 1000,
) -> StochasticChoiceResult:
    """
    Fit a random utility model to stochastic choice data.

    Random utility models assume the consumer has utility U_i = V_i + epsilon_i
    where V_i is deterministic and epsilon_i is random. Different assumptions
    about epsilon distribution lead to different models:
    - Logit: epsilon ~ Gumbel (IIA holds)
    - Probit: epsilon ~ Normal
    - Luce: probability proportional to utility

    Args:
        log: StochasticChoiceLog with choice frequency data
        model_type: Type of model ("logit", "probit", "luce")
        max_iterations: Maximum optimization iterations

    Returns:
        StochasticChoiceResult with model parameters and fit statistics

    Example:
        >>> from pyrevealed import StochasticChoiceLog, fit_random_utility_model
        >>> result = fit_random_utility_model(choice_data, model_type="logit")
        >>> print(f"Model: {result.model_type}")
        >>> print(f"Satisfies IIA: {result.satisfies_iia}")
        >>> print(f"Log-likelihood: {result.log_likelihood:.2f}")

    References:
        Chambers & Echenique (2016), Chapter 13
        McFadden, D. (1974). "Conditional Logit Analysis of Qualitative Choice Behavior"
    """
    start_time = time.perf_counter()

    n_menus = log.num_menus

    # Estimate item utilities
    if model_type == "logit":
        utilities, parameters = _fit_logit_model(log, max_iterations)
    elif model_type == "luce":
        utilities, parameters = _fit_luce_model(log)
    else:
        # Default to logit
        utilities, parameters = _fit_logit_model(log, max_iterations)

    # Compute predicted choice probabilities
    choice_probabilities = _compute_choice_probabilities(
        log, utilities, model_type
    )

    # Compute log-likelihood
    log_likelihood = _compute_log_likelihood(log, choice_probabilities)

    # Compute AIC and BIC
    n_params = len(utilities)
    n_obs = sum(log.total_observations_per_menu)
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(n_obs) * n_params - 2 * log_likelihood

    # Test IIA (Independence of Irrelevant Alternatives)
    satisfies_iia = check_independence_irrelevant_alternatives(log)

    # Test regularity (monotonicity)
    regularity_violations = _find_regularity_violations(log)

    computation_time = (time.perf_counter() - start_time) * 1000

    return StochasticChoiceResult(
        model_type=model_type,
        parameters=parameters,
        satisfies_iia=satisfies_iia,
        choice_probabilities=choice_probabilities,
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        regularity_violations=regularity_violations,
        computation_time_ms=computation_time,
    )


def test_mcfadden_axioms(
    log: "StochasticChoiceLog",
) -> dict:
    """
    Test McFadden's axioms for random utility maximization.

    The axioms include:
    1. Regularity: P(x|A) >= P(x|B) when A ⊆ B (removing options doesn't decrease choice probability)
    2. IIA: P(x|A)/P(y|A) = P(x|B)/P(y|B) for all A,B containing x,y

    Args:
        log: StochasticChoiceLog with choice frequency data

    Returns:
        Dictionary with axiom test results
    """
    satisfies_iia = check_independence_irrelevant_alternatives(log)
    regularity_violations = _find_regularity_violations(log)
    satisfies_regularity = len(regularity_violations) == 0

    return {
        "satisfies_iia": satisfies_iia,
        "satisfies_regularity": satisfies_regularity,
        "regularity_violations": regularity_violations,
        "is_rum_consistent": satisfies_iia and satisfies_regularity,
    }


def check_independence_irrelevant_alternatives(
    log: "StochasticChoiceLog",
    tolerance: float = 0.1,
) -> bool:
    """
    Test Independence of Irrelevant Alternatives (IIA).

    IIA states that the relative odds of choosing x over y should not
    depend on what other alternatives are available:
    P(x|A) / P(y|A) = P(x|B) / P(y|B) for all menus A, B containing both x and y.

    Args:
        log: StochasticChoiceLog with choice frequency data
        tolerance: Tolerance for ratio comparison

    Returns:
        True if IIA approximately holds

    Note:
        IIA is a strong condition that often fails in practice
        (e.g., red bus/blue bus paradox).
    """
    n_menus = log.num_menus

    # For each pair of items, check if odds ratio is consistent across menus
    items = sorted(log.all_items)

    for x in items:
        for y in items:
            if x >= y:
                continue

            odds_ratios = []

            for m_idx in range(n_menus):
                menu = log.menus[m_idx]
                if x in menu and y in menu:
                    p_x = log.get_choice_probability(m_idx, x)
                    p_y = log.get_choice_probability(m_idx, y)

                    if p_y > 1e-10:
                        ratio = p_x / p_y
                        odds_ratios.append(ratio)

            # Check if odds ratios are consistent
            if len(odds_ratios) >= 2:
                cv = np.std(odds_ratios) / max(np.mean(odds_ratios), 1e-10)
                if cv > tolerance:
                    return False

    return True


def estimate_choice_probabilities(
    log: "StochasticChoiceLog",
    utilities: NDArray[np.float64],
    model_type: str = "logit",
) -> NDArray[np.float64]:
    """
    Estimate choice probabilities given utilities.

    Args:
        log: StochasticChoiceLog with menu structure
        utilities: Array of item utilities
        model_type: Type of model

    Returns:
        Array of choice probabilities (flattened)
    """
    return _compute_choice_probabilities(log, utilities, model_type)


def fit_luce_model(
    log: "StochasticChoiceLog",
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """
    Fit Luce choice model to stochastic choice data.

    The Luce model (also called Bradley-Terry) assumes:
    P(x|A) = v(x) / Σ_{y ∈ A} v(y)

    where v(x) is the "choice value" of item x.

    Args:
        log: StochasticChoiceLog with choice frequency data

    Returns:
        Tuple of (utilities, parameters)
    """
    return _fit_luce_model(log)


def _fit_logit_model(
    log: "StochasticChoiceLog",
    max_iterations: int = 1000,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """
    Fit multinomial logit model using MLE.
    """
    n_items = max(log.all_items) + 1

    # Initial utilities
    utilities = np.zeros(n_items)

    # Objective: negative log-likelihood
    def neg_log_likelihood(u: NDArray[np.float64]) -> float:
        ll = 0.0
        for m_idx in range(log.num_menus):
            menu = log.menus[m_idx]
            freqs = log.choice_frequencies[m_idx]
            total = log.total_observations_per_menu[m_idx]

            if total == 0:
                continue

            # Validate non-empty menu
            if len(menu) == 0:
                continue

            # Compute choice probabilities using log-sum-exp trick for numerical stability
            # This prevents overflow when utilities are large
            menu_arr = np.array(list(menu))
            u_menu = u[menu_arr]
            max_u = np.max(u_menu)  # Subtract max for numerical stability
            exp_u = np.exp(u_menu - max_u)
            log_sum_exp = max_u + np.log(np.sum(exp_u))

            for item, count in freqs.items():
                if count > 0:
                    # log(p) = u[item] - log_sum_exp
                    log_p = u[item] - log_sum_exp
                    ll += count * log_p

        return -ll

    # Optimize
    result = minimize(
        neg_log_likelihood,
        utilities,
        method="BFGS",
        options={"maxiter": max_iterations},
    )

    utilities = result.x

    # Normalize so minimum utility is 0
    utilities = utilities - np.min(utilities)

    parameters = {
        "scale": 1.0,  # Logit scale parameter
        "convergence": float(result.success),
    }

    return utilities, parameters


def _fit_luce_model(
    log: "StochasticChoiceLog",
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """
    Fit Luce choice model using simple frequency-based estimation.
    """
    n_items = max(log.all_items) + 1

    # Estimate v(x) from choice frequencies
    # Use: v(x) ∝ average choice probability across menus containing x
    choice_counts = np.zeros(n_items)
    appearance_counts = np.zeros(n_items)

    for m_idx in range(log.num_menus):
        menu = log.menus[m_idx]
        freqs = log.choice_frequencies[m_idx]
        total = log.total_observations_per_menu[m_idx]

        for item in menu:
            appearance_counts[item] += total
            choice_counts[item] += freqs.get(item, 0)

    # Estimate utilities as log of choice values
    utilities = np.zeros(n_items)
    for i in range(n_items):
        if appearance_counts[i] > 0:
            v_i = choice_counts[i] / appearance_counts[i]
            utilities[i] = np.log(max(v_i, 1e-10))
        else:
            utilities[i] = -10.0  # Very low utility for unseen items

    # Normalize
    utilities = utilities - np.min(utilities)

    parameters = {
        "method": "frequency_based",
    }

    return utilities, parameters


def _compute_choice_probabilities(
    log: "StochasticChoiceLog",
    utilities: NDArray[np.float64],
    model_type: str,
) -> NDArray[np.float64]:
    """
    Compute choice probabilities for all menus.

    Uses log-sum-exp trick for numerical stability.
    """
    all_probs = []

    for m_idx in range(log.num_menus):
        menu = log.menus[m_idx]

        # Handle empty menus
        if len(menu) == 0:
            continue

        menu_arr = np.array(list(menu))
        u_menu = utilities[menu_arr]

        if model_type == "logit" or model_type == "luce":
            # Use log-sum-exp trick for numerical stability
            max_u = np.max(u_menu)
            exp_u = np.exp(u_menu - max_u)
            sum_exp_u = np.sum(exp_u)
            probs = exp_u / sum_exp_u
        else:
            # Default to logit with log-sum-exp
            max_u = np.max(u_menu)
            exp_u = np.exp(u_menu - max_u)
            sum_exp_u = np.sum(exp_u)
            probs = exp_u / sum_exp_u

        all_probs.extend(probs)

    return np.array(all_probs)


def _compute_log_likelihood(
    log: "StochasticChoiceLog",
    choice_probabilities: NDArray[np.float64],
) -> float:
    """
    Compute log-likelihood of the model.
    """
    ll = 0.0
    prob_idx = 0

    for m_idx in range(log.num_menus):
        menu = list(log.menus[m_idx])
        freqs = log.choice_frequencies[m_idx]

        for item in menu:
            count = freqs.get(item, 0)
            if count > 0:
                p = choice_probabilities[prob_idx]
                ll += count * np.log(max(p, 1e-10))
            prob_idx += 1

    return ll


def _find_regularity_violations(
    log: "StochasticChoiceLog",
    tolerance: float = 0.01,
) -> list[int]:
    """
    Find observations that violate regularity.

    Regularity: if A ⊆ B, then P(x|A) >= P(x|B) for all x ∈ A.
    (Removing options should not decrease choice probability.)

    Args:
        log: StochasticChoiceLog with choice frequency data
        tolerance: Tolerance for probability comparison (default 0.01)
    """
    violations = []
    n_menus = log.num_menus

    for m1 in range(n_menus):
        for m2 in range(n_menus):
            if m1 == m2:
                continue

            menu1 = log.menus[m1]
            menu2 = log.menus[m2]

            # Check if menu1 ⊆ menu2
            if menu1.issubset(menu2):
                # For each item in menu1, P(x|menu1) should >= P(x|menu2)
                for item in menu1:
                    p1 = log.get_choice_probability(m1, item)
                    p2 = log.get_choice_probability(m2, item)

                    if p1 < p2 - tolerance:
                        violations.append(m1)
                        break

    return list(set(violations))


def fit_from_deterministic(
    log: "MenuChoiceLog",
    model_type: str = "logit",
) -> StochasticChoiceResult:
    """
    Fit a stochastic model to deterministic choice data.

    Treats each deterministic choice as a single observation and
    aggregates by menu to create stochastic choice data.

    Args:
        log: MenuChoiceLog with deterministic choices
        model_type: Type of stochastic model to fit

    Returns:
        StochasticChoiceResult with fitted model
    """
    from pyrevealed.core.session import StochasticChoiceLog

    # Convert to stochastic format
    stochastic_log = StochasticChoiceLog.from_repeated_choices(
        log.menus, log.choices
    )

    return fit_random_utility_model(stochastic_log, model_type)


# =============================================================================
# LEGACY ALIASES
# =============================================================================

fit_rum = fit_random_utility_model
"""Legacy alias: use fit_random_utility_model instead."""

check_iia = check_independence_irrelevant_alternatives
"""Legacy alias: use check_independence_irrelevant_alternatives instead."""

test_regularity = test_mcfadden_axioms
"""Legacy alias: use test_mcfadden_axioms instead."""
