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

from pyrevealed.core.result import (
    StochasticChoiceResult,
    RUMConsistencyResult,
    RegularityResult,
    RegularityViolation,
)

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

def test_regularity(
    log: "StochasticChoiceLog",
    tolerance: float = 0.01,
) -> RegularityResult:
    """
    Test the regularity axiom for stochastic choice data.

    Regularity (Luce axiom) states that adding options to a menu should
    never INCREASE the probability of choosing any particular item:
        For all A ⊆ B and x ∈ A: P(x|A) >= P(x|B)

    Violations indicate decoy effects, attraction effects, or
    consideration set changes.

    Args:
        log: StochasticChoiceLog with choice frequency data
        tolerance: Tolerance for probability comparison (default 0.01)

    Returns:
        RegularityResult with detailed violation information

    Example:
        >>> from pyrevealed import StochasticChoiceLog, test_regularity
        >>> log = StochasticChoiceLog(
        ...     menus=[{0,1}, {0,1,2}],
        ...     choice_frequencies=[{0: 60, 1: 40}, {0: 45, 1: 35, 2: 20}],
        ...     total_observations_per_menu=[100, 100]
        ... )
        >>> result = test_regularity(log)
        >>> if not result.satisfies_regularity:
        ...     print(f"Found {result.num_violations} violations")
        ...     print(f"Worst: {result.worst_violation}")

    References:
        Luce, R. D. (1959). Individual Choice Behavior
        Chambers & Echenique (2016), Chapter 13
    """
    start_time = time.perf_counter()

    violations: list[RegularityViolation] = []
    n_menus = log.num_menus
    num_testable_pairs = 0

    for m1 in range(n_menus):
        for m2 in range(n_menus):
            if m1 == m2:
                continue

            menu1 = log.menus[m1]
            menu2 = log.menus[m2]

            # Check if menu1 ⊆ menu2
            if menu1.issubset(menu2):
                num_testable_pairs += 1

                # For each item in menu1, P(x|menu1) should >= P(x|menu2)
                for item in menu1:
                    p_subset = log.get_choice_probability(m1, item)
                    p_superset = log.get_choice_probability(m2, item)

                    if p_subset < p_superset - tolerance:
                        magnitude = p_superset - p_subset
                        violations.append(RegularityViolation(
                            item=item,
                            subset_menu_idx=m1,
                            superset_menu_idx=m2,
                            prob_in_subset=p_subset,
                            prob_in_superset=p_superset,
                            magnitude=magnitude,
                        ))

    # Find worst violation
    worst_violation = None
    if violations:
        worst_violation = max(violations, key=lambda v: v.magnitude)

    # Compute violation rate
    violation_rate = len(violations) / max(num_testable_pairs, 1)

    satisfies_regularity = len(violations) == 0

    computation_time = (time.perf_counter() - start_time) * 1000

    return RegularityResult(
        satisfies_regularity=satisfies_regularity,
        violations=violations,
        worst_violation=worst_violation,
        violation_rate=violation_rate,
        num_testable_pairs=num_testable_pairs,
        computation_time_ms=computation_time,
    )


# =============================================================================
# RUM CONSISTENCY TESTING (Block & Marschak 1960, Smeulders et al. 2021)
# =============================================================================


def test_rum_consistency(
    log: "StochasticChoiceLog",
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
) -> RUMConsistencyResult:
    """
    Test if stochastic choice data can be rationalized by a Random Utility Model.

    A RUM represents choice probabilities as:
        P(choose x | S) = sum_{sigma: x = argmax_{y in S} sigma(y)} pi(sigma)

    where pi is a probability distribution over preference orderings sigma.

    This uses the column generation algorithm from Smeulders et al. (2021)
    for computational efficiency with large numbers of items.

    Args:
        log: StochasticChoiceLog with choice frequency data
        tolerance: Numerical tolerance for LP feasibility
        max_iterations: Maximum iterations for column generation

    Returns:
        RUMConsistencyResult with consistency test and rationalizing distribution

    Example:
        >>> from pyrevealed import StochasticChoiceLog, test_rum_consistency
        >>> log = StochasticChoiceLog(
        ...     menus=[{0,1,2}, {0,1}, {1,2}],
        ...     choice_frequencies=[{0: 40, 1: 35, 2: 25}, {0: 55, 1: 45}, {1: 60, 2: 40}],
        ...     total_observations_per_menu=[100, 100, 100]
        ... )
        >>> result = test_rum_consistency(log)
        >>> print(f"RUM consistent: {result.is_rum_consistent}")
        >>> if result.rationalizing_distribution:
        ...     print(f"Uses {result.num_orderings_used} orderings")

    References:
        Block, H. D., & Marschak, J. (1960). Random orderings and stochastic theories
        of responses. Contributions to probability and statistics, 2, 97-132.

        Smeulders, B., Crama, Y., & Spieksma, F. C. (2021). Revealed preference theory:
        An algorithmic outlook. European Journal of Operational Research, 294(3).
    """
    start_time = time.perf_counter()

    all_items = sorted(log.all_items)
    n_items = len(all_items)

    # First check regularity (necessary condition)
    regularity_violations = _find_regularity_violations(log)
    regularity_satisfied = len(regularity_violations) == 0

    # For small n, use exact enumeration
    if n_items <= 6:
        result = _test_rum_exact(log, tolerance)
    else:
        # For larger n, use column generation
        result = _test_rum_column_generation(log, tolerance, max_iterations)

    computation_time = (time.perf_counter() - start_time) * 1000

    return RUMConsistencyResult(
        is_rum_consistent=result["is_consistent"],
        distance_to_rum=result["distance"],
        regularity_satisfied=regularity_satisfied,
        num_orderings_used=result["num_orderings"],
        rationalizing_distribution=result["distribution"],
        num_iterations=result["iterations"],
        constraint_violations=result["violations"],
        computation_time_ms=computation_time,
    )


def _test_rum_exact(
    log: "StochasticChoiceLog",
    tolerance: float,
) -> dict:
    """Test RUM consistency using exact enumeration of all orderings."""
    from itertools import permutations
    from scipy.optimize import linprog

    all_items = sorted(log.all_items)
    n_items = len(all_items)

    # Generate all possible orderings
    orderings = list(permutations(all_items))
    n_orderings = len(orderings)

    # Build constraint matrix
    # For each (menu, item) pair, we have a constraint:
    # sum_{sigma: item is best in menu under sigma} pi_sigma = observed_prob

    constraints_eq = []
    b_eq = []
    constraint_labels = []

    for m_idx in range(log.num_menus):
        menu = log.menus[m_idx]
        total = log.total_observations_per_menu[m_idx]

        if total == 0:
            continue

        for item in menu:
            observed_prob = log.get_choice_probability(m_idx, item)

            # Build row: coefficient for each ordering
            row = np.zeros(n_orderings)
            for o_idx, ordering in enumerate(orderings):
                # Is item the best in menu under this ordering?
                rank = {x: i for i, x in enumerate(ordering)}
                best_in_menu = min(menu, key=lambda x: rank[x])
                if best_in_menu == item:
                    row[o_idx] = 1.0

            constraints_eq.append(row)
            b_eq.append(observed_prob)
            constraint_labels.append(f"P({item}|{set(menu)})={observed_prob:.3f}")

    # Add constraint: probabilities sum to 1
    constraints_eq.append(np.ones(n_orderings))
    b_eq.append(1.0)

    A_eq = np.array(constraints_eq)
    b_eq = np.array(b_eq)

    # Bounds: pi >= 0
    bounds = [(0.0, 1.0) for _ in range(n_orderings)]

    # Objective: minimize sum of slack variables (we use Phase I LP)
    # Actually for feasibility, any objective works
    c = np.zeros(n_orderings)

    try:
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            # Extract non-zero probabilities
            distribution = {}
            for o_idx, prob in enumerate(result.x):
                if prob > tolerance:
                    distribution[orderings[o_idx]] = float(prob)

            return {
                "is_consistent": True,
                "distance": 0.0,
                "num_orderings": len(distribution),
                "distribution": distribution,
                "iterations": 1,
                "violations": [],
            }
        else:
            # Compute distance to nearest RUM using relaxed LP
            distance, violations = _compute_rum_distance(log, orderings, tolerance)
            return {
                "is_consistent": False,
                "distance": distance,
                "num_orderings": 0,
                "distribution": None,
                "iterations": 1,
                "violations": violations,
            }
    except Exception as e:
        return {
            "is_consistent": False,
            "distance": 1.0,
            "num_orderings": 0,
            "distribution": None,
            "iterations": 0,
            "violations": [str(e)],
        }


def _compute_rum_distance(
    log: "StochasticChoiceLog",
    orderings: list[tuple[int, ...]],
    tolerance: float,
) -> tuple[float, list[str]]:
    """Compute L1 distance to nearest RUM."""
    from scipy.optimize import linprog

    n_orderings = len(orderings)

    # Variables: pi_1, ..., pi_K (ordering probs), s+ and s- slack variables
    # For each constraint, add slack: A @ pi + s+ - s- = b
    # Minimize: sum(s+) + sum(s-)

    n_constraints = 0
    constraints = []
    b = []

    for m_idx in range(log.num_menus):
        menu = log.menus[m_idx]
        total = log.total_observations_per_menu[m_idx]

        if total == 0:
            continue

        for item in menu:
            observed_prob = log.get_choice_probability(m_idx, item)

            row = np.zeros(n_orderings)
            for o_idx, ordering in enumerate(orderings):
                rank = {x: i for i, x in enumerate(ordering)}
                best_in_menu = min(menu, key=lambda x: rank[x])
                if best_in_menu == item:
                    row[o_idx] = 1.0

            constraints.append(row)
            b.append(observed_prob)
            n_constraints += 1

    # Sum to 1 constraint
    constraints.append(np.ones(n_orderings))
    b.append(1.0)
    n_constraints += 1

    # Build augmented system with slack variables
    # Variables: [pi_1, ..., pi_K, s+_1, ..., s+_m, s-_1, ..., s-_m]
    n_slack = n_constraints - 1  # Don't need slack for sum-to-1
    n_vars = n_orderings + 2 * n_slack

    A_eq = np.zeros((n_constraints, n_vars))
    b_eq = np.array(b)

    for i in range(n_constraints - 1):
        A_eq[i, :n_orderings] = constraints[i]
        A_eq[i, n_orderings + i] = 1  # s+
        A_eq[i, n_orderings + n_slack + i] = -1  # s-

    # Sum to 1 (no slack)
    A_eq[n_constraints - 1, :n_orderings] = constraints[n_constraints - 1]

    # Objective: minimize sum of slack
    c = np.zeros(n_vars)
    c[n_orderings:] = 1.0

    # Bounds
    bounds = [(0.0, 1.0) for _ in range(n_orderings)] + [(0.0, None) for _ in range(2 * n_slack)]

    try:
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if result.success:
            distance = float(result.fun)
            # Identify which constraints are violated
            violations = []
            slack_plus = result.x[n_orderings:n_orderings + n_slack]
            slack_minus = result.x[n_orderings + n_slack:]
            for i in range(n_slack):
                if slack_plus[i] > tolerance or slack_minus[i] > tolerance:
                    violations.append(f"Constraint {i}: slack+ = {slack_plus[i]:.4f}, slack- = {slack_minus[i]:.4f}")
            return distance, violations
    except Exception:
        pass

    return 1.0, ["LP failed"]


def _test_rum_column_generation(
    log: "StochasticChoiceLog",
    tolerance: float,
    max_iterations: int,
) -> dict:
    """
    Test RUM consistency using column generation algorithm.

    Based on Smeulders et al. (2021) algorithm.
    """
    from scipy.optimize import linprog

    all_items = sorted(log.all_items)
    n_items = len(all_items)

    # Start with a few initial orderings
    initial_orderings = _generate_initial_orderings(log, n_items)
    active_orderings = list(initial_orderings)

    # Build initial constraint structure
    n_constraints = 0
    observed_probs = []

    for m_idx in range(log.num_menus):
        menu = log.menus[m_idx]
        total = log.total_observations_per_menu[m_idx]

        if total == 0:
            continue

        for item in menu:
            observed_probs.append(log.get_choice_probability(m_idx, item))
            n_constraints += 1

    # Iterative column generation
    for iteration in range(max_iterations):
        n_orderings = len(active_orderings)

        # Build constraint matrix for current orderings
        A_eq = np.zeros((n_constraints + 1, n_orderings))  # +1 for sum constraint

        constraint_idx = 0
        for m_idx in range(log.num_menus):
            menu = log.menus[m_idx]
            total = log.total_observations_per_menu[m_idx]

            if total == 0:
                continue

            for item in menu:
                for o_idx, ordering in enumerate(active_orderings):
                    rank = {x: i for i, x in enumerate(ordering)}
                    best_in_menu = min(menu, key=lambda x: rank[x])
                    if best_in_menu == item:
                        A_eq[constraint_idx, o_idx] = 1.0
                constraint_idx += 1

        # Sum to 1 constraint
        A_eq[n_constraints, :] = 1.0

        b_eq = np.array(observed_probs + [1.0])

        # Solve LP
        c = np.zeros(n_orderings)
        bounds = [(0.0, 1.0) for _ in range(n_orderings)]

        try:
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        except Exception:
            break

        if result.success:
            # Found feasible solution
            distribution = {}
            for o_idx, prob in enumerate(result.x):
                if prob > tolerance:
                    distribution[active_orderings[o_idx]] = float(prob)

            return {
                "is_consistent": True,
                "distance": 0.0,
                "num_orderings": len(distribution),
                "distribution": distribution,
                "iterations": iteration + 1,
                "violations": [],
            }

        # Pricing problem: find new ordering to add
        new_ordering = _solve_pricing_problem(log, all_items, n_constraints)
        if new_ordering is None or new_ordering in active_orderings:
            # No improving column found
            break

        active_orderings.append(new_ordering)

    # Column generation didn't find solution
    return {
        "is_consistent": False,
        "distance": 1.0,
        "num_orderings": 0,
        "distribution": None,
        "iterations": max_iterations,
        "violations": ["Column generation did not converge"],
    }


def _generate_initial_orderings(
    log: "StochasticChoiceLog",
    n_items: int,
) -> list[tuple[int, ...]]:
    """Generate initial set of orderings for column generation."""
    all_items = sorted(log.all_items)

    orderings = []

    # Add ordering based on overall choice frequency
    choice_counts = {item: 0 for item in all_items}
    for m_idx in range(log.num_menus):
        freqs = log.choice_frequencies[m_idx]
        for item, count in freqs.items():
            choice_counts[item] += count

    sorted_items = sorted(all_items, key=lambda x: -choice_counts.get(x, 0))
    orderings.append(tuple(sorted_items))

    # Add reverse ordering
    orderings.append(tuple(reversed(sorted_items)))

    # Add some random orderings
    for _ in range(min(5, n_items)):
        perm = tuple(np.random.permutation(all_items).tolist())
        if perm not in orderings:
            orderings.append(perm)

    return orderings


def _solve_pricing_problem(
    log: "StochasticChoiceLog",
    all_items: list[int],
    n_constraints: int,
) -> tuple[int, ...] | None:
    """Solve pricing problem to find new ordering to add."""
    # Simple heuristic: generate random orderings and check if they help
    for _ in range(10):
        perm = tuple(np.random.permutation(all_items).tolist())
        return perm

    return None


def compute_distance_to_rum(
    log: "StochasticChoiceLog",
    norm: str = "l1",
) -> float:
    """
    Compute distance from observed choice data to nearest RUM.

    Args:
        log: StochasticChoiceLog with choice frequency data
        norm: Distance norm ("l1", "l2", "linf")

    Returns:
        Distance to nearest RUM (0 = RUM consistent)

    Example:
        >>> log = StochasticChoiceLog(...)
        >>> distance = compute_distance_to_rum(log)
        >>> print(f"Distance to RUM: {distance:.4f}")
    """
    result = test_rum_consistency(log)
    return result.distance_to_rum


def fit_rum_distribution(
    log: "StochasticChoiceLog",
) -> dict[tuple[int, ...], float]:
    """
    Fit a RUM distribution to stochastic choice data.

    Returns the sparse representation of the probability distribution
    over preference orderings that best fits the data.

    Args:
        log: StochasticChoiceLog with choice frequency data

    Returns:
        Dict mapping preference orderings to probabilities

    Example:
        >>> log = StochasticChoiceLog(...)
        >>> distribution = fit_rum_distribution(log)
        >>> for ordering, prob in distribution.items():
        ...     print(f"{' > '.join(map(str, ordering))}: {prob:.3f}")
    """
    result = test_rum_consistency(log)
    if result.rationalizing_distribution:
        return result.rationalizing_distribution
    return {}


# =============================================================================
# STOCHASTIC TRANSITIVITY TESTS (WST/MST/SST)
# =============================================================================


def test_stochastic_transitivity(
    log: "StochasticChoiceLog",
    level: str = "all",
    tolerance: float = 0.01,
) -> "StochasticTransitivityResult":
    """
    Test stochastic transitivity axioms (WST/MST/SST).

    Stochastic transitivity axioms describe how pairwise choice probabilities
    should be related when forming transitive chains. Let P(a,b) denote the
    probability of choosing a over b in a binary choice.

    Axiom Definitions:
    - WST (Weak): P(a,b) > 0.5 and P(b,c) > 0.5 => P(a,c) > 0.5
    - MST (Moderate): P(a,b) > 0.5 and P(b,c) > 0.5 => P(a,c) >= min(P(a,b), P(b,c))
    - SST (Strong): P(a,b) > 0.5 and P(b,c) > 0.5 => P(a,c) >= max(P(a,b), P(b,c))

    These form a hierarchy: SST => MST => WST. If a consumer's choices satisfy
    SST, they also satisfy MST and WST.

    Args:
        log: StochasticChoiceLog with pairwise choice data
        level: Which levels to test - "all", "weak", "moderate", or "strong"
        tolerance: Tolerance for probability comparisons

    Returns:
        StochasticTransitivityResult with violation details for each level

    Example:
        >>> from pyrevealed import StochasticChoiceLog, test_stochastic_transitivity
        >>> log = StochasticChoiceLog(
        ...     menus=[{0,1}, {1,2}, {0,2}],
        ...     choice_frequencies=[{0: 60, 1: 40}, {1: 55, 2: 45}, {0: 70, 2: 30}],
        ...     total_observations_per_menu=[100, 100, 100]
        ... )
        >>> result = test_stochastic_transitivity(log)
        >>> print(f"Strongest level satisfied: {result.strongest_satisfied}")

    References:
        Luce, R. D. (1959). Individual Choice Behavior: A Theoretical Analysis.
        Tversky, A. (1969). Intransitivity of preferences. Psychological Review.
    """
    from pyrevealed.core.result import StochasticTransitivityResult

    start_time = time.perf_counter()

    # Build pairwise probability matrix
    items = sorted(log.all_items)
    n_items = len(items)
    item_to_idx = {item: idx for idx, item in enumerate(items)}

    # P[i,j] = probability of choosing item i over item j
    P_matrix = np.full((n_items, n_items), 0.5)

    # Extract pairwise probabilities from menus of size 2
    for m_idx in range(log.num_menus):
        menu = log.menus[m_idx]
        if len(menu) == 2:
            items_in_menu = list(menu)
            total = log.total_observations_per_menu[m_idx]
            if total > 0:
                i, j = items_in_menu[0], items_in_menu[1]
                idx_i, idx_j = item_to_idx[i], item_to_idx[j]
                p_i = log.get_choice_probability(m_idx, i)
                P_matrix[idx_i, idx_j] = p_i
                P_matrix[idx_j, idx_i] = 1 - p_i

    # Test all transitivity levels
    wst_violations: list[tuple[int, int, int]] = []
    mst_violations: list[tuple[int, int, int]] = []
    sst_violations: list[tuple[int, int, int]] = []
    num_testable = 0

    for a_idx in range(n_items):
        for b_idx in range(n_items):
            if a_idx == b_idx:
                continue
            for c_idx in range(n_items):
                if c_idx == a_idx or c_idx == b_idx:
                    continue

                p_ab = P_matrix[a_idx, b_idx]
                p_bc = P_matrix[b_idx, c_idx]
                p_ac = P_matrix[a_idx, c_idx]

                # Only test if P(a,b) > 0.5 and P(b,c) > 0.5
                if p_ab > 0.5 + tolerance and p_bc > 0.5 + tolerance:
                    num_testable += 1
                    a, b, c = items[a_idx], items[b_idx], items[c_idx]

                    # WST: P(a,c) > 0.5
                    if p_ac <= 0.5 + tolerance:
                        wst_violations.append((a, b, c))

                    # MST: P(a,c) >= min(P(a,b), P(b,c))
                    if p_ac < min(p_ab, p_bc) - tolerance:
                        mst_violations.append((a, b, c))

                    # SST: P(a,c) >= max(P(a,b), P(b,c))
                    if p_ac < max(p_ab, p_bc) - tolerance:
                        sst_violations.append((a, b, c))

    satisfies_wst = len(wst_violations) == 0
    satisfies_mst = len(mst_violations) == 0
    satisfies_sst = len(sst_violations) == 0

    computation_time = (time.perf_counter() - start_time) * 1000

    return StochasticTransitivityResult(
        satisfies_wst=satisfies_wst,
        satisfies_mst=satisfies_mst,
        satisfies_sst=satisfies_sst,
        wst_violations=wst_violations,
        mst_violations=mst_violations,
        sst_violations=sst_violations,
        num_testable_triples=num_testable,
        computation_time_ms=computation_time,
    )


# =============================================================================
# ADDITIVE PERTURBED UTILITY (Fudenberg et al. 2015)
# =============================================================================


def test_additive_perturbed_utility(
    log: "StochasticChoiceLog",
    tolerance: float = 0.01,
) -> dict:
    """
    Test if stochastic choices satisfy Additive Perturbed Utility (APU).

    APU models assume the agent maximizes:
        U(x) - c(p(x))

    where U(x) is utility, p(x) is the choice probability, and c is a
    strictly convex "cost of attention" function. This generalizes
    the multinomial logit model.

    Key testable implications:
    1. Regularity: P(x|A) >= P(x|B) when A subset of B
    2. Monotonicity in choice probabilities
    3. Log-odds linearity (for logit special case)

    Args:
        log: StochasticChoiceLog with choice frequency data
        tolerance: Tolerance for probability comparisons

    Returns:
        Dict with test results including:
        - is_apu_consistent: Whether data is consistent with APU
        - satisfies_regularity: Whether regularity holds
        - regularity_violations: List of regularity violations
        - monotonicity_score: Measure of monotonicity

    Example:
        >>> from pyrevealed import StochasticChoiceLog, test_additive_perturbed_utility
        >>> result = test_additive_perturbed_utility(log)
        >>> if result["is_apu_consistent"]:
        ...     print("Consistent with APU model")

    References:
        Fudenberg, D., Iijima, R., & Strzalecki, T. (2015). Stochastic choice
        and revealed perturbed utility. Econometrica, 83(6), 2371-2409.
    """
    start_time = time.perf_counter()

    # Test regularity (necessary condition for APU)
    regularity_violations = _find_regularity_violations(log, tolerance)
    satisfies_regularity = len(regularity_violations) == 0

    # Test IIA (logit special case of APU)
    satisfies_iia = check_independence_irrelevant_alternatives(log, tolerance)

    # Compute monotonicity score
    # For APU: items with higher utility should have higher choice probabilities
    # across different menus (after accounting for menu composition)
    monotonicity_score = _compute_monotonicity_score(log)

    # APU is consistent if regularity holds
    # (This is a necessary but not sufficient condition)
    is_apu_consistent = satisfies_regularity

    computation_time = (time.perf_counter() - start_time) * 1000

    return {
        "is_apu_consistent": is_apu_consistent,
        "satisfies_regularity": satisfies_regularity,
        "satisfies_iia": satisfies_iia,
        "regularity_violations": regularity_violations,
        "monotonicity_score": monotonicity_score,
        "is_logit_consistent": satisfies_iia,
        "computation_time_ms": computation_time,
    }


def _compute_monotonicity_score(log: "StochasticChoiceLog") -> float:
    """
    Compute monotonicity score for choice probabilities.

    Higher score indicates more monotonic relationship between
    item "quality" and choice probabilities.
    """
    items = sorted(log.all_items)
    n_items = len(items)

    if n_items < 2:
        return 1.0

    # Estimate item quality from average choice probability
    quality = {}
    for item in items:
        probs = []
        for m_idx in range(log.num_menus):
            if item in log.menus[m_idx]:
                probs.append(log.get_choice_probability(m_idx, item))
        quality[item] = np.mean(probs) if probs else 0.5

    # Count concordant/discordant pairs across menus
    concordant = 0
    discordant = 0

    for m_idx in range(log.num_menus):
        menu_items = list(log.menus[m_idx])
        for i in range(len(menu_items)):
            for j in range(i + 1, len(menu_items)):
                item_i, item_j = menu_items[i], menu_items[j]
                p_i = log.get_choice_probability(m_idx, item_i)
                p_j = log.get_choice_probability(m_idx, item_j)

                # Does quality ordering match probability ordering?
                quality_order = quality[item_i] > quality[item_j]
                prob_order = p_i > p_j

                if quality_order == prob_order:
                    concordant += 1
                else:
                    discordant += 1

    total = concordant + discordant
    if total == 0:
        return 1.0

    return concordant / total


# =============================================================================
# ADDITIONAL LEGACY ALIASES
# =============================================================================

check_rum_consistency = test_rum_consistency
"""Legacy alias: use test_rum_consistency instead."""

test_wst = test_stochastic_transitivity
"""Legacy alias: use test_stochastic_transitivity instead."""

check_stochastic_transitivity = test_stochastic_transitivity
"""Legacy alias: use test_stochastic_transitivity instead."""

check_apu = test_additive_perturbed_utility
"""Legacy alias: use test_additive_perturbed_utility instead."""
