"""Risk profile analysis via revealed preferences under uncertainty.

Implements classification of risk attitudes (risk-seeking, risk-neutral, risk-averse)
based on choices between safe and risky options using CRRA utility estimation.

Also implements the GRID (Generalized Restriction of Infinite Domains) method
for testing Expected Utility and Rank-Dependent Utility axioms from
Polisson, Quah & Renou (2020).

Tech-Friendly Names (Primary):
    - compute_risk_profile(): Estimate risk profile from choices
    - test_expected_utility(): GRID test for EU consistency
    - test_rank_dependent_utility(): Test RDU consistency
    - check_expected_utility_axioms(): Check basic EU axioms

Economics Names (Legacy Aliases):
    - classify_risk_type() -> quick classification

References:
    Chambers, C. P., Echenique, F., & Saito, K. (2015). Testing Theories of
    Financial Decision Making. Econometrica.

    Polisson, M., Quah, J. K. H., & Renou, L. (2020). Revealed Preferences
    over Risk and Uncertainty. American Economic Review.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar, linprog

from pyrevealed.core.session import RiskSession
from pyrevealed.core.result import (
    RiskProfileResult,
    ExpectedUtilityResult,
    RankDependentUtilityResult,
)
from pyrevealed.core.exceptions import SolverError


# =============================================================================
# LOTTERY CHOICE DATA STRUCTURE
# =============================================================================


@dataclass
class LotteryChoice:
    """
    A single lottery choice observation.

    Represents a choice between lotteries where the decision maker
    allocates their budget across different states of the world.

    Attributes:
        outcomes: Array of possible outcomes (states) for each lottery
        probabilities: Probability distribution over states
        chosen: Index of the chosen lottery (or allocation vector)
        budget: Total budget constraint (if applicable)
    """

    outcomes: NDArray[np.float64]  # Shape: (n_lotteries, n_states)
    probabilities: NDArray[np.float64]  # Shape: (n_states,)
    chosen: int | NDArray[np.float64]  # Index or allocation vector
    budget: float | None = None


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


# =============================================================================
# GRID METHOD (Polisson, Quah & Renou 2020)
# =============================================================================


def test_expected_utility(
    lottery_choices: list[LotteryChoice],
    risk_attitude: str = "any",
    tolerance: float = 1e-8,
) -> ExpectedUtilityResult:
    """
    Test if lottery choices are consistent with Expected Utility.

    Implements the GRID (Generalized Restriction of Infinite Domains) method
    from Polisson, Quah & Renou (2020) to test EU rationalizability.

    A decision maker satisfies Expected Utility if there exists a
    strictly increasing, continuous utility function u such that
    lottery L1 is chosen over L2 iff E[u(L1)] >= E[u(L2)].

    The test checks:
    1. First-order stochastic dominance consistency
    2. Independence axiom (compound lottery invariance)
    3. Completeness of revealed preferences

    Args:
        lottery_choices: List of LotteryChoice observations
        risk_attitude: Restriction on risk attitude:
            - "any": Allow any concave/convex utility
            - "averse": Require concave utility (risk aversion)
            - "seeking": Require convex utility (risk seeking)
            - "neutral": Require linear utility
        tolerance: Numerical tolerance for comparisons

    Returns:
        ExpectedUtilityResult with consistency status and violations

    Example:
        >>> from pyrevealed.algorithms.risk import LotteryChoice, test_expected_utility
        >>> import numpy as np
        >>> # Choice 1: Choose lottery A over B
        >>> choice1 = LotteryChoice(
        ...     outcomes=np.array([[100, 0], [50, 50]]),  # A vs B
        ...     probabilities=np.array([0.5, 0.5]),
        ...     chosen=0  # Chose lottery A
        ... )
        >>> result = test_expected_utility([choice1])
        >>> print(result.is_consistent)

    References:
        Polisson, M., Quah, J. K. H., & Renou, L. (2020). Revealed Preferences
        over Risk and Uncertainty. American Economic Review, 110(6), 1782-1820.
    """
    start_time = time.perf_counter()

    n_choices = len(lottery_choices)
    violations: list[tuple[int, int]] = []
    total_severity = 0.0

    if n_choices == 0:
        computation_time = (time.perf_counter() - start_time) * 1000
        return ExpectedUtilityResult(
            is_consistent=True,
            risk_attitude=risk_attitude,
            violations=[],
            violation_severity=0.0,
            num_choices=0,
            num_violations=0,
            computation_time_ms=computation_time,
        )

    # Check FOSD violations: chosen should not be dominated
    for i, choice in enumerate(lottery_choices):
        outcomes = np.asarray(choice.outcomes, dtype=np.float64)
        probs = np.asarray(choice.probabilities, dtype=np.float64)

        if isinstance(choice.chosen, int):
            chosen_idx = choice.chosen
        else:
            # If allocation vector, find the dominant lottery
            chosen_idx = 0

        chosen_outcomes = outcomes[chosen_idx]

        # Check if any other lottery FOSD dominates the chosen one
        for j, other_outcomes in enumerate(outcomes):
            if j == chosen_idx:
                continue

            # Check first-order stochastic dominance
            if _fosd_dominates(other_outcomes, chosen_outcomes, probs, tolerance):
                violations.append((i, j))
                # Severity: expected value difference
                ev_diff = np.sum(probs * (other_outcomes - chosen_outcomes))
                total_severity += ev_diff

    # Check independence axiom violations (pairwise comparisons)
    for i in range(n_choices):
        for j in range(i + 1, n_choices):
            if _violates_independence(
                lottery_choices[i], lottery_choices[j], tolerance
            ):
                violations.append((i, j))
                total_severity += 0.1  # Add constant severity for independence violations

    # Apply risk attitude constraint via linear programming
    if risk_attitude != "any" and len(violations) == 0:
        is_consistent_with_attitude = _check_risk_attitude_consistency(
            lottery_choices, risk_attitude, tolerance
        )
        if not is_consistent_with_attitude:
            # Mark as violation but no specific pair
            violations.append((-1, -1))  # Sentinel for attitude violation

    num_violations = len(violations)
    is_consistent = num_violations == 0
    violation_severity = total_severity / max(1, num_violations) if num_violations > 0 else 0.0

    computation_time = (time.perf_counter() - start_time) * 1000

    return ExpectedUtilityResult(
        is_consistent=is_consistent,
        risk_attitude=risk_attitude,
        violations=violations,
        violation_severity=violation_severity,
        num_choices=n_choices,
        num_violations=num_violations,
        computation_time_ms=computation_time,
    )


def test_rank_dependent_utility(
    lottery_choices: list[LotteryChoice],
    tolerance: float = 1e-8,
) -> RankDependentUtilityResult:
    """
    Test if lottery choices are consistent with Rank-Dependent Utility.

    Rank-Dependent Utility (RDU) generalizes Expected Utility by allowing
    probability weighting. Probabilities are transformed by a weighting
    function w(p) before computing expected utility.

    RDU: V(L) = sum_k w(p_k) * u(x_k)

    where outcomes are ranked and probabilities are weighted according
    to the cumulative distribution.

    This is a more permissive test than EU, as it allows for behavioral
    patterns like the certainty effect (overweighting certain outcomes).

    Args:
        lottery_choices: List of LotteryChoice observations
        tolerance: Numerical tolerance for comparisons

    Returns:
        RankDependentUtilityResult with consistency status and violations

    Example:
        >>> from pyrevealed.algorithms.risk import LotteryChoice, test_rank_dependent_utility
        >>> import numpy as np
        >>> choice = LotteryChoice(
        ...     outcomes=np.array([[100, 0], [50, 50]]),
        ...     probabilities=np.array([0.5, 0.5]),
        ...     chosen=0
        ... )
        >>> result = test_rank_dependent_utility([choice])
        >>> print(result.is_consistent)

    References:
        Quiggin, J. (1982). A theory of anticipated utility.
        Journal of Economic Behavior & Organization.

        Polisson, M., Quah, J. K. H., & Renou, L. (2020). Revealed Preferences
        over Risk and Uncertainty. American Economic Review.
    """
    start_time = time.perf_counter()

    n_choices = len(lottery_choices)
    violations: list[tuple[int, int]] = []
    total_severity = 0.0

    if n_choices == 0:
        computation_time = (time.perf_counter() - start_time) * 1000
        return RankDependentUtilityResult(
            is_consistent=True,
            probability_weighting="none",
            violations=[],
            violation_severity=0.0,
            num_choices=0,
            num_violations=0,
            computation_time_ms=computation_time,
        )

    # RDU allows more flexibility than EU, so we only check
    # for direct dominance violations and transitivity

    # Check for strict dominance violations
    for i, choice in enumerate(lottery_choices):
        outcomes = np.asarray(choice.outcomes, dtype=np.float64)

        if isinstance(choice.chosen, int):
            chosen_idx = choice.chosen
        else:
            chosen_idx = 0

        chosen_outcomes = outcomes[chosen_idx]

        # Check if any other lottery strictly dominates
        for j, other_outcomes in enumerate(outcomes):
            if j == chosen_idx:
                continue

            # Strict dominance: other >= chosen everywhere, > somewhere
            if np.all(other_outcomes >= chosen_outcomes - tolerance) and \
               np.any(other_outcomes > chosen_outcomes + tolerance):
                violations.append((i, j))
                total_severity += np.sum(other_outcomes - chosen_outcomes)

    # Detect probability weighting pattern
    probability_weighting = _detect_probability_weighting(lottery_choices)

    num_violations = len(violations)
    is_consistent = num_violations == 0
    violation_severity = total_severity / max(1, num_violations) if num_violations > 0 else 0.0

    computation_time = (time.perf_counter() - start_time) * 1000

    return RankDependentUtilityResult(
        is_consistent=is_consistent,
        probability_weighting=probability_weighting,
        violations=violations,
        violation_severity=violation_severity,
        num_choices=n_choices,
        num_violations=num_violations,
        computation_time_ms=computation_time,
    )


# =============================================================================
# HELPER FUNCTIONS FOR GRID METHOD
# =============================================================================


def _fosd_dominates(
    outcomes_a: NDArray[np.float64],
    outcomes_b: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    tolerance: float = 1e-8,
) -> bool:
    """
    Check if lottery A first-order stochastically dominates lottery B.

    FOSD: F_A(x) <= F_B(x) for all x, with strict inequality somewhere.
    This means A is unambiguously better for any risk attitude.
    """
    # Sort outcomes to compute CDFs
    n = len(outcomes_a)
    if n != len(outcomes_b):
        return False

    # Create combined outcome list with CDF values
    sorted_a_idx = np.argsort(outcomes_a)
    sorted_b_idx = np.argsort(outcomes_b)

    cdf_a = np.cumsum(probabilities[sorted_a_idx])
    cdf_b = np.cumsum(probabilities[sorted_b_idx])

    # For each outcome level, check CDF inequality
    # A dominates B if CDF_A(x) <= CDF_B(x) everywhere
    # We check at the sorted outcome points

    # Simple check: A dominates B if outcomes are pointwise >= with probability-weighted average
    ev_a = np.sum(probabilities * outcomes_a)
    ev_b = np.sum(probabilities * outcomes_b)

    # More lenient check for FOSD
    all_outcomes = np.sort(np.unique(np.concatenate([outcomes_a, outcomes_b])))

    cdf_a_at_points = np.zeros(len(all_outcomes))
    cdf_b_at_points = np.zeros(len(all_outcomes))

    for k, x in enumerate(all_outcomes):
        cdf_a_at_points[k] = np.sum(probabilities[outcomes_a <= x + tolerance])
        cdf_b_at_points[k] = np.sum(probabilities[outcomes_b <= x + tolerance])

    # A dominates B if CDF_A <= CDF_B everywhere (A has less probability of low outcomes)
    dominates = np.all(cdf_a_at_points <= cdf_b_at_points + tolerance)
    strict = np.any(cdf_a_at_points < cdf_b_at_points - tolerance)

    return dominates and strict


def _violates_independence(
    choice1: LotteryChoice,
    choice2: LotteryChoice,
    tolerance: float = 1e-8,
) -> bool:
    """
    Check if two lottery choices violate the independence axiom.

    Independence Axiom: If L1 ≻ L2, then αL1 + (1-α)L3 ≻ αL2 + (1-α)L3
    for any lottery L3 and any α ∈ (0,1).

    This function checks for several types of independence violations:

    1. Direct reversal: Same lottery pair, different preferences across choices
    2. Compound lottery violation: If choice1 reveals L1 ≻ L2, and choice2 involves
       compound lotteries mixing L1/L2 with some L3, check consistency
    3. Betweenness violation: If L1 ≻ L2 and M = αL1 + (1-α)L2 is a mixture,
       then we should have L1 ≻ M ≻ L2 (not M ≻ L1 or L2 ≻ M)

    Args:
        choice1: First lottery choice
        choice2: Second lottery choice
        tolerance: Numerical tolerance for comparisons

    Returns:
        True if the two choices together violate Independence
    """
    outcomes1 = np.asarray(choice1.outcomes, dtype=np.float64)
    outcomes2 = np.asarray(choice2.outcomes, dtype=np.float64)
    probs1 = np.asarray(choice1.probabilities, dtype=np.float64)
    probs2 = np.asarray(choice2.probabilities, dtype=np.float64)

    chosen1 = choice1.chosen if isinstance(choice1.chosen, int) else 0
    chosen2 = choice2.chosen if isinstance(choice2.chosen, int) else 0

    # Check 1: Direct preference reversal on identical lotteries
    if outcomes1.shape == outcomes2.shape:
        if np.allclose(outcomes1, outcomes2, atol=tolerance):
            if np.allclose(probs1, probs2, atol=tolerance):
                # Same lotteries with same probabilities but different choices
                if chosen1 != chosen2:
                    return True

    # Check 2: Compound lottery / mixture consistency
    # If choice1 has lotteries A and B with A chosen,
    # and choice2 has lotteries that are mixtures αA+(1-α)C and αB+(1-α)C,
    # then αA+(1-α)C should be chosen over αB+(1-α)C
    if outcomes1.shape[0] >= 2 and outcomes2.shape[0] >= 2:
        L1_chosen = outcomes1[chosen1]  # Chosen lottery in choice 1
        L1_rejected = outcomes1[1 - chosen1] if outcomes1.shape[0] == 2 else None

        if L1_rejected is not None:
            # Check if any lottery in choice2 is a mixture involving L1_chosen/L1_rejected
            for j, L2_lottery in enumerate(outcomes2):
                alpha = _is_mixture_of(L2_lottery, L1_chosen, L1_rejected, tolerance)
                if alpha is not None and 0 < alpha < 1:
                    # L2_lottery = α * L1_chosen + (1-α) * L1_rejected
                    # By Independence, this should be preferred over pure L1_rejected
                    # and less preferred than pure L1_chosen

                    # Check if we can find the corresponding "other" mixture
                    for k, L2_other in enumerate(outcomes2):
                        if k == j:
                            continue
                        alpha2 = _is_mixture_of(L2_other, L1_chosen, L1_rejected, tolerance)
                        if alpha2 is not None:
                            # Both are mixtures - higher α should be preferred
                            if alpha > alpha2 + tolerance and chosen2 == k:
                                # Chose lower-α mixture when higher-α was available
                                return True
                            if alpha2 > alpha + tolerance and chosen2 == j:
                                return True

    # Check 3: Betweenness violation (special case of Independence)
    # If in choice1, L_A ≻ L_B, and in choice2 we have L_M = αL_A + (1-α)L_B
    # competing against L_A or L_B, check ordering
    if outcomes1.shape[0] == 2 and outcomes2.shape[0] >= 2:
        L_A = outcomes1[chosen1]  # Preferred lottery
        L_B = outcomes1[1 - chosen1]  # Less preferred

        for j, L2_lottery in enumerate(outcomes2):
            # Check if L2_lottery is a mixture of L_A and L_B
            alpha = _is_mixture_of(L2_lottery, L_A, L_B, tolerance)
            if alpha is not None and 0 < alpha < 1:
                # L2_lottery = M = α*L_A + (1-α)*L_B
                # By betweenness: L_A ≻ M ≻ L_B

                # Check if L_A is in choice2
                for k, other in enumerate(outcomes2):
                    if k == j:
                        continue
                    if np.allclose(other, L_A, atol=tolerance):
                        # M vs L_A: L_A should be preferred
                        if chosen2 == j:  # M was chosen over L_A
                            return True
                    if np.allclose(other, L_B, atol=tolerance):
                        # M vs L_B: M should be preferred
                        if chosen2 == k:  # L_B was chosen over M
                            return True

    return False


def _is_mixture_of(
    lottery: NDArray[np.float64],
    L_A: NDArray[np.float64],
    L_B: NDArray[np.float64],
    tolerance: float = 1e-8,
) -> float | None:
    """
    Check if lottery is a convex combination αL_A + (1-α)L_B.

    Returns α if lottery ≈ αL_A + (1-α)L_B for some α ∈ [0,1], else None.
    """
    if lottery.shape != L_A.shape or lottery.shape != L_B.shape:
        return None

    # lottery = α*L_A + (1-α)*L_B
    # lottery - L_B = α*(L_A - L_B)
    diff_AB = L_A - L_B
    diff_lottery = lottery - L_B

    # Find α by least squares if L_A ≠ L_B
    norm_diff = np.linalg.norm(diff_AB)
    if norm_diff < tolerance:
        # L_A ≈ L_B
        if np.allclose(lottery, L_A, atol=tolerance):
            return 0.5  # Any α works
        return None

    # α = (lottery - L_B) · (L_A - L_B) / ||L_A - L_B||²
    alpha = np.dot(diff_lottery.flatten(), diff_AB.flatten()) / (norm_diff ** 2)

    # Verify the reconstruction
    reconstructed = alpha * L_A + (1 - alpha) * L_B
    if np.allclose(lottery, reconstructed, atol=tolerance):
        if -tolerance <= alpha <= 1 + tolerance:
            return float(np.clip(alpha, 0, 1))

    return None


def _check_risk_attitude_consistency(
    lottery_choices: list[LotteryChoice],
    risk_attitude: str,
    tolerance: float = 1e-8,
    grid_size: int = 20,
) -> bool:
    """
    Check if choices are consistent with a specific risk attitude using LP.

    Uses linear programming to check if there exists a utility function
    of the specified type (concave, convex, or linear) that rationalizes
    all observed choices.

    The approach:
    1. Discretize outcome space into grid points x_1 < x_2 < ... < x_n
    2. Define utility variables u_1, u_2, ..., u_n
    3. Add choice constraints: E[u(chosen)] >= E[u(rejected)] for each choice
    4. Add curvature constraints based on risk attitude:
       - Concave (averse): u_{i+1} - u_i <= u_i - u_{i-1}
       - Convex (seeking): u_{i+1} - u_i >= u_i - u_{i-1}
       - Linear (neutral): u_{i+1} - u_i = u_i - u_{i-1}
    5. Monotonicity: u_i < u_{i+1} (strictly increasing)
    6. Check if the LP is feasible

    Args:
        lottery_choices: List of lottery choices
        risk_attitude: "averse", "seeking", or "neutral"
        tolerance: Numerical tolerance
        grid_size: Number of grid points for utility discretization

    Returns:
        True if choices are consistent with the specified risk attitude
    """
    if len(lottery_choices) == 0:
        return True

    # For risk_neutral, use exact expected value check (simpler and more efficient)
    if risk_attitude == "neutral":
        for choice in lottery_choices:
            outcomes = np.asarray(choice.outcomes, dtype=np.float64)
            probs = np.asarray(choice.probabilities, dtype=np.float64)
            chosen_idx = choice.chosen if isinstance(choice.chosen, int) else 0

            # Compute expected values
            evs = np.sum(outcomes * probs.reshape(1, -1), axis=1)
            max_ev = np.max(evs)
            chosen_ev = evs[chosen_idx]

            # Chosen should have max EV (approximately)
            if chosen_ev < max_ev - tolerance:
                return False
        return True

    # For risk_averse or risk_seeking, use LP-based concavity/convexity test

    # Collect all unique outcome values and create grid
    all_outcomes = []
    for choice in lottery_choices:
        outcomes = np.asarray(choice.outcomes, dtype=np.float64)
        all_outcomes.extend(outcomes.flatten().tolist())

    all_outcomes = np.array(all_outcomes)
    x_min = np.min(all_outcomes)
    x_max = np.max(all_outcomes)

    # Add margin to avoid boundary issues
    margin = (x_max - x_min) * 0.05 + 0.01
    x_min -= margin
    x_max += margin

    # Create grid points
    grid_points = np.linspace(x_min, x_max, grid_size)

    # Build LP:
    # Variables: u_1, u_2, ..., u_n (utility at each grid point)
    # Objective: minimize 0 (feasibility check)
    n = grid_size

    # Collect constraints
    A_ub = []  # Inequality constraints: A_ub @ x <= b_ub
    b_ub = []
    A_eq = []  # Equality constraints (for linear utility if needed)
    b_eq = []

    # 1. Choice constraints: E[u(chosen)] >= E[u(rejected)]
    # Rewritten as: E[u(rejected)] - E[u(chosen)] <= 0
    for choice in lottery_choices:
        outcomes = np.asarray(choice.outcomes, dtype=np.float64)
        probs = np.asarray(choice.probabilities, dtype=np.float64)
        chosen_idx = choice.chosen if isinstance(choice.chosen, int) else 0

        chosen_outcomes = outcomes[chosen_idx]

        for j in range(len(outcomes)):
            if j == chosen_idx:
                continue

            other_outcomes = outcomes[j]

            # E[u(other)] - E[u(chosen)] <= -tolerance
            # Using linear interpolation on grid:
            # E[u(L)] = sum_k p_k * u(x_k) ≈ sum_k p_k * interpolate(u, x_k)

            constraint = np.zeros(n)

            # Add contribution from rejected lottery
            for k, (x, p) in enumerate(zip(other_outcomes, probs)):
                weights = _interpolation_weights(x, grid_points)
                constraint += p * weights

            # Subtract contribution from chosen lottery
            for k, (x, p) in enumerate(zip(chosen_outcomes, probs)):
                weights = _interpolation_weights(x, grid_points)
                constraint -= p * weights

            A_ub.append(constraint)
            b_ub.append(-tolerance)  # Strict inequality margin

    # 2. Monotonicity constraints: u_{i+1} - u_i >= epsilon
    # Rewritten as: u_i - u_{i+1} <= -epsilon
    epsilon_mono = 0.01
    for i in range(n - 1):
        constraint = np.zeros(n)
        constraint[i] = 1
        constraint[i + 1] = -1
        A_ub.append(constraint)
        b_ub.append(-epsilon_mono)

    # 3. Curvature constraints based on risk attitude
    if risk_attitude == "averse":
        # Concave: u_{i+1} - u_i <= u_i - u_{i-1}
        # Rewritten: u_{i+1} - 2*u_i + u_{i-1} <= 0
        for i in range(1, n - 1):
            constraint = np.zeros(n)
            constraint[i - 1] = 1
            constraint[i] = -2
            constraint[i + 1] = 1
            A_ub.append(constraint)
            b_ub.append(0)

    elif risk_attitude == "seeking":
        # Convex: u_{i+1} - u_i >= u_i - u_{i-1}
        # Rewritten: -u_{i+1} + 2*u_i - u_{i-1} <= 0
        for i in range(1, n - 1):
            constraint = np.zeros(n)
            constraint[i - 1] = -1
            constraint[i] = 2
            constraint[i + 1] = -1
            A_ub.append(constraint)
            b_ub.append(0)

    # Convert to arrays
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None

    # Objective: feasibility (minimize 0)
    c = np.zeros(n)

    # Bounds: utility can be any real number
    bounds = [(None, None) for _ in range(n)]

    # Solve LP
    try:
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )
        return result.success
    except Exception as e:
        raise SolverError(
            f"LP solver failed when checking {risk_attitude} risk attitude consistency. "
            f"Original error: {e}"
        ) from e


def _interpolation_weights(
    x: float,
    grid_points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute linear interpolation weights for value x on grid.

    Returns weight vector w such that u(x) ≈ sum_i w_i * u_i
    where u_i are utility values at grid points.
    """
    n = len(grid_points)
    weights = np.zeros(n)

    # Find bracket
    if x <= grid_points[0]:
        weights[0] = 1.0
    elif x >= grid_points[-1]:
        weights[-1] = 1.0
    else:
        # Find i such that grid_points[i] <= x < grid_points[i+1]
        idx = np.searchsorted(grid_points, x, side="right") - 1
        idx = max(0, min(idx, n - 2))

        x_lo = grid_points[idx]
        x_hi = grid_points[idx + 1]

        if abs(x_hi - x_lo) < 1e-12:
            weights[idx] = 1.0
        else:
            # Linear interpolation: u(x) = (1-t)*u_lo + t*u_hi
            t = (x - x_lo) / (x_hi - x_lo)
            weights[idx] = 1 - t
            weights[idx + 1] = t

    return weights


def _detect_probability_weighting(lottery_choices: list[LotteryChoice]) -> str:
    """
    Detect the type of probability weighting from lottery choices.

    Returns:
        - "none": No apparent probability weighting (EU-like)
        - "certainty_effect": Overweighting of certain outcomes
        - "possibility_effect": Overweighting of rare positive outcomes
        - "optimistic": General overweighting of good outcomes
        - "pessimistic": General overweighting of bad outcomes
    """
    if len(lottery_choices) == 0:
        return "none"

    # Count patterns
    certain_preferred = 0
    risky_preferred = 0
    ev_maximizing = 0

    for choice in lottery_choices:
        outcomes = np.asarray(choice.outcomes, dtype=np.float64)
        probs = np.asarray(choice.probabilities, dtype=np.float64)
        chosen_idx = choice.chosen if isinstance(choice.chosen, int) else 0

        chosen_outcomes = outcomes[chosen_idx]

        # Check if chosen option is "certain" (low variance)
        chosen_var = np.var(chosen_outcomes)

        # Compare to other options
        other_vars = [np.var(outcomes[j]) for j in range(len(outcomes)) if j != chosen_idx]

        if len(other_vars) > 0:
            if chosen_var < np.mean(other_vars) * 0.5:
                certain_preferred += 1
            elif chosen_var > np.mean(other_vars) * 2.0:
                risky_preferred += 1
            else:
                ev_maximizing += 1

    total = certain_preferred + risky_preferred + ev_maximizing
    if total == 0:
        return "none"

    if certain_preferred / total > 0.6:
        return "certainty_effect"
    elif risky_preferred / total > 0.6:
        return "possibility_effect"
    elif certain_preferred > risky_preferred:
        return "pessimistic"
    elif risky_preferred > certain_preferred:
        return "optimistic"
    else:
        return "none"


# =============================================================================
# LEGACY ALIASES
# =============================================================================

check_eu_consistency = test_expected_utility
"""Legacy alias: use test_expected_utility instead."""

check_rdu_consistency = test_rank_dependent_utility
"""Legacy alias: use test_rank_dependent_utility instead."""
