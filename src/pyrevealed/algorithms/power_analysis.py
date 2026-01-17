"""Power analysis for revealed preference tests (Beatty & Crawford 2011, Bronars 1987).

This module implements statistical power and predictive success measures for
evaluating the meaningfulness of GARP tests.

Key Concepts:
    - Relative Area: Proportion of outcome space that is GARP-consistent
    - Selten's Measure: m = r - a (pass rate minus relative area)
    - Smoothed Hit Rate: Distance-based measure for near-misses
    - Bayesian Credibility: Posterior probability of utility maximization

References:
    Beatty, T. K., & Crawford, I. A. (2011). How demanding is the revealed
    preference approach to demand? American Economic Review, 101(6), 2782-2795.

    Bronars, S. G. (1987). The power of nonparametric tests of preference
    maximization. Econometrica, 55(3), 693-698.

    Selten, R. (1991). Properties of a measure of predictive success.
    Mathematical Social Sciences, 21(2), 153-167.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pyrevealed.core.session import ConsumerSession

from pyrevealed.core.result import (
    SeltenMeasureResult,
    RelativeAreaResult,
    SmoothedHitRateResult,
    BayesianCredibilityResult,
    OptimalEfficiencyResult,
)


# =============================================================================
# SELTEN'S PREDICTIVE SUCCESS MEASURE (Beatty & Crawford 2011)
# =============================================================================


def compute_selten_measure(
    session: "ConsumerSession",
    n_simulations: int = 1000,
    algorithm: int = 1,
    tolerance: float = 1e-10,
    random_seed: int | None = None,
) -> SeltenMeasureResult:
    """
    Compute Selten's predictive success measure m = r - a.

    This measure captures how well the utility maximization hypothesis
    performs beyond chance. It combines:
    - r: Whether the data passes GARP (1 if pass, 0 if fail)
    - a: Relative area (proportion of random behavior that passes GARP)

    Key insight: Simply passing GARP is insufficient. If 90% of random
    behavior would also pass (a = 0.9), passing is uninformative.

    Interpretation:
    - m → 1: Demanding restrictions + data satisfies them (strong evidence)
    - m ≈ 0: Model performs about as well as random (uninformative)
    - m → -1: Easy restrictions + data fails them (bad fit)

    Args:
        session: ConsumerSession with prices and quantities
        n_simulations: Number of Monte Carlo simulations for area estimation
        algorithm: Random bundle generation algorithm (1, 2, or 3)
            - 1: Uniform on budget hyperplane (corners equally likely)
            - 2: N i.i.d. uniform, normalized (corners rare)
            - 3: Algorithm 2 weighted by actual mean budget shares
        tolerance: Numerical tolerance for GARP detection
        random_seed: Optional seed for reproducibility

    Returns:
        SeltenMeasureResult with m, r, a, and interpretation

    Example:
        >>> from pyrevealed import BehaviorLog, compute_selten_measure
        >>> result = compute_selten_measure(log)
        >>> print(f"Selten's m: {result.measure:.3f}")
        >>> print(f"Pass rate r: {result.pass_rate}")
        >>> print(f"Relative area a: {result.relative_area:.3f}")
        >>> if result.measure > 0.1:
        ...     print("Data performs meaningfully better than random")

    References:
        Beatty, T. K., & Crawford, I. A. (2011). How demanding is the
        revealed preference approach to demand? AER, 101(6), 2782-2795.

        Selten, R. (1991). Properties of a measure of predictive success.
    """
    start_time = time.perf_counter()

    from pyrevealed.algorithms.garp import check_garp

    # Compute pass rate r
    garp_result = check_garp(session, tolerance)
    pass_rate = 1.0 if garp_result.is_consistent else 0.0

    # Compute relative area a via Monte Carlo
    area_result = compute_relative_area(
        session,
        n_simulations=n_simulations,
        algorithm=algorithm,
        tolerance=tolerance,
        random_seed=random_seed,
    )
    relative_area = area_result.relative_area

    # Selten's measure: m = r - a
    measure = pass_rate - relative_area

    computation_time = (time.perf_counter() - start_time) * 1000

    return SeltenMeasureResult(
        measure=measure,
        pass_rate=pass_rate,
        relative_area=relative_area,
        n_simulations=n_simulations,
        algorithm=algorithm,
        is_meaningful=measure > 0.05,  # Threshold from B&C empirical work
        computation_time_ms=computation_time,
    )


# =============================================================================
# RELATIVE AREA (Beatty & Crawford 2011)
# =============================================================================


def compute_relative_area(
    session: "ConsumerSession",
    n_simulations: int = 1000,
    algorithm: int = 1,
    tolerance: float = 1e-10,
    random_seed: int | None = None,
) -> RelativeAreaResult:
    """
    Compute relative area of GARP-consistent outcome space.

    The relative area 'a' measures how demanding the GARP test is for
    a given price configuration. It equals the probability that random
    behavior would satisfy GARP.

    Interpretation:
    - a ≈ 1: GARP is very easy to satisfy (low power)
    - a ≈ 0: GARP is very hard to satisfy (high power)
    - a = 1 - Bronars_power (approximately)

    Note: Many real-world datasets have a ≈ 1 because budget sets
    don't intersect much, making GARP an "unmissable target."

    Args:
        session: ConsumerSession with prices and quantities
        n_simulations: Number of Monte Carlo simulations
        algorithm: Random bundle generation algorithm (1, 2, or 3)
        tolerance: Numerical tolerance for GARP detection
        random_seed: Optional seed for reproducibility

    Returns:
        RelativeAreaResult with area estimate and confidence interval

    Example:
        >>> from pyrevealed import BehaviorLog, compute_relative_area
        >>> result = compute_relative_area(log, n_simulations=2000)
        >>> print(f"Relative area: {result.relative_area:.3f}")
        >>> print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
        >>> if result.relative_area > 0.9:
        ...     print("Warning: GARP test has low discriminatory power")

    References:
        Beatty, T. K., & Crawford, I. A. (2011). How demanding is the
        revealed preference approach to demand? AER, 101(6), 2782-2795.
    """
    start_time = time.perf_counter()

    if random_seed is not None:
        np.random.seed(random_seed)

    from pyrevealed.algorithms.garp import check_garp
    from pyrevealed.core.session import ConsumerSession as CS

    prices = session.prices
    expenditures = session.own_expenditures

    # Compute mean budget shares for algorithm 3
    if algorithm == 3:
        quantities = session.quantities
        mean_shares = _compute_mean_budget_shares(prices, quantities, expenditures)
    else:
        mean_shares = None

    n_consistent = 0
    pass_indicators = np.zeros(n_simulations, dtype=bool)

    for sim in range(n_simulations):
        # Generate random quantities using specified algorithm
        random_quantities = _generate_random_bundles_algorithm(
            prices, expenditures, algorithm, mean_shares
        )

        # Create temporary session and check GARP
        random_session = CS(prices=prices, quantities=random_quantities)
        garp_result = check_garp(random_session, tolerance)

        if garp_result.is_consistent:
            n_consistent += 1
            pass_indicators[sim] = True

    # Relative area = proportion of random simulations that pass GARP
    relative_area = n_consistent / n_simulations

    # Standard error and confidence interval (binomial)
    std_error = np.sqrt(relative_area * (1 - relative_area) / n_simulations)
    ci_lower = max(0.0, relative_area - 1.96 * std_error)
    ci_upper = min(1.0, relative_area + 1.96 * std_error)

    computation_time = (time.perf_counter() - start_time) * 1000

    return RelativeAreaResult(
        relative_area=relative_area,
        std_error=std_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_simulations=n_simulations,
        n_consistent=n_consistent,
        algorithm=algorithm,
        computation_time_ms=computation_time,
    )


def _compute_mean_budget_shares(
    prices: NDArray[np.float64],
    quantities: NDArray[np.float64],
    expenditures: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute mean budget shares from observed data."""
    T, N = prices.shape
    budget_shares = np.zeros((T, N))

    for i in range(T):
        if expenditures[i] > 0:
            budget_shares[i] = (prices[i] * quantities[i]) / expenditures[i]

    return np.mean(budget_shares, axis=0)


def _generate_random_bundles_algorithm(
    prices: NDArray[np.float64],
    expenditures: NDArray[np.float64],
    algorithm: int,
    mean_shares: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    Generate random bundles using specified algorithm.

    Algorithm 1: Uniform on budget hyperplane
        - Budget shares drawn from symmetric Dirichlet(1,...,1)
        - Corner solutions equally likely

    Algorithm 2: N i.i.d. uniform, normalized
        - Draw N uniform [0,1] values, normalize to sum to 1
        - Corners rare, expected share = 1/N

    Algorithm 3: Algorithm 2 weighted by actual mean budget shares
        - Mimics actual data distribution
        - Most realistic for power analysis
    """
    T, N = prices.shape
    random_quantities = np.zeros((T, N))

    for i in range(T):
        if algorithm == 1:
            # Symmetric Dirichlet(1,...,1) = uniform on simplex
            shares = np.random.dirichlet(np.ones(N))

        elif algorithm == 2:
            # N i.i.d. uniform, normalized
            draws = np.random.uniform(0, 1, N)
            shares = draws / np.sum(draws)

        elif algorithm == 3:
            # Weighted by mean budget shares
            if mean_shares is not None and np.sum(mean_shares) > 0:
                # Use Dirichlet with concentration = mean_shares * N
                # This centers the distribution around mean_shares
                alpha = mean_shares * N + 0.1  # Add small constant for stability
                shares = np.random.dirichlet(alpha)
            else:
                # Fall back to algorithm 2
                draws = np.random.uniform(0, 1, N)
                shares = draws / np.sum(draws)
        else:
            # Default to algorithm 1
            shares = np.random.dirichlet(np.ones(N))

        # Convert shares to quantities: q_ij = share_j * e_i / p_ij
        random_quantities[i] = shares * expenditures[i] / prices[i]

    return random_quantities


# =============================================================================
# SMOOTHED HIT RATE (Beatty & Crawford 2011)
# =============================================================================


def compute_smoothed_hit_rate(
    session: "ConsumerSession",
    n_simulations: int = 1000,
    algorithm: int = 1,
    tolerance: float = 1e-10,
    random_seed: int | None = None,
) -> SmoothedHitRateResult:
    """
    Compute smoothed hit rate for GARP violators.

    For data that violates GARP, the standard pass rate r = 0. The smoothed
    hit rate rd = 1 - d/d_max distinguishes near-misses from wild misses:
    - d: Distance from observed data to GARP-consistent region
    - d_max: Maximum possible distance

    This allows computing a generalized predictive success md = rd - a
    even for violating data.

    Args:
        session: ConsumerSession with prices and quantities
        n_simulations: Number of simulations for area and distance estimation
        algorithm: Random bundle generation algorithm
        tolerance: Numerical tolerance
        random_seed: Optional seed for reproducibility

    Returns:
        SmoothedHitRateResult with rd, distance metrics, and interpretation

    Example:
        >>> from pyrevealed import BehaviorLog, compute_smoothed_hit_rate
        >>> result = compute_smoothed_hit_rate(log)
        >>> print(f"Smoothed hit rate: {result.smoothed_rate:.3f}")
        >>> if result.distance < result.max_distance / 2:
        ...     print("Near miss - behavior is close to rational")

    References:
        Beatty, T. K., & Crawford, I. A. (2011). Section IV.
    """
    start_time = time.perf_counter()

    if random_seed is not None:
        np.random.seed(random_seed)

    from pyrevealed.algorithms.garp import check_garp
    from pyrevealed.algorithms.aei import compute_aei

    # Check if data satisfies GARP
    garp_result = check_garp(session, tolerance)

    if garp_result.is_consistent:
        # No violations, smoothed rate = 1
        computation_time = (time.perf_counter() - start_time) * 1000
        return SmoothedHitRateResult(
            smoothed_rate=1.0,
            distance=0.0,
            max_distance=1.0,
            aei=1.0,
            is_consistent=True,
            computation_time_ms=computation_time,
        )

    # Compute distance using AEI (1 - AEI measures deviation from rationality)
    aei_result = compute_aei(session, tolerance=tolerance)
    aei = aei_result.efficiency_index

    # Distance from consistency: d = 1 - AEI
    # Max distance = 1 (when AEI = 0)
    distance = 1.0 - aei
    max_distance = 1.0

    # Smoothed hit rate: rd = 1 - d/d_max = AEI
    smoothed_rate = aei

    computation_time = (time.perf_counter() - start_time) * 1000

    return SmoothedHitRateResult(
        smoothed_rate=smoothed_rate,
        distance=distance,
        max_distance=max_distance,
        aei=aei,
        is_consistent=False,
        computation_time_ms=computation_time,
    )


# =============================================================================
# GENERALIZED PREDICTIVE SUCCESS (Beatty & Crawford 2011)
# =============================================================================


def compute_generalized_predictive_success(
    session: "ConsumerSession",
    n_simulations: int = 1000,
    algorithm: int = 1,
    tolerance: float = 1e-10,
    random_seed: int | None = None,
) -> SeltenMeasureResult:
    """
    Compute generalized predictive success md = rd - a.

    This extends Selten's measure to handle GARP violators by using
    the smoothed hit rate rd instead of binary pass rate r.

    Interpretation:
    - md > 0: Data is closer to rational than random behavior
    - md ≈ 0: Data performs about as well as random
    - md < 0: Data is further from rational than random behavior

    Args:
        session: ConsumerSession with prices and quantities
        n_simulations: Number of Monte Carlo simulations
        algorithm: Random bundle generation algorithm
        tolerance: Numerical tolerance
        random_seed: Optional seed for reproducibility

    Returns:
        SeltenMeasureResult with generalized measure

    Example:
        >>> from pyrevealed import BehaviorLog, compute_generalized_predictive_success
        >>> result = compute_generalized_predictive_success(log)
        >>> print(f"Generalized m: {result.measure:.3f}")
    """
    start_time = time.perf_counter()

    # Compute smoothed hit rate
    smoothed_result = compute_smoothed_hit_rate(
        session, n_simulations, algorithm, tolerance, random_seed
    )
    smoothed_rate = smoothed_result.smoothed_rate

    # Compute relative area
    area_result = compute_relative_area(
        session, n_simulations, algorithm, tolerance, random_seed
    )
    relative_area = area_result.relative_area

    # Generalized measure: md = rd - a
    measure = smoothed_rate - relative_area

    computation_time = (time.perf_counter() - start_time) * 1000

    return SeltenMeasureResult(
        measure=measure,
        pass_rate=smoothed_rate,  # Using smoothed rate
        relative_area=relative_area,
        n_simulations=n_simulations,
        algorithm=algorithm,
        is_meaningful=measure > 0.05,
        computation_time_ms=computation_time,
    )


# =============================================================================
# BAYESIAN CREDIBILITY (Crawford 2019 mini-course)
# =============================================================================


def compute_bayesian_credibility(
    session: "ConsumerSession",
    prior_rational: float = 0.5,
    n_simulations: int = 1000,
    algorithm: int = 1,
    tolerance: float = 1e-10,
    random_seed: int | None = None,
) -> BayesianCredibilityResult:
    """
    Compute Bayesian posterior probability of utility maximization.

    Uses Bayes' rule to update beliefs about rationality:
        P(Rational | Pass GARP) = P(R) / [P(R) + P(G|~R)(1-P(R))]

    where:
    - P(R): Prior probability consumer is rational
    - P(G|R) = 1: Rational consumer always passes GARP
    - P(G|~R) = relative_area: Probability random passes GARP

    Key insight: If test has low power (P(G|~R) ≈ 1), passing GARP
    doesn't update beliefs much. Need high power for strong inference.

    Args:
        session: ConsumerSession with prices and quantities
        prior_rational: Prior probability of utility maximization (default: 0.5)
        n_simulations: Monte Carlo simulations for area estimation
        algorithm: Random bundle generation algorithm
        tolerance: Numerical tolerance
        random_seed: Optional seed for reproducibility

    Returns:
        BayesianCredibilityResult with posterior and likelihood ratio

    Example:
        >>> from pyrevealed import BehaviorLog, compute_bayesian_credibility
        >>> result = compute_bayesian_credibility(log, prior_rational=0.5)
        >>> print(f"Posterior P(rational): {result.posterior:.3f}")
        >>> print(f"Bayes factor: {result.bayes_factor:.2f}")
        >>> if result.posterior > 0.9:
        ...     print("Strong evidence for utility maximization")

    References:
        Crawford, I. (2019). Mini Course on Empirical Revealed Preference.
        University of Oxford lecture notes.
    """
    start_time = time.perf_counter()

    from pyrevealed.algorithms.garp import check_garp

    # Check if data passes GARP
    garp_result = check_garp(session, tolerance)
    passes_garp = garp_result.is_consistent

    # Compute relative area (probability random passes)
    area_result = compute_relative_area(
        session, n_simulations, algorithm, tolerance, random_seed
    )
    p_pass_given_random = area_result.relative_area

    # Likelihoods
    p_pass_given_rational = 1.0  # Rational always passes

    if passes_garp:
        # Bayes' rule: P(R|G) = P(G|R)P(R) / [P(G|R)P(R) + P(G|~R)P(~R)]
        numerator = p_pass_given_rational * prior_rational
        denominator = numerator + p_pass_given_random * (1 - prior_rational)

        if denominator > 0:
            posterior = numerator / denominator
        else:
            posterior = prior_rational  # Edge case

        # Bayes factor = P(G|R) / P(G|~R)
        if p_pass_given_random > 0:
            bayes_factor = p_pass_given_rational / p_pass_given_random
        else:
            bayes_factor = float("inf")
    else:
        # Data fails GARP - posterior depends on whether we allow noise
        # Under strict model: P(R|~G) = 0
        # Under noisy model: would need error model
        posterior = 0.0
        bayes_factor = 0.0

    # Strength of evidence interpretation (Jeffreys scale)
    if bayes_factor > 100:
        evidence_strength = "decisive"
    elif bayes_factor > 30:
        evidence_strength = "very_strong"
    elif bayes_factor > 10:
        evidence_strength = "strong"
    elif bayes_factor > 3:
        evidence_strength = "moderate"
    elif bayes_factor > 1:
        evidence_strength = "anecdotal"
    else:
        evidence_strength = "against"

    computation_time = (time.perf_counter() - start_time) * 1000

    return BayesianCredibilityResult(
        posterior=posterior,
        prior=prior_rational,
        likelihood_ratio=bayes_factor,
        bayes_factor=bayes_factor,
        p_pass_given_rational=p_pass_given_rational,
        p_pass_given_random=p_pass_given_random,
        passes_garp=passes_garp,
        evidence_strength=evidence_strength,
        computation_time_ms=computation_time,
    )


# =============================================================================
# OPTIMAL AFRIAT EFFICIENCY (Beatty & Crawford 2011)
# =============================================================================


def compute_optimal_efficiency(
    session: "ConsumerSession",
    n_simulations: int = 500,
    n_efficiency_levels: int = 20,
    algorithm: int = 1,
    tolerance: float = 1e-6,
    random_seed: int | None = None,
) -> OptimalEfficiencyResult:
    """
    Find efficiency level e* that maximizes predictive success m(e).

    Instead of testing GARP at e=1, this function searches across efficiency
    levels to find e* = argmax_e [r(e) - a(e)], where:
    - r(e) = 1 if data passes GARP at efficiency e, else 0
    - a(e) = relative area at efficiency e (Monte Carlo estimate)

    This is useful because:
    1. At high efficiency, random behavior rarely violates GARP (low power)
    2. At low efficiency, even consistent data may violate (low signal)
    3. The optimal e* balances these tradeoffs

    Args:
        session: ConsumerSession with prices and quantities
        n_simulations: Monte Carlo simulations per efficiency level
        n_efficiency_levels: Number of efficiency levels to test (grid size)
        algorithm: Random bundle generation algorithm (1, 2, or 3)
            - 1: Uniform on budget hyperplane (corners equally likely)
            - 2: N i.i.d. uniform, normalized (corners rare)
            - 3: Algorithm 2 weighted by actual mean budget shares
        tolerance: Numerical tolerance for GARP detection
        random_seed: Optional seed for reproducibility

    Returns:
        OptimalEfficiencyResult with e*, m(e*), and full grid of values

    Example:
        >>> from pyrevealed import BehaviorLog, compute_optimal_efficiency
        >>> result = compute_optimal_efficiency(log, n_simulations=200)
        >>> print(f"Optimal e*: {result.optimal_efficiency:.3f}")
        >>> print(f"Max m(e*): {result.optimal_measure:.3f}")
        >>> if result.optimal_efficiency < result.aei:
        ...     print("Relaxing efficiency improves signal strength")

    References:
        Beatty, T. K., & Crawford, I. A. (2011). How demanding is the
        revealed preference approach to demand? AER, 101(6), 2782-2795.
    """
    start_time = time.perf_counter()

    if random_seed is not None:
        np.random.seed(random_seed)

    from pyrevealed.algorithms.garp import check_garp
    from pyrevealed.algorithms.aei import compute_aei
    from pyrevealed.core.session import ConsumerSession as CS

    prices = session.prices
    expenditures = session.own_expenditures

    # First compute the AEI to know the range where data passes GARP
    aei_result = compute_aei(session, tolerance=tolerance)
    aei = aei_result.efficiency_index

    # Compute mean budget shares for algorithm 3
    if algorithm == 3:
        quantities = session.quantities
        mean_shares = _compute_mean_budget_shares(prices, quantities, expenditures)
    else:
        mean_shares = None

    # Create grid of efficiency levels from near 0 to 1
    efficiency_levels = np.linspace(0.05, 1.0, n_efficiency_levels).tolist()

    pass_rates: list[float] = []
    relative_areas: list[float] = []
    measures: list[float] = []

    for e_level in efficiency_levels:
        # Check if actual data passes GARP at this efficiency level
        # We deflate budgets by efficiency factor
        if e_level >= aei - tolerance:
            # Data passes GARP at this efficiency level
            pass_rate = 1.0
        else:
            # Data violates GARP at this efficiency level
            pass_rate = 0.0

        # Estimate relative area at this efficiency level via Monte Carlo
        n_consistent = 0
        for _ in range(n_simulations):
            # Generate random quantities
            random_quantities = _generate_random_bundles_algorithm(
                prices, expenditures, algorithm, mean_shares
            )

            # Create temporary session and check GARP at this efficiency
            random_session = CS(prices=prices, quantities=random_quantities)
            random_aei = compute_aei(random_session, tolerance=tolerance)

            # Random data passes GARP at efficiency e if its AEI >= e
            if random_aei.efficiency_index >= e_level - tolerance:
                n_consistent += 1

        relative_area = n_consistent / n_simulations

        # Compute Selten's measure at this efficiency level
        measure = pass_rate - relative_area

        pass_rates.append(pass_rate)
        relative_areas.append(relative_area)
        measures.append(measure)

    # Find optimal efficiency level (argmax of measures)
    max_measure_idx = np.argmax(measures)
    optimal_efficiency = efficiency_levels[max_measure_idx]
    optimal_measure = measures[max_measure_idx]

    computation_time = (time.perf_counter() - start_time) * 1000

    return OptimalEfficiencyResult(
        optimal_efficiency=optimal_efficiency,
        optimal_measure=optimal_measure,
        efficiency_levels=efficiency_levels,
        measures=measures,
        pass_rates=pass_rates,
        relative_areas=relative_areas,
        aei=aei,
        computation_time_ms=computation_time,
    )


# Tech-friendly alias
compute_optimal_predictive_efficiency = compute_optimal_efficiency
"""
Find efficiency level that maximizes predictive success.

This searches across efficiency levels to find the level at which
utility maximization provides the strongest evidence relative to random.

Use this to identify the "most informative" efficiency threshold.
"""


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

# compute_power_metric: Tech-friendly name for compute_selten_measure
compute_power_metric = compute_selten_measure
"""
Compute power metric m = r - a (Selten's predictive success).

This measures how meaningful a GARP test result is by comparing
the actual pass/fail result to what random behavior would produce.

Use this to answer: "Does passing GARP mean anything for this data?"
"""

# compute_test_demandingness: Tech-friendly name for compute_relative_area
compute_test_demandingness = compute_relative_area
"""
Compute how demanding the GARP test is for given prices.

The relative area measures the probability that random behavior
would satisfy GARP. Low area = demanding test, high area = easy test.

Use this to assess the quality of price variation in your data.
"""

# compute_near_miss_score: Tech-friendly name for compute_smoothed_hit_rate
compute_near_miss_score = compute_smoothed_hit_rate
"""
Compute near-miss score for GARP violators.

For data that violates GARP, this distinguishes near-misses
(almost rational) from wild misses (far from rational).

Use this when you want a continuous measure even for violators.
"""

# compute_rationality_posterior: Tech-friendly alias
compute_rationality_posterior = compute_bayesian_credibility
"""
Compute Bayesian posterior probability of rational behavior.

Updates prior beliefs about rationality given the GARP test result
and the power of the test for this price configuration.

Use this for principled statistical inference about rationality.
"""
