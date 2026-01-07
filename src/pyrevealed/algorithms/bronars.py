"""Bronars' Power Index for statistical significance of GARP tests."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from pyrevealed.core.session import ConsumerSession
from pyrevealed.core.result import BronarsPowerResult


def compute_bronars_power(
    session: ConsumerSession,
    n_simulations: int = 1000,
    tolerance: float = 1e-10,
    random_seed: int | None = None,
    store_simulation_values: bool = True,
) -> BronarsPowerResult:
    """
    Compute Bronars' Power Index for statistical significance of GARP test.

    Bronars' Power Index measures the discriminatory power of the GARP test
    for the given price configuration. It answers: "If a user passed GARP,
    is that result statistically meaningful?"

    The algorithm simulates random behavior on the observed budget constraints:
    1. For each observation, generate random quantities that exhaust the budget
    2. Check if the random behavior violates GARP
    3. Power = fraction of random behaviors that violate GARP

    Interpretation:
    - Power > 0.7: High power, passing GARP is statistically significant
    - Power 0.5-0.7: Moderate power, interpret with caution
    - Power < 0.5: Low power, even random behavior would likely pass GARP

    Args:
        session: ConsumerSession with prices and quantities
        n_simulations: Number of random behavior simulations (default: 1000)
        tolerance: Numerical tolerance for GARP detection
        random_seed: Optional seed for reproducibility
        store_simulation_values: If True, store individual AEI values (default: True)

    Returns:
        BronarsPowerResult with power index and simulation details

    Example:
        >>> import numpy as np
        >>> from pyrevealed import ConsumerSession, compute_bronars_power
        >>> prices = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
        >>> quantities = np.array([[4.0, 1.0], [1.0, 4.0], [2.0, 2.0]])
        >>> session = ConsumerSession(prices=prices, quantities=quantities)
        >>> result = compute_bronars_power(session, n_simulations=500)
        >>> print(f"Power: {result.power_index:.3f}")
        >>> if result.is_significant:
        ...     print("GARP test has good discriminatory power")

    References:
        Bronars, S. G. (1987). The power of nonparametric tests of preference
        maximization. Econometrica, 55(3), 693-698.
    """
    start_time = time.perf_counter()

    if random_seed is not None:
        np.random.seed(random_seed)

    # Import here to avoid circular imports
    from pyrevealed.algorithms.garp import check_garp
    from pyrevealed.algorithms.aei import compute_aei

    T = session.num_observations
    N = session.num_goods
    prices = session.prices
    expenditures = session.own_expenditures  # e_i = p_i @ x_i

    n_violations = 0
    aei_values = np.zeros(n_simulations) if store_simulation_values else None

    for sim in range(n_simulations):
        # Generate random quantities on budget hyperplanes
        random_quantities = _generate_random_bundles(prices, expenditures)

        # Create temporary session with random quantities
        random_session = ConsumerSession(
            prices=prices,
            quantities=random_quantities,
        )

        # Check GARP
        garp_result = check_garp(random_session, tolerance)

        if not garp_result.is_consistent:
            n_violations += 1

        # Compute AEI for detailed analysis
        if store_simulation_values:
            aei_result = compute_aei(random_session, tolerance=1e-4, max_iterations=20)
            aei_values[sim] = aei_result.efficiency_index

    power_index = n_violations / n_simulations

    # Compute mean AEI of random simulations
    if store_simulation_values and aei_values is not None:
        mean_aei = float(np.mean(aei_values))
    else:
        mean_aei = 0.0

    computation_time = (time.perf_counter() - start_time) * 1000

    return BronarsPowerResult(
        power_index=power_index,
        is_significant=power_index > 0.5,
        n_simulations=n_simulations,
        n_violations=n_violations,
        mean_integrity_random=mean_aei,
        simulation_integrity_values=aei_values,
        computation_time_ms=computation_time,
    )


def _generate_random_bundles(
    prices: NDArray[np.float64],
    expenditures: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Generate random bundles uniformly distributed on budget hyperplanes.

    For each observation i, generates quantity vector q_i such that:
    - q_i >= 0 (non-negative quantities)
    - p_i @ q_i = e_i (budget constraint satisfied exactly)

    Uses Dirichlet distribution to generate uniform budget shares.

    Args:
        prices: T x N price matrix
        expenditures: T-length array of budgets (e_i = p_i @ x_i)

    Returns:
        T x N matrix of random quantities
    """
    T, N = prices.shape
    random_quantities = np.zeros((T, N))

    for i in range(T):
        # Generate random budget shares using symmetric Dirichlet(1, 1, ..., 1)
        # This gives uniform distribution over the simplex
        shares = np.random.dirichlet(np.ones(N))

        # Convert to quantities: q_ij = share_j * e_i / p_ij
        # Budget share for good j: (p_ij * q_ij) / e_i = share_j
        # So q_ij = share_j * e_i / p_ij
        random_quantities[i] = shares * expenditures[i] / prices[i]

    return random_quantities


def compute_bronars_power_fast(
    session: ConsumerSession,
    n_simulations: int = 1000,
    tolerance: float = 1e-10,
    random_seed: int | None = None,
) -> BronarsPowerResult:
    """
    Fast version of Bronars' Power Index (binary GARP only, no AEI).

    This version only checks binary GARP pass/fail for each simulation,
    which is faster but doesn't provide mean_integrity_random.

    Args:
        session: ConsumerSession with prices and quantities
        n_simulations: Number of random behavior simulations (default: 1000)
        tolerance: Numerical tolerance for GARP detection
        random_seed: Optional seed for reproducibility

    Returns:
        BronarsPowerResult with power index (mean_integrity_random will be 0.0)
    """
    start_time = time.perf_counter()

    if random_seed is not None:
        np.random.seed(random_seed)

    from pyrevealed.algorithms.garp import check_garp

    prices = session.prices
    expenditures = session.own_expenditures

    n_violations = 0

    for _ in range(n_simulations):
        random_quantities = _generate_random_bundles(prices, expenditures)
        random_session = ConsumerSession(prices=prices, quantities=random_quantities)
        garp_result = check_garp(random_session, tolerance)

        if not garp_result.is_consistent:
            n_violations += 1

    power_index = n_violations / n_simulations
    computation_time = (time.perf_counter() - start_time) * 1000

    return BronarsPowerResult(
        power_index=power_index,
        is_significant=power_index > 0.5,
        n_simulations=n_simulations,
        n_violations=n_violations,
        mean_integrity_random=0.0,
        simulation_integrity_values=None,
        computation_time_ms=computation_time,
    )


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

# compute_test_power: Tech-friendly name for compute_bronars_power
compute_test_power = compute_bronars_power
"""
Compute the statistical power of the consistency test.

This is the tech-friendly alias for compute_bronars_power.

The test power measures whether a passed consistency test is meaningful:
- Power > 0.7: High discriminatory power, passing GARP is significant
- Power 0.5-0.7: Moderate power, interpret with caution
- Power < 0.5: Low power, even random behavior would pass

Use this for:
- Validating that consistency scores are meaningful
- Determining if more observations are needed
- Assessing quality of price variation in data

Example:
    >>> from pyrevealed import BehaviorLog, compute_test_power
    >>> result = compute_test_power(user_log, n_simulations=500)
    >>> if result.power_index < 0.5:
    ...     print("Warning: GARP test has low discriminatory power")
"""

compute_test_power_fast = compute_bronars_power_fast
"""
Fast version of compute_test_power (binary pass/fail only).

Use when you only need the power index and don't need mean_integrity_random.
"""
