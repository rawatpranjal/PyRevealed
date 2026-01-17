"""
Pathological fixtures for EVALs - designed to break algorithms.

Each fixture creates data that targets specific numerical or algorithmic weaknesses.
"""

import numpy as np
import pytest
from pyrevealed.core.session import (
    BehaviorLog,
    MenuChoiceLog,
    StochasticChoiceLog,
    RiskChoiceLog,
    ProductionLog,
)


# =============================================================================
# MINIMAL DATA FIXTURES (T=1, N=1, empty)
# =============================================================================


@pytest.fixture
def single_observation_log():
    """T=1 - breaks algorithms expecting T>=2 for pairwise comparisons."""
    return BehaviorLog(
        cost_vectors=np.array([[1.0, 2.0]]),
        action_vectors=np.array([[3.0, 1.0]]),
    )


@pytest.fixture
def single_feature_log():
    """N=1 single good - creates degenerate matrices."""
    return BehaviorLog(
        cost_vectors=np.array([[1.0], [1.5], [2.0]]),
        action_vectors=np.array([[5.0], [4.0], [3.0]]),
    )


@pytest.fixture
def two_observations_log():
    """T=2 - minimum for pairwise comparison but edge case for many algorithms."""
    return BehaviorLog(
        cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
        action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
    )


@pytest.fixture
def empty_menu_choice_log():
    """Single-item menus - degenerate for choice analysis."""
    return MenuChoiceLog(
        menus=[frozenset({0}), frozenset({1}), frozenset({2})],
        choices=[0, 1, 2],
    )


@pytest.fixture
def single_menu_stochastic_log():
    """Single menu stochastic data - minimal for frequency-based tests."""
    return StochasticChoiceLog(
        menus=[frozenset({0, 1, 2})],
        choice_frequencies=[{0: 50, 1: 30, 2: 20}],
    )


# =============================================================================
# NEAR-ZERO NUMERICAL HAZARDS
# =============================================================================


@pytest.fixture
def near_zero_prices():
    """Prices at edge of numerical precision - 1e-300."""
    return BehaviorLog(
        cost_vectors=np.array([[1e-300, 1e-300], [1e-300, 1e-300]]),
        action_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
    )


@pytest.fixture
def near_zero_quantities():
    """Quantities at edge of numerical precision."""
    return BehaviorLog(
        cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
        action_vectors=np.array([[1e-300, 1e-300], [1e-300, 1e-300]]),
    )


@pytest.fixture
def mixed_scale_prices():
    """Prices varying by 15 orders of magnitude."""
    return BehaviorLog(
        cost_vectors=np.array([[1e-7, 1e8], [1e-7, 1e8]]),
        action_vectors=np.array([[1e7, 1e-8], [1e7, 1e-8]]),
    )


@pytest.fixture
def near_tolerance_expenditure_log():
    """Expenditures differing by exactly the tolerance amount."""
    # p1 @ q1 = 5.0, p1 @ q2 = 5.0 + 1e-10 (exactly at tolerance boundary)
    return BehaviorLog(
        cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
        action_vectors=np.array([[2.5, 2.5], [2.5 + 5e-11, 2.5 + 5e-11]]),
    )


# =============================================================================
# SINGULAR/ILL-CONDITIONED MATRICES
# =============================================================================


@pytest.fixture
def multicollinear_prices():
    """Perfectly correlated prices - singular regression matrix."""
    # p2 = 2 * p1 for all observations
    return BehaviorLog(
        cost_vectors=np.array([
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
        ]),
        action_vectors=np.array([
            [4.0, 1.0],
            [2.0, 1.5],
            [1.5, 1.0],
            [1.0, 0.8],
        ]),
    )


@pytest.fixture
def constant_prices():
    """All prices identical - zero variance in design matrix."""
    return BehaviorLog(
        cost_vectors=np.array([
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]),
        action_vectors=np.array([
            [4.0, 1.0],
            [3.0, 2.0],
            [2.0, 3.0],
            [1.0, 4.0],
        ]),
    )


@pytest.fixture
def proportional_bundles():
    """All bundles are scalar multiples - rank deficient."""
    return BehaviorLog(
        cost_vectors=np.array([
            [1.0, 2.0],
            [2.0, 1.0],
            [1.5, 1.5],
        ]),
        action_vectors=np.array([
            [2.0, 4.0],  # q = (2, 4)
            [4.0, 8.0],  # q = 2 * (2, 4)
            [3.0, 6.0],  # q = 1.5 * (2, 4)
        ]),
    )


@pytest.fixture
def high_condition_number_log():
    """Ill-conditioned expenditure matrix - condition number > 1e15."""
    return BehaviorLog(
        cost_vectors=np.array([
            [1e15, 1e-15],
            [1e-15, 1e15],
        ]),
        action_vectors=np.array([
            [1e-15, 1e15],
            [1e15, 1e-15],
        ]),
    )


# =============================================================================
# EXTREME VALUES
# =============================================================================


@pytest.fixture
def max_float_prices():
    """Prices near float64 maximum."""
    max_val = np.finfo(np.float64).max / 1000  # Slightly below max to allow multiplication
    return BehaviorLog(
        cost_vectors=np.array([[max_val, max_val], [max_val, max_val]]),
        action_vectors=np.array([[1e-300, 1e-300], [1e-300, 1e-300]]),
    )


@pytest.fixture
def subnormal_values():
    """Values in subnormal float range."""
    tiny = np.finfo(np.float64).tiny / 10
    return BehaviorLog(
        cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),  # Normal prices
        action_vectors=np.array([[tiny, tiny], [tiny, tiny]]),
    )


@pytest.fixture
def extreme_ratio_log():
    """Prices with ratio of 1e300 between goods."""
    return BehaviorLog(
        cost_vectors=np.array([[1e150, 1e-150], [1e150, 1e-150]]),
        action_vectors=np.array([[1e-150, 1e150], [1e-150, 1e150]]),
    )


# =============================================================================
# STOCHASTIC CHOICE EDGE CASES
# =============================================================================


@pytest.fixture
def zero_frequency_stochastic():
    """All zero frequencies in one menu."""
    return StochasticChoiceLog(
        menus=[frozenset({0, 1, 2}), frozenset({0, 1})],
        choice_frequencies=[{0: 0, 1: 0, 2: 0}, {0: 50, 1: 50}],
    )


@pytest.fixture
def single_choice_stochastic():
    """Deterministic choice (100% probability)."""
    return StochasticChoiceLog(
        menus=[frozenset({0, 1, 2})],
        choice_frequencies=[{0: 100, 1: 0, 2: 0}],
    )


@pytest.fixture
def many_items_stochastic():
    """7 items - triggers factorial explosion in RUM (7! = 5040 orderings)."""
    items = frozenset(range(7))
    freqs = {i: 10 + i for i in range(7)}
    return StochasticChoiceLog(
        menus=[items],
        choice_frequencies=[freqs],
    )


@pytest.fixture
def nested_menus_stochastic():
    """Nested menus for regularity testing - A subset B subset C."""
    return StochasticChoiceLog(
        menus=[
            frozenset({0}),
            frozenset({0, 1}),
            frozenset({0, 1, 2}),
        ],
        choice_frequencies=[
            {0: 100},
            {0: 60, 1: 40},
            {0: 45, 1: 35, 2: 20},
        ],
    )


# =============================================================================
# INTERTEMPORAL EDGE CASES
# =============================================================================


@pytest.fixture
def extreme_delay_choices():
    """Delays up to t=800 - delta^800 underflows for delta < 1."""
    from pyrevealed.algorithms.intertemporal import DatedChoice
    return [
        DatedChoice(
            amounts=np.array([100.0, 200.0]),
            dates=np.array([0, 800]),
            chosen=1,  # Chose the delayed option
        ),
        DatedChoice(
            amounts=np.array([100.0, 150.0]),
            dates=np.array([0, 1]),
            chosen=0,  # Chose immediate
        ),
    ]


@pytest.fixture
def zero_amount_choices():
    """Zero consumption amounts - log(0) issues."""
    from pyrevealed.algorithms.intertemporal import DatedChoice
    return [
        DatedChoice(
            amounts=np.array([0.0, 100.0]),
            dates=np.array([0, 1]),
            chosen=1,
        ),
    ]


@pytest.fixture
def identical_timing_choices():
    """All options at same time - no discounting possible."""
    from pyrevealed.algorithms.intertemporal import DatedChoice
    return [
        DatedChoice(
            amounts=np.array([100.0, 150.0]),
            dates=np.array([5, 5]),  # Same date
            chosen=1,
        ),
    ]


# =============================================================================
# LARGE DATA FOR PERFORMANCE
# =============================================================================


@pytest.fixture
def large_t_500():
    """T=500 observations - O(T^3) stress for Floyd-Warshall."""
    np.random.seed(42)
    T, N = 500, 5
    return BehaviorLog(
        cost_vectors=np.random.rand(T, N) + 0.1,
        action_vectors=np.random.rand(T, N) + 0.1,
    )


@pytest.fixture
def large_t_1000():
    """T=1000 observations - should test <60s completion."""
    np.random.seed(42)
    T, N = 1000, 5
    return BehaviorLog(
        cost_vectors=np.random.rand(T, N) + 0.1,
        action_vectors=np.random.rand(T, N) + 0.1,
    )


@pytest.fixture
def large_n_100():
    """N=100 goods - stress for LP solvers in utility recovery."""
    np.random.seed(42)
    T, N = 20, 100
    return BehaviorLog(
        cost_vectors=np.random.rand(T, N) + 0.1,
        action_vectors=np.random.rand(T, N) + 0.1,
    )


# =============================================================================
# GARP VIOLATION PATTERNS
# =============================================================================


@pytest.fixture
def warp_violation_log():
    """Simple WARP violation - direct contradiction."""
    return BehaviorLog(
        cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
        action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
    )


@pytest.fixture
def garp_3_cycle_log():
    """3-cycle GARP violation (WARP holds, GARP fails)."""
    return BehaviorLog(
        cost_vectors=np.array([
            [1.0, 2.0, 3.0],
            [3.0, 1.0, 2.0],
            [2.0, 3.0, 1.0],
        ]),
        action_vectors=np.array([
            [3.0, 0.5, 0.5],
            [0.5, 3.0, 0.5],
            [0.5, 0.5, 3.0],
        ]),
    )


@pytest.fixture
def dense_violation_log():
    """Many overlapping violation cycles."""
    np.random.seed(123)
    T, N = 20, 3
    # Random data tends to have many violations
    return BehaviorLog(
        cost_vectors=np.random.rand(T, N) + 0.1,
        action_vectors=np.random.rand(T, N) * 10,
    )


# =============================================================================
# PRODUCTION LOG EDGE CASES
# =============================================================================


@pytest.fixture
def zero_output_production():
    """Zero output quantities - profit calculation issues."""
    return ProductionLog(
        input_prices=np.array([[1.0, 2.0]]),
        input_quantities=np.array([[10.0, 5.0]]),
        output_prices=np.array([[10.0]]),
        output_quantities=np.array([[0.0]]),  # Zero output
    )


@pytest.fixture
def negative_profit_production():
    """Production with negative profit."""
    return ProductionLog(
        input_prices=np.array([[10.0, 20.0]]),
        input_quantities=np.array([[10.0, 10.0]]),  # Cost = 300
        output_prices=np.array([[10.0]]),
        output_quantities=np.array([[10.0]]),  # Revenue = 100
    )


# =============================================================================
# RISK CHOICE EDGE CASES
# =============================================================================


@pytest.fixture
def zero_probability_lottery():
    """Lottery with zero probability outcomes."""
    return RiskChoiceLog(
        safe_values=np.array([50.0]),
        risky_outcomes=np.array([[100.0, 0.0]]),
        risky_probabilities=np.array([[1.0, 0.0]]),  # Second outcome impossible
        choices=np.array([True]),
    )


@pytest.fixture
def extreme_outcomes_lottery():
    """Lottery with extreme outcome values."""
    return RiskChoiceLog(
        safe_values=np.array([1e100]),
        risky_outcomes=np.array([[1e150, 1e-150]]),
        risky_probabilities=np.array([[0.5, 0.5]]),
        choices=np.array([True]),
    )


# =============================================================================
# WELFARE ANALYSIS EDGE CASES
# =============================================================================


@pytest.fixture
def baseline_equals_policy_log():
    """Baseline and policy are identical - zero welfare change."""
    log = BehaviorLog(
        cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
        action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
    )
    return log, log  # Same for baseline and policy


@pytest.fixture
def extreme_price_change():
    """Extreme price change - 1000x price ratio."""
    baseline = BehaviorLog(
        cost_vectors=np.array([[1.0, 1.0]]),
        action_vectors=np.array([[5.0, 5.0]]),
    )
    policy = BehaviorLog(
        cost_vectors=np.array([[1000.0, 0.001]]),  # Extreme change
        action_vectors=np.array([[0.005, 5000.0]]),
    )
    return baseline, policy
