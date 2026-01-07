"""Pytest fixtures for PyRevealed tests."""

import numpy as np
import pytest

from pyrevealed import ConsumerSession


@pytest.fixture
def simple_consistent_session() -> ConsumerSession:
    """
    Session with 3 observations that satisfies GARP.

    When good A is relatively cheap, consumer buys more A.
    When good B is relatively cheap, consumer buys more B.
    This is consistent rational behavior.
    """
    prices = np.array([
        [1.0, 2.0],  # Obs 0: A is cheap
        [2.0, 1.0],  # Obs 1: B is cheap
        [1.5, 1.5],  # Obs 2: Equal prices
    ])
    quantities = np.array([
        [4.0, 1.0],  # Chose more of A (cheap good)
        [1.0, 4.0],  # Chose more of B (cheap good)
        [2.0, 2.0],  # Balanced choice
    ])
    return ConsumerSession(prices=prices, quantities=quantities, session_id="consistent")


@pytest.fixture
def simple_violation_session() -> ConsumerSession:
    """
    Session with clear GARP violation (WARP violation).

    Each bundle is much cheaper at the OTHER observation's prices,
    creating strict preferences in both directions - a clear cycle.

    At obs 0 prices [1, 0.1]: bundle 1 costs 0.1 << 1 (strictly preferred 0 over 1)
    At obs 1 prices [0.1, 1]: bundle 0 costs 0.1 << 1 (strictly preferred 1 over 0)
    This creates a cycle 0 -> 1 -> 0 with both edges strict.
    """
    prices = np.array([
        [1.0, 0.1],   # Good A expensive, Good B cheap
        [0.1, 1.0],   # Good A cheap, Good B expensive
    ])
    quantities = np.array([
        [1.0, 0.0],   # Chose only A (the expensive good!)
        [0.0, 1.0],   # Chose only B (the expensive good!)
    ])
    return ConsumerSession(prices=prices, quantities=quantities, session_id="violation")


@pytest.fixture
def borderline_session() -> ConsumerSession:
    """
    Session that is close to violating GARP but doesn't.

    The bundles are barely affordable at each other's prices,
    so no strict preference is revealed.
    """
    prices = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
    ])
    quantities = np.array([
        [2.0, 2.0],  # Total: 4
        [2.0, 2.0],  # Total: 4 (same bundle, no violation possible)
    ])
    return ConsumerSession(prices=prices, quantities=quantities, session_id="borderline")


@pytest.fixture
def three_cycle_violation_session() -> ConsumerSession:
    """
    Session with a length-3 GARP violation cycle.

    Each observation chooses the expensive good while the others are cheap,
    creating strict preferences forming a cycle A -> B -> C -> A.
    """
    prices = np.array([
        [1.0, 0.1, 0.1],  # Good A expensive
        [0.1, 1.0, 0.1],  # Good B expensive
        [0.1, 0.1, 1.0],  # Good C expensive
    ])
    quantities = np.array([
        [1.0, 0.0, 0.0],  # Chose A (expensive!)
        [0.0, 1.0, 0.0],  # Chose B (expensive!)
        [0.0, 0.0, 1.0],  # Chose C (expensive!)
    ])
    return ConsumerSession(prices=prices, quantities=quantities, session_id="three_cycle")


@pytest.fixture
def large_random_session() -> ConsumerSession:
    """
    Large session for performance testing.

    Random data that may or may not satisfy GARP.
    """
    np.random.seed(42)
    T, N = 100, 10
    prices = np.random.uniform(0.5, 2.0, (T, N))
    quantities = np.random.uniform(0.0, 10.0, (T, N))
    return ConsumerSession(prices=prices, quantities=quantities, session_id="large_random")


@pytest.fixture
def single_observation_session() -> ConsumerSession:
    """
    Session with only one observation.

    Trivially satisfies GARP (no pairs to compare).
    """
    prices = np.array([[1.0, 2.0]])
    quantities = np.array([[3.0, 1.0]])
    return ConsumerSession(prices=prices, quantities=quantities, session_id="single")


@pytest.fixture
def many_goods_session() -> ConsumerSession:
    """
    Session with many goods (5) for testing scalability.
    """
    prices = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [5.0, 4.0, 3.0, 2.0, 1.0],
        [3.0, 3.0, 3.0, 3.0, 3.0],
    ])
    quantities = np.array([
        [5.0, 2.0, 1.0, 0.5, 0.2],  # Prefer cheap goods
        [0.2, 0.5, 1.0, 2.0, 5.0],  # Prefer cheap goods
        [1.0, 1.0, 1.0, 1.0, 1.0],  # Equal
    ])
    return ConsumerSession(prices=prices, quantities=quantities, session_id="many_goods")


# =============================================================================
# LANCASTER CHARACTERISTICS MODEL FIXTURES
# =============================================================================


@pytest.fixture
def simple_lancaster_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple 2-product, 2-characteristic Lancaster setup.

    Products: Apple (95 cal, 4.4g fiber), Banana (105 cal, 3.1g fiber)
    Characteristics: calories, fiber

    Returns:
        Tuple of (prices, quantities, attribute_matrix)
    """
    attribute_matrix = np.array([
        [95.0, 4.4],   # Apple: calories, fiber per unit
        [105.0, 3.1],  # Banana: calories, fiber per unit
    ])
    prices = np.array([
        [1.0, 0.5],
        [0.8, 0.6],
        [1.2, 0.4],
    ])
    quantities = np.array([
        [2.0, 3.0],
        [4.0, 1.0],
        [1.0, 5.0],
    ])
    return prices, quantities, attribute_matrix


@pytest.fixture
def well_specified_lancaster_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Well-specified Lancaster setup: N=3 products, K=2 characteristics, full rank.

    The attribute matrix has full column rank (rank = K = 2).

    Returns:
        Tuple of (prices, quantities, attribute_matrix)
    """
    attribute_matrix = np.array([
        [1.0, 0.0],   # Product 1: only characteristic 1
        [0.0, 1.0],   # Product 2: only characteristic 2
        [0.5, 0.5],   # Product 3: both characteristics
    ])
    prices = np.array([
        [2.0, 3.0, 2.5],
        [1.5, 4.0, 2.75],
    ])
    quantities = np.array([
        [3.0, 2.0, 1.0],
        [1.0, 1.0, 4.0],
    ])
    return prices, quantities, attribute_matrix


@pytest.fixture
def identity_attribute_matrix() -> np.ndarray:
    """
    Identity-like attribute matrix where products map 1:1 to characteristics.

    Useful for testing that shadow prices equal product prices.
    """
    return np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])


@pytest.fixture
def rank_deficient_attribute_matrix() -> np.ndarray:
    """
    Attribute matrix with K=3 but only rank=2 (linearly dependent columns).

    Used to test rank-deficiency warnings.
    """
    return np.array([
        [1.0, 2.0, 3.0],  # Product 1
        [2.0, 4.0, 6.0],  # Product 2: linearly dependent (2 * Product 1)
    ])
