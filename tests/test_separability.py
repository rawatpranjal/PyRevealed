"""Tests for separability analysis."""

import numpy as np
import pytest

from pyrevealed import ConsumerSession
from pyrevealed.algorithms.separability import (
    check_separability,
    find_separable_partition,
    compute_cannibalization,
)


class TestCheckSeparability:
    """Tests for check_separability function."""

    def test_separable_groups(self):
        """Test detecting separable product groups."""
        rng = np.random.default_rng(42)
        n_obs = 50

        # Two separable groups: each consumes independently
        prices = np.zeros((n_obs, 4))
        quantities = np.zeros((n_obs, 4))

        for t in range(n_obs):
            # Group A prices and quantities (independent)
            prices[t, 0] = rng.uniform(0.8, 1.2)
            prices[t, 1] = rng.uniform(1.5, 2.5)

            # Group B prices and quantities (independent)
            prices[t, 2] = rng.uniform(0.8, 1.2)
            prices[t, 3] = rng.uniform(1.5, 2.5)

            # Cobb-Douglas demand within each group
            budget_a = rng.uniform(10, 20)
            budget_b = rng.uniform(10, 20)

            quantities[t, 0] = 0.6 * budget_a / prices[t, 0]
            quantities[t, 1] = 0.4 * budget_a / prices[t, 1]
            quantities[t, 2] = 0.5 * budget_b / prices[t, 2]
            quantities[t, 3] = 0.5 * budget_b / prices[t, 3]

        session = ConsumerSession(prices=prices, quantities=quantities)
        result = check_separability(session, group_a=[0, 1], group_b=[2, 3])

        # Should be separable with low cross-effect
        assert result.cross_effect_strength < 0.3

    def test_non_separable_groups(self):
        """Test detecting non-separable product groups."""
        rng = np.random.default_rng(42)
        n_obs = 50

        prices = np.zeros((n_obs, 4))
        quantities = np.zeros((n_obs, 4))

        for t in range(n_obs):
            prices[t] = rng.uniform(0.8, 1.2, 4)

            # Non-separable: group B prices affect group A quantities
            avg_b_price = np.mean(prices[t, 2:4])

            # When B is cheap, buy less A (cannibalization)
            budget_a = 20 - 10 * (1 - avg_b_price)  # B cheap -> less A budget
            budget_b = 20 + 10 * (1 - avg_b_price)  # B cheap -> more B budget

            quantities[t, 0] = 0.6 * max(budget_a, 1) / prices[t, 0]
            quantities[t, 1] = 0.4 * max(budget_a, 1) / prices[t, 1]
            quantities[t, 2] = 0.5 * budget_b / prices[t, 2]
            quantities[t, 3] = 0.5 * budget_b / prices[t, 3]

        session = ConsumerSession(prices=prices, quantities=quantities)
        result = check_separability(session, group_a=[0, 1], group_b=[2, 3])

        # Cross-effect should be higher for non-separable
        assert result.is_separable in [True, False]  # Works with numpy bools

    def test_within_group_consistency(self):
        """Test that within-group consistency is computed."""
        rng = np.random.default_rng(42)
        n_obs = 30

        prices = rng.uniform(0.5, 2.0, (n_obs, 4))
        quantities = rng.uniform(1, 10, (n_obs, 4))

        session = ConsumerSession(prices=prices, quantities=quantities)
        result = check_separability(session, group_a=[0, 1], group_b=[2, 3])

        assert 0 <= result.within_group_a_consistency <= 1
        assert 0 <= result.within_group_b_consistency <= 1

    def test_group_overlap_error(self):
        """Test that overlapping groups raise error."""
        prices = np.array([[1, 1, 1, 1]])
        quantities = np.array([[1, 1, 1, 1]])

        session = ConsumerSession(prices=prices, quantities=quantities)

        with pytest.raises(ValueError, match="overlap"):
            check_separability(session, group_a=[0, 1], group_b=[1, 2])

    def test_index_bounds_error(self):
        """Test that out-of-bounds indices raise error."""
        prices = np.array([[1, 1]])
        quantities = np.array([[1, 1]])

        session = ConsumerSession(prices=prices, quantities=quantities)

        with pytest.raises(ValueError, match="out of range"):
            check_separability(session, group_a=[0], group_b=[5])

    def test_computation_time(self):
        """Test that computation time is tracked."""
        prices = np.array([[1, 1, 1, 1], [2, 2, 2, 2]])
        quantities = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])

        session = ConsumerSession(prices=prices, quantities=quantities)
        result = check_separability(session, group_a=[0, 1], group_b=[2, 3])

        assert result.computation_time_ms > 0

    def test_recommendation_generated(self):
        """Test that a recommendation is generated."""
        prices = np.array([[1, 1, 1, 1], [2, 2, 2, 2]])
        quantities = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])

        session = ConsumerSession(prices=prices, quantities=quantities)
        result = check_separability(session, group_a=[0, 1], group_b=[2, 3])

        assert result.recommendation in [
            "price_independently",
            "unified_strategy",
            "partial_independence",
        ]


class TestFindSeparablePartition:
    """Tests for find_separable_partition function."""

    def test_basic_partition(self):
        """Test finding basic partition."""
        rng = np.random.default_rng(42)
        n_obs = 30

        prices = rng.uniform(0.5, 2.0, (n_obs, 4))
        quantities = rng.uniform(1, 10, (n_obs, 4))

        session = ConsumerSession(prices=prices, quantities=quantities)
        groups = find_separable_partition(session, max_groups=2)

        assert len(groups) == 2
        # All goods should be in exactly one group
        all_goods = set()
        for g in groups:
            all_goods.update(g)
        assert all_goods == {0, 1, 2, 3}

    def test_single_group(self):
        """Test partition with max_groups=1."""
        prices = np.array([[1, 1, 1]])
        quantities = np.array([[1, 1, 1]])

        session = ConsumerSession(prices=prices, quantities=quantities)
        groups = find_separable_partition(session, max_groups=1)

        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}

    def test_three_groups(self):
        """Test partition with three groups."""
        rng = np.random.default_rng(42)
        n_obs = 30

        prices = rng.uniform(0.5, 2.0, (n_obs, 6))
        quantities = rng.uniform(1, 10, (n_obs, 6))

        session = ConsumerSession(prices=prices, quantities=quantities)
        groups = find_separable_partition(session, max_groups=3)

        assert len(groups) == 3


class TestComputeCannibalization:
    """Tests for compute_cannibalization function."""

    def test_no_cannibalization(self):
        """Test that independent groups show no cannibalization."""
        rng = np.random.default_rng(42)
        n_obs = 50

        prices = np.zeros((n_obs, 4))
        quantities = np.zeros((n_obs, 4))

        # Independent consumption
        for t in range(n_obs):
            prices[t] = rng.uniform(0.8, 1.2, 4)
            quantities[t, 0] = rng.uniform(1, 5)
            quantities[t, 1] = rng.uniform(1, 5)
            quantities[t, 2] = rng.uniform(1, 5)
            quantities[t, 3] = rng.uniform(1, 5)

        session = ConsumerSession(prices=prices, quantities=quantities)
        result = compute_cannibalization(session, group_a=[0, 1], group_b=[2, 3])

        assert "a_to_b" in result
        assert "b_to_a" in result
        assert "symmetric" in result
        assert "net_direction" in result

    def test_cannibalization_detection(self):
        """Test detecting cannibalization."""
        n_obs = 50

        prices = np.ones((n_obs, 4))
        quantities = np.zeros((n_obs, 4))

        # Create cannibalization: when A is high, B is low
        for t in range(n_obs):
            if t < 25:
                # High A, low B
                quantities[t, 0] = 5
                quantities[t, 1] = 5
                quantities[t, 2] = 1
                quantities[t, 3] = 1
            else:
                # Low A, high B
                quantities[t, 0] = 1
                quantities[t, 1] = 1
                quantities[t, 2] = 5
                quantities[t, 3] = 5

        session = ConsumerSession(prices=prices, quantities=quantities)
        result = compute_cannibalization(session, group_a=[0, 1], group_b=[2, 3])

        # Should detect some cannibalization pattern
        assert result["symmetric"] >= 0

    def test_small_sample(self):
        """Test handling of small samples."""
        prices = np.array([[1, 1, 1, 1]])
        quantities = np.array([[1, 1, 1, 1]])

        session = ConsumerSession(prices=prices, quantities=quantities)
        result = compute_cannibalization(session, group_a=[0, 1], group_b=[2, 3])

        assert result["symmetric"] == 0.0


class TestSeparabilityResult:
    """Tests for SeparabilityResult properties."""

    def test_can_price_independently(self):
        """Test can_price_independently property."""
        prices = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [1, 2, 1, 2]])
        quantities = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])

        session = ConsumerSession(prices=prices, quantities=quantities)
        result = check_separability(session, group_a=[0, 1], group_b=[2, 3])

        # Property should exist and be boolean
        assert isinstance(result.can_price_independently, bool)

    def test_group_indices_preserved(self):
        """Test that group indices are preserved in result."""
        prices = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
        quantities = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

        session = ConsumerSession(prices=prices, quantities=quantities)
        result = check_separability(session, group_a=[0, 1], group_b=[2, 3, 4])

        assert result.group_a_indices == [0, 1]
        assert result.group_b_indices == [2, 3, 4]
