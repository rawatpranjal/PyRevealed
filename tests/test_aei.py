"""Tests for Afriat Efficiency Index computation."""

import numpy as np
import pytest

from pyrevealed import ConsumerSession, compute_aei


class TestAEIPerfectConsistency:
    """Tests for perfectly consistent data (AEI = 1.0)."""

    def test_consistent_session_aei_is_one(self, simple_consistent_session):
        """Test that consistent data has AEI = 1.0."""
        result = compute_aei(simple_consistent_session)

        assert result.efficiency_index == pytest.approx(1.0)
        assert result.is_perfectly_consistent is True
        assert result.binary_search_iterations == 0

    def test_single_observation_aei_is_one(self, single_observation_session):
        """Test that single observation has AEI = 1.0."""
        result = compute_aei(single_observation_session)

        assert result.efficiency_index == pytest.approx(1.0)
        assert result.is_perfectly_consistent is True


class TestAEIViolations:
    """Tests for data with GARP violations (AEI < 1.0)."""

    def test_violation_session_aei_less_than_one(self, simple_violation_session):
        """Test that violation data has AEI < 1.0."""
        result = compute_aei(simple_violation_session)

        assert result.efficiency_index < 1.0
        assert result.is_perfectly_consistent is False
        assert result.binary_search_iterations > 0

    def test_aei_is_bounded(self, simple_violation_session):
        """Test that AEI is in [0, 1]."""
        result = compute_aei(simple_violation_session)

        assert 0.0 <= result.efficiency_index <= 1.0


class TestAEIProperties:
    """Tests for AEI result properties."""

    def test_waste_fraction(self, simple_violation_session):
        """Test waste_fraction property."""
        result = compute_aei(simple_violation_session)

        expected_waste = 1.0 - result.efficiency_index
        assert result.waste_fraction == pytest.approx(expected_waste)

    def test_garp_result_at_threshold(self, simple_violation_session):
        """Test that GARP result at threshold is consistent."""
        result = compute_aei(simple_violation_session)

        # At the AEI threshold, GARP should be satisfied
        assert result.garp_result_at_threshold.is_consistent is True


class TestBinarySearchParameters:
    """Tests for binary search parameters."""

    def test_tolerance_affects_precision(self, simple_violation_session):
        """Test that smaller tolerance gives more precise result."""
        result_coarse = compute_aei(simple_violation_session, tolerance=1e-2)
        result_fine = compute_aei(simple_violation_session, tolerance=1e-8)

        # Fine result should use more iterations
        assert result_fine.binary_search_iterations >= result_coarse.binary_search_iterations

        # Results should be similar but fine should be more precise
        assert abs(result_fine.efficiency_index - result_coarse.efficiency_index) < 0.02

    def test_max_iterations_respected(self, simple_violation_session):
        """Test that max_iterations parameter is respected."""
        result = compute_aei(simple_violation_session, max_iterations=5)

        assert result.binary_search_iterations <= 5

    def test_computation_time_tracked(self, simple_consistent_session):
        """Test that computation time is tracked."""
        result = compute_aei(simple_consistent_session)

        assert result.computation_time_ms > 0


class TestAEISpecificValues:
    """Tests for specific AEI values in known scenarios."""

    def test_extreme_violation_low_aei(self):
        """Test that extreme violations give low AEI."""
        # Create extreme violation: choose expensive good while cheap option available
        # At each observation, the chosen bundle costs way more than the alternative
        prices = np.array([[1.0, 0.1], [0.1, 1.0]])
        quantities = np.array([[1.0, 0.0], [0.0, 1.0]])

        session = ConsumerSession(prices=prices, quantities=quantities)
        result = compute_aei(session)

        # AEI should be significantly below 1 (this is a clear violation)
        assert result.efficiency_index < 0.5

    def test_mild_violation_high_aei(self):
        """Test that mild violations give high AEI."""
        # Create mild violation: bundles are similar
        prices = np.array([[1.0, 1.0], [1.0, 1.0]])
        quantities = np.array([[5.0, 4.0], [4.0, 5.0]])

        session = ConsumerSession(prices=prices, quantities=quantities)
        result = compute_aei(session)

        # AEI should be close to 1 (mild violation)
        assert result.efficiency_index > 0.8
