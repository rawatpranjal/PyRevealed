"""Tests for Money Pump Index computation."""

import numpy as np
import pytest

from pyrevealed import ConsumerSession, compute_mpi


class TestMPIConsistent:
    """Tests for consistent data (MPI = 0)."""

    def test_consistent_session_mpi_is_zero(self, simple_consistent_session):
        """Test that consistent data has MPI = 0."""
        result = compute_mpi(simple_consistent_session)

        assert result.mpi_value == pytest.approx(0.0)
        assert result.is_consistent is True
        assert result.worst_cycle is None
        assert len(result.cycle_costs) == 0

    def test_single_observation_mpi_is_zero(self, single_observation_session):
        """Test that single observation has MPI = 0."""
        result = compute_mpi(single_observation_session)

        assert result.mpi_value == pytest.approx(0.0)
        assert result.is_consistent is True


class TestMPIViolations:
    """Tests for data with violations (MPI > 0)."""

    def test_violation_session_mpi_positive(self, simple_violation_session):
        """Test that violation data has MPI > 0."""
        result = compute_mpi(simple_violation_session)

        assert result.mpi_value > 0
        assert result.is_consistent == False  # noqa: E712
        assert result.worst_cycle is not None

    def test_mpi_is_bounded(self, simple_violation_session):
        """Test that MPI is in [0, 1]."""
        result = compute_mpi(simple_violation_session)

        assert 0.0 <= result.mpi_value <= 1.0


class TestMPIProperties:
    """Tests for MPI result properties."""

    def test_total_expenditure(self, simple_consistent_session):
        """Test total_expenditure property."""
        result = compute_mpi(simple_consistent_session)

        expected = simple_consistent_session.own_expenditures.sum()
        assert result.total_expenditure == pytest.approx(expected)

    def test_num_cycles(self, simple_violation_session):
        """Test num_cycles property."""
        result = compute_mpi(simple_violation_session)

        assert result.num_cycles >= 1


class TestMPIComputationTime:
    """Tests for computation time."""

    def test_computation_time_tracked(self, simple_consistent_session):
        """Test that computation time is tracked."""
        result = compute_mpi(simple_consistent_session)

        assert result.computation_time_ms > 0


class TestMPISpecificValues:
    """Tests for specific MPI scenarios."""

    def test_maximum_waste_scenario(self):
        """Test MPI in a maximum waste scenario."""
        # Each observation chooses the expensive good while the cheap one is available
        # This creates a strong violation with high MPI
        prices = np.array([[1.0, 0.1], [0.1, 1.0]])
        quantities = np.array([[1.0, 0.0], [0.0, 1.0]])

        session = ConsumerSession(prices=prices, quantities=quantities)
        result = compute_mpi(session)

        # Should have significant MPI (high waste due to irrational choices)
        assert result.mpi_value > 0

    def test_cycle_costs_structure(self, simple_violation_session):
        """Test that cycle_costs has correct structure."""
        result = compute_mpi(simple_violation_session)

        for cycle, cost in result.cycle_costs:
            # Each entry should be a (cycle, float) pair
            assert isinstance(cycle, tuple)
            assert isinstance(cost, float)
            assert cost >= 0
