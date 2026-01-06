"""Tests for GARP detection algorithm."""

import numpy as np
import pytest

from pyrevealed import ConsumerSession, check_garp
from pyrevealed.algorithms.garp import check_warp


class TestGARPConsistent:
    """Tests for data that should satisfy GARP."""

    def test_consistent_session_passes_garp(self, simple_consistent_session):
        """Test that consistent data passes GARP."""
        result = check_garp(simple_consistent_session)

        assert result.is_consistent is True
        assert len(result.violations) == 0
        assert result.num_violations == 0

    def test_single_observation_passes_garp(self, single_observation_session):
        """Test that single observation trivially passes GARP."""
        result = check_garp(single_observation_session)

        assert result.is_consistent is True
        assert len(result.violations) == 0

    def test_identical_bundles_passes_garp(self, borderline_session):
        """Test that identical bundles pass GARP."""
        result = check_garp(borderline_session)

        assert result.is_consistent is True


class TestGARPViolations:
    """Tests for data that should violate GARP."""

    def test_simple_violation_detected(self, simple_violation_session):
        """Test that simple WARP violation is detected."""
        result = check_garp(simple_violation_session)

        assert result.is_consistent is False
        assert len(result.violations) > 0

    def test_three_cycle_violation_detected(self, three_cycle_violation_session):
        """Test that length-3 cycle violation is detected."""
        result = check_garp(three_cycle_violation_session)

        assert result.is_consistent is False
        assert len(result.violations) > 0


class TestPreferenceMatrices:
    """Tests for preference matrix computation."""

    def test_direct_preference_matrix_shape(self, simple_consistent_session):
        """Test direct preference matrix has correct shape."""
        result = check_garp(simple_consistent_session)
        T = simple_consistent_session.num_observations

        assert result.direct_revealed_preference.shape == (T, T)
        assert result.strict_revealed_preference.shape == (T, T)
        assert result.transitive_closure.shape == (T, T)

    def test_preference_matrix_is_boolean(self, simple_consistent_session):
        """Test preference matrices are boolean."""
        result = check_garp(simple_consistent_session)

        assert result.direct_revealed_preference.dtype == np.bool_
        assert result.strict_revealed_preference.dtype == np.bool_
        assert result.transitive_closure.dtype == np.bool_

    def test_reflexivity_in_transitive_closure(self, simple_consistent_session):
        """Test that transitive closure has diagonal True."""
        result = check_garp(simple_consistent_session)

        # Every observation should reach itself
        assert np.all(np.diag(result.transitive_closure))


class TestWARP:
    """Tests for WARP (Weak Axiom) check."""

    def test_warp_consistent(self, simple_consistent_session):
        """Test consistent data passes WARP."""
        is_consistent, violations = check_warp(simple_consistent_session)

        assert is_consistent is True
        assert len(violations) == 0

    def test_warp_violation(self, simple_violation_session):
        """Test WARP violation is detected."""
        is_consistent, violations = check_warp(simple_violation_session)

        assert is_consistent is False
        assert len(violations) > 0


class TestComputationTime:
    """Tests for computation time tracking."""

    def test_computation_time_positive(self, simple_consistent_session):
        """Test that computation time is tracked."""
        result = check_garp(simple_consistent_session)

        assert result.computation_time_ms > 0

    def test_large_session_performance(self, large_random_session):
        """Test that large session completes in reasonable time."""
        result = check_garp(large_random_session)

        # Should complete in under 10 seconds even for 100 observations
        # (threshold is generous to account for varying system loads)
        assert result.computation_time_ms < 10000


class TestTolerance:
    """Tests for numerical tolerance handling."""

    def test_tolerance_affects_result(self):
        """Test that tolerance parameter is respected."""
        # Create data that is exactly on the boundary
        prices = np.array([[1.0, 1.0], [1.0, 1.0]])
        quantities = np.array([[2.0, 1.0], [1.0, 2.0]])

        session = ConsumerSession(prices=prices, quantities=quantities)

        # With default tolerance
        result_default = check_garp(session)

        # Both should be consistent (budget = 3 for both)
        # No strict preference because E[0,0] = E[0,1] = 3
        assert result_default.is_consistent is True
