"""
EVAL: Minimal data edge cases (T=1, N=1, empty data).

These tests expose how algorithms handle degenerate inputs.
"""

import numpy as np
import pytest
from pyrevealed.core.session import BehaviorLog, MenuChoiceLog, StochasticChoiceLog
from pyrevealed.core.exceptions import InsufficientDataError


class TestSingleObservation:
    """EVAL: T=1 single observation breaks pairwise algorithms."""

    def test_garp_single_observation(self, single_observation_log):
        """EVAL: GARP with T=1 - no pairs to compare."""
        from pyrevealed.algorithms.garp import check_garp
        # Should handle gracefully, not crash
        result = check_garp(single_observation_log)
        # With T=1, should be trivially consistent
        assert result.is_consistent, "T=1 should be trivially GARP-consistent"

    def test_aei_single_observation(self, single_observation_log):
        """EVAL: AEI with T=1 - binary search edge case."""
        from pyrevealed.algorithms.aei import compute_aei
        result = compute_aei(single_observation_log)
        assert result.efficiency_index == 1.0, "T=1 should have AEI=1.0"

    def test_mpi_single_observation(self, single_observation_log):
        """EVAL: MPI with T=1 - no cycles possible."""
        from pyrevealed.algorithms.mpi import compute_mpi
        result = compute_mpi(single_observation_log)
        assert result.mpi_value == 0.0, "T=1 should have MPI=0.0"

    def test_integrability_single_observation(self, single_observation_log):
        """EVAL: Slutsky test with T=1 - insufficient data for regression."""
        from pyrevealed.algorithms.integrability import test_integrability
        # This should either warn or handle gracefully
        result = test_integrability(single_observation_log)
        # Check it doesn't crash and returns something reasonable
        assert hasattr(result, 'is_integrable')


class TestSingleGood:
    """EVAL: N=1 single good creates degenerate matrices."""

    def test_garp_single_good(self, single_feature_log):
        """EVAL: GARP with N=1 - scalar expenditure matrix."""
        from pyrevealed.algorithms.garp import check_garp
        result = check_garp(single_feature_log)
        assert hasattr(result, 'is_consistent')

    def test_slutsky_single_good(self, single_feature_log):
        """EVAL: Slutsky matrix for N=1 - 1x1 matrix edge case."""
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix
        S = compute_slutsky_matrix(single_feature_log)
        assert S.shape == (1, 1), "N=1 should give 1x1 Slutsky matrix"
        # For single good, own-price effect should be finite
        assert np.isfinite(S[0, 0]), "Slutsky element should be finite"

    def test_welfare_single_good(self, single_feature_log):
        """EVAL: CV/EV with N=1 - degenerate optimization."""
        from pyrevealed.algorithms.welfare import compute_compensating_variation
        try:
            cv = compute_compensating_variation(
                single_feature_log, single_feature_log, method="bounds"
            )
            assert np.isfinite(cv), "CV should be finite"
        except Exception as e:
            pytest.fail(f"Single good CV should not crash: {e}")


class TestTwoObservations:
    """EVAL: T=2 minimum for comparison but edge case."""

    def test_garp_two_observations(self, two_observations_log):
        """EVAL: GARP with T=2 - minimum for pairwise comparison."""
        from pyrevealed.algorithms.garp import check_garp
        result = check_garp(two_observations_log)
        assert hasattr(result, 'is_consistent')
        assert hasattr(result, 'violations')

    def test_regression_two_observations(self, two_observations_log):
        """EVAL: Slutsky regression with T=2 - underdetermined system."""
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix
        # T=2 with N=2 is underdetermined for regression
        S = compute_slutsky_matrix(two_observations_log)
        assert S.shape == (2, 2)
        # Results may be unreliable but shouldn't crash


class TestEmptyAndDegenerate:
    """EVAL: Empty or degenerate data structures."""

    def test_create_empty_behavior_log_fails(self):
        """EVAL: Creating BehaviorLog with empty arrays should fail."""
        with pytest.raises((ValueError, InsufficientDataError)):
            BehaviorLog(
                cost_vectors=np.array([]).reshape(0, 2),
                action_vectors=np.array([]).reshape(0, 2),
            )

    def test_single_item_menus(self, empty_menu_choice_log):
        """EVAL: MenuChoiceLog with single-item menus - trivial choices."""
        from pyrevealed.algorithms.abstract_choice import check_sarp
        result = check_sarp(empty_menu_choice_log)
        # Single-item menus are trivially consistent
        assert result.is_consistent

    def test_stochastic_single_menu(self, single_menu_stochastic_log):
        """EVAL: StochasticChoiceLog with single menu - limited testing possible."""
        from pyrevealed.algorithms.stochastic import test_regularity
        result = test_regularity(single_menu_stochastic_log)
        # With single menu, no subset relationships to test
        assert result.num_testable_pairs == 0


class TestMinimalValidation:
    """EVAL: Validation behavior with minimal data."""

    def test_zero_prices_rejected(self):
        """EVAL: Zero prices should be rejected during validation."""
        with pytest.raises(Exception):  # ValueRangeError expected
            BehaviorLog(
                cost_vectors=np.array([[0.0, 1.0]]),
                action_vectors=np.array([[1.0, 1.0]]),
            )

    def test_negative_quantities_rejected(self):
        """EVAL: Negative quantities should be rejected."""
        with pytest.raises(Exception):  # ValueRangeError expected
            BehaviorLog(
                cost_vectors=np.array([[1.0, 1.0]]),
                action_vectors=np.array([[-1.0, 1.0]]),
            )

    def test_mismatched_dimensions_rejected(self):
        """EVAL: Mismatched array dimensions should be rejected."""
        with pytest.raises(Exception):  # DimensionError expected
            BehaviorLog(
                cost_vectors=np.array([[1.0, 2.0]]),
                action_vectors=np.array([[1.0, 2.0, 3.0]]),
            )


class TestBoundaryObservations:
    """EVAL: Boundary cases for observation counts."""

    def test_three_observations_minimum_for_cycles(self):
        """EVAL: Need T>=3 for potential cycles (GARP vs WARP distinction)."""
        from pyrevealed.algorithms.garp import check_garp, check_warp

        # With T=2, GARP and WARP should be equivalent
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]]),
        )
        garp = check_garp(log)
        warp = check_warp(log)

        # For T=2, GARP violation implies WARP violation
        if not garp.is_consistent:
            assert not warp.is_consistent, "T=2 GARP violation must be WARP violation"

    @pytest.mark.parametrize("T", [3, 4, 5])
    def test_small_T_garp(self, T):
        """EVAL: GARP with small T values."""
        from pyrevealed.algorithms.garp import check_garp
        np.random.seed(42)
        log = BehaviorLog(
            cost_vectors=np.random.rand(T, 3) + 0.1,
            action_vectors=np.random.rand(T, 3) + 0.1,
        )
        result = check_garp(log)
        assert hasattr(result, 'is_consistent')
        assert hasattr(result, 'computation_time_ms')
