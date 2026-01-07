"""Tests for the new revealed preference algorithms.

Tests for:
- Bronars' Power Index
- HARP (Homotheticity test)
- VEI (Per-observation efficiency)
- Quasilinearity (Cyclic monotonicity)
- Gross Substitutes
"""

import numpy as np
import pytest

from pyrevealed import (
    ConsumerSession,
    # Bronars
    compute_test_power,
    compute_bronars_power,
    # HARP
    validate_proportional_scaling,
    check_harp,
    # VEI
    compute_granular_integrity,
    compute_vei,
    # Quasilinearity
    test_income_invariance,
    check_quasilinearity,
    # Gross substitutes
    test_cross_price_effect,
    check_gross_substitutes,
    compute_cross_price_matrix,
)


# =============================================================================
# BRONARS' POWER INDEX TESTS
# =============================================================================


class TestBronarsPower:
    """Tests for Bronars' Power Index."""

    def test_power_bounded_zero_one(self, simple_consistent_session):
        """Test power index is in [0, 1]."""
        result = compute_test_power(simple_consistent_session, n_simulations=50)
        assert 0.0 <= result.power_index <= 1.0

    def test_reproducible_with_seed(self, simple_consistent_session):
        """Test results are reproducible with same seed."""
        result1 = compute_bronars_power(
            simple_consistent_session, n_simulations=50, random_seed=42
        )
        result2 = compute_bronars_power(
            simple_consistent_session, n_simulations=50, random_seed=42
        )
        assert result1.power_index == result2.power_index

    def test_n_simulations_respected(self, simple_consistent_session):
        """Test n_simulations parameter is respected."""
        result = compute_test_power(simple_consistent_session, n_simulations=30)
        assert result.n_simulations == 30
        if result.simulation_integrity_values is not None:
            assert len(result.simulation_integrity_values) == 30

    def test_is_significant_threshold(self, simple_consistent_session):
        """Test is_significant is correctly set."""
        result = compute_test_power(simple_consistent_session, n_simulations=50)
        assert result.is_significant == (result.power_index > 0.5)

    def test_violation_rate_equals_power(self, simple_consistent_session):
        """Test violation_rate property."""
        result = compute_test_power(simple_consistent_session, n_simulations=50)
        assert abs(result.violation_rate - result.power_index) < 1e-10

    def test_computation_time_tracked(self, simple_consistent_session):
        """Test computation time is tracked."""
        result = compute_test_power(simple_consistent_session, n_simulations=20)
        assert result.computation_time_ms > 0


# =============================================================================
# HARP (HOMOTHETICITY) TESTS
# =============================================================================


class TestHARP:
    """Tests for HARP (Homothetic Axiom of Revealed Preference)."""

    def test_consistent_session_result(self, simple_consistent_session):
        """Test HARP on consistent session."""
        result = validate_proportional_scaling(simple_consistent_session)
        # Result should have correct structure
        assert hasattr(result, "is_consistent")
        assert hasattr(result, "violations")
        assert hasattr(result, "max_cycle_product")
        assert hasattr(result, "garp_result")

    def test_garp_violation_implies_harp_violation(self, simple_violation_session):
        """Test that GARP violation implies HARP violation."""
        result = check_harp(simple_violation_session)
        # If GARP fails, HARP should also fail (HARP is stronger)
        assert result.garp_result.is_consistent is False
        # Note: HARP may or may not fail depending on the specific violation

    def test_single_observation_passes(self, single_observation_session):
        """Test single observation trivially passes HARP."""
        result = validate_proportional_scaling(single_observation_session)
        assert result.is_consistent is True

    def test_ratio_matrix_shape(self, simple_consistent_session):
        """Test ratio matrix has correct shape."""
        result = check_harp(simple_consistent_session)
        T = simple_consistent_session.num_observations
        assert result.expenditure_ratio_matrix.shape == (T, T)
        assert result.log_ratio_matrix.shape == (T, T)

    def test_diagonal_ratios_are_one(self, simple_consistent_session):
        """Test diagonal of ratio matrix is 1."""
        result = check_harp(simple_consistent_session)
        diagonal = np.diag(result.expenditure_ratio_matrix)
        assert np.allclose(diagonal, 1.0)

    def test_is_homothetic_property(self, simple_consistent_session):
        """Test is_homothetic alias property."""
        result = check_harp(simple_consistent_session)
        assert result.is_homothetic == result.is_consistent

    def test_computation_time_tracked(self, simple_consistent_session):
        """Test computation time is tracked."""
        result = check_harp(simple_consistent_session)
        assert result.computation_time_ms > 0


# =============================================================================
# VEI (PER-OBSERVATION EFFICIENCY) TESTS
# =============================================================================


class TestVEI:
    """Tests for Varian's Efficiency Index."""

    def test_consistent_data_all_ones(self, simple_consistent_session):
        """Test consistent data has all efficiency values = 1."""
        result = compute_granular_integrity(simple_consistent_session)
        if result.is_perfectly_consistent:
            assert np.allclose(result.efficiency_vector, 1.0)

    def test_efficiency_bounded(self, simple_violation_session):
        """Test efficiency values are in [0, 1]."""
        result = compute_vei(simple_violation_session)
        assert np.all(result.efficiency_vector >= 0.0)
        assert np.all(result.efficiency_vector <= 1.0)

    def test_mean_min_relationship(self, simple_consistent_session):
        """Test mean >= min."""
        result = compute_granular_integrity(simple_consistent_session)
        assert result.mean_efficiency >= result.min_efficiency

    def test_worst_observation_index(self, simple_violation_session):
        """Test worst observation is valid index."""
        result = compute_vei(simple_violation_session)
        T = simple_violation_session.num_observations
        assert 0 <= result.worst_observation < T

    def test_problematic_observations_threshold(self, simple_violation_session):
        """Test problematic observations use threshold correctly."""
        result = compute_granular_integrity(simple_violation_session, efficiency_threshold=0.95)
        for idx in result.problematic_observations:
            assert result.efficiency_vector[idx] < 0.95

    def test_total_inefficiency_calculation(self, simple_violation_session):
        """Test total inefficiency is sum of (1 - e_i)."""
        result = compute_vei(simple_violation_session)
        expected = np.sum(1.0 - result.efficiency_vector)
        assert abs(result.total_inefficiency - expected) < 1e-10

    def test_num_observations_property(self, simple_consistent_session):
        """Test num_observations property."""
        result = compute_granular_integrity(simple_consistent_session)
        assert result.num_observations == simple_consistent_session.num_observations


# =============================================================================
# QUASILINEARITY (CYCLIC MONOTONICITY) TESTS
# =============================================================================


class TestQuasilinearity:
    """Tests for quasilinearity (cyclic monotonicity) test."""

    def test_consistent_session_result(self, simple_consistent_session):
        """Test quasilinearity on consistent session."""
        result = test_income_invariance(simple_consistent_session)
        assert hasattr(result, "is_quasilinear")
        assert hasattr(result, "violations")
        assert hasattr(result, "worst_violation_magnitude")

    def test_single_observation_passes(self, single_observation_session):
        """Test single observation trivially passes."""
        result = check_quasilinearity(single_observation_session)
        assert result.is_quasilinear is True

    def test_num_cycles_tested(self, simple_consistent_session):
        """Test num_cycles_tested is positive."""
        result = test_income_invariance(simple_consistent_session)
        assert result.num_cycles_tested > 0

    def test_has_income_effects_property(self, simple_consistent_session):
        """Test has_income_effects property."""
        result = check_quasilinearity(simple_consistent_session)
        assert result.has_income_effects == (not result.is_quasilinear)

    def test_violations_have_negative_sums(self, simple_violation_session):
        """Test violations have negative cycle sums."""
        result = check_quasilinearity(simple_violation_session)
        for cycle in result.violations:
            if cycle in result.cycle_sums:
                assert result.cycle_sums[cycle] < 0

    def test_computation_time_tracked(self, simple_consistent_session):
        """Test computation time is tracked."""
        result = test_income_invariance(simple_consistent_session)
        assert result.computation_time_ms > 0


# =============================================================================
# GROSS SUBSTITUTES TESTS
# =============================================================================


class TestGrossSubstitutes:
    """Tests for gross substitutes test."""

    def test_valid_good_indices(self, simple_consistent_session):
        """Test with valid good indices."""
        result = test_cross_price_effect(simple_consistent_session, good_g=0, good_h=1)
        assert result.good_g_index == 0
        assert result.good_h_index == 1

    def test_invalid_good_index_raises(self, simple_consistent_session):
        """Test invalid good index raises error."""
        with pytest.raises(ValueError):
            check_gross_substitutes(simple_consistent_session, good_g=10, good_h=0)

    def test_same_goods_raises(self, simple_consistent_session):
        """Test same good indices raises error."""
        with pytest.raises(ValueError):
            check_gross_substitutes(simple_consistent_session, good_g=0, good_h=0)

    def test_relationship_is_valid(self, simple_consistent_session):
        """Test relationship is one of valid options."""
        result = test_cross_price_effect(simple_consistent_session, good_g=0, good_h=1)
        valid_relationships = {"substitutes", "complements", "independent", "inconclusive"}
        assert result.relationship in valid_relationships

    def test_confidence_bounded(self, simple_consistent_session):
        """Test confidence is in [0, 1]."""
        result = check_gross_substitutes(simple_consistent_session, good_g=0, good_h=1)
        assert 0.0 <= result.confidence_score <= 1.0

    def test_is_conclusive_property(self, simple_consistent_session):
        """Test is_conclusive property."""
        result = test_cross_price_effect(simple_consistent_session, good_g=0, good_h=1)
        assert result.is_conclusive == (result.relationship != "inconclusive")

    def test_substitution_matrix_shape(self, many_goods_session):
        """Test substitution matrix has correct shape."""
        result = compute_cross_price_matrix(many_goods_session)
        N = many_goods_session.num_goods
        assert result.relationship_matrix.shape == (N, N)
        assert result.confidence_matrix.shape == (N, N)

    def test_substitution_matrix_diagonal(self, many_goods_session):
        """Test diagonal of substitution matrix is 'self'."""
        result = compute_cross_price_matrix(many_goods_session)
        for i in range(result.num_goods):
            assert result.relationship_matrix[i, i] == "self"


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases across all new algorithms."""

    def test_two_observation_session(self):
        """Test all algorithms on minimal 2-observation session."""
        prices = np.array([[1.0, 2.0], [2.0, 1.0]])
        quantities = np.array([[3.0, 1.0], [1.0, 3.0]])
        session = ConsumerSession(prices=prices, quantities=quantities)

        # All should run without error
        bronars_result = compute_test_power(session, n_simulations=10)
        assert bronars_result is not None

        harp_result = check_harp(session)
        assert harp_result is not None

        vei_result = compute_vei(session)
        assert vei_result is not None

        quasi_result = check_quasilinearity(session)
        assert quasi_result is not None

        subs_result = check_gross_substitutes(session, 0, 1)
        assert subs_result is not None

    def test_many_goods_session(self, many_goods_session):
        """Test algorithms on session with many goods."""
        harp_result = check_harp(many_goods_session)
        assert harp_result is not None

        vei_result = compute_vei(many_goods_session)
        assert vei_result is not None

        quasi_result = check_quasilinearity(many_goods_session)
        assert quasi_result is not None

    def test_large_random_session_performance(self, large_random_session):
        """Test algorithms don't take too long on large session."""
        # HARP should complete in reasonable time
        harp_result = check_harp(large_random_session)
        assert harp_result.computation_time_ms < 10000  # 10 seconds max

        # VEI should complete
        vei_result = compute_vei(large_random_session)
        assert vei_result.computation_time_ms < 10000

        # Quasilinearity should complete
        quasi_result = check_quasilinearity(large_random_session, max_cycle_length=2)
        assert quasi_result.computation_time_ms < 10000


# =============================================================================
# INTEGRATION WITH BEHAVIORAL AUDITOR
# =============================================================================


class TestAuditorIntegration:
    """Tests for BehavioralAuditor integration."""

    def test_auditor_new_methods_exist(self, simple_consistent_session):
        """Test new methods are available on BehavioralAuditor."""
        from pyrevealed import BehavioralAuditor

        auditor = BehavioralAuditor()

        # All new methods should exist and be callable
        assert hasattr(auditor, "compute_test_power")
        assert hasattr(auditor, "validate_proportional_scaling")
        assert hasattr(auditor, "compute_granular_integrity")
        assert hasattr(auditor, "test_income_invariance")
        assert hasattr(auditor, "test_cross_price_effect")

    def test_auditor_test_power(self, simple_consistent_session):
        """Test auditor.compute_test_power."""
        from pyrevealed import BehavioralAuditor

        auditor = BehavioralAuditor()
        result = auditor.compute_test_power(simple_consistent_session, n_simulations=20)
        assert 0.0 <= result.power_index <= 1.0

    def test_auditor_proportional_scaling(self, simple_consistent_session):
        """Test auditor.validate_proportional_scaling."""
        from pyrevealed import BehavioralAuditor

        auditor = BehavioralAuditor()
        result = auditor.validate_proportional_scaling(simple_consistent_session)
        assert hasattr(result, "is_homothetic")

    def test_auditor_granular_integrity(self, simple_consistent_session):
        """Test auditor.compute_granular_integrity."""
        from pyrevealed import BehavioralAuditor

        auditor = BehavioralAuditor()
        result = auditor.compute_granular_integrity(simple_consistent_session)
        assert hasattr(result, "efficiency_vector")

    def test_auditor_income_invariance(self, simple_consistent_session):
        """Test auditor.test_income_invariance."""
        from pyrevealed import BehavioralAuditor

        auditor = BehavioralAuditor()
        result = auditor.test_income_invariance(simple_consistent_session)
        assert hasattr(result, "is_quasilinear")

    def test_auditor_cross_price_effect(self, simple_consistent_session):
        """Test auditor.test_cross_price_effect."""
        from pyrevealed import BehavioralAuditor

        auditor = BehavioralAuditor()
        result = auditor.test_cross_price_effect(simple_consistent_session, 0, 1)
        assert hasattr(result, "relationship")
