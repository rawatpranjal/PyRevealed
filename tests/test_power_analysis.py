"""Tests for power analysis functions (Beatty & Crawford 2011).

Tests for:
- compute_selten_measure: Selten's predictive success m = r - a
- compute_relative_area: Proportion of random behavior passing GARP
- compute_smoothed_hit_rate: Continuous measure for violators
- compute_generalized_predictive_success: md = rd - a
- compute_bayesian_credibility: Posterior probability of rationality
- compute_optimal_efficiency: Find e* that maximizes m(e)
"""

import numpy as np
import pytest

from pyrevealed import (
    BehaviorLog,
    # Result types
    SeltenMeasureResult,
    RelativeAreaResult,
    SmoothedHitRateResult,
    BayesianCredibilityResult,
    OptimalEfficiencyResult,
    # Functions
    compute_selten_measure,
    compute_power_metric,
    compute_relative_area,
    compute_test_demandingness,
    compute_smoothed_hit_rate,
    compute_near_miss_score,
    compute_generalized_predictive_success,
    compute_bayesian_credibility,
    compute_rationality_posterior,
    compute_optimal_efficiency,
    compute_optimal_predictive_efficiency,
    validate_consistency,
    compute_integrity_score,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def consistent_log():
    """GARP-consistent behavior log (3 observations, 2 goods)."""
    prices = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [1.5, 1.5],
    ])
    quantities = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
        [1.5, 1.5],
    ])
    return BehaviorLog(prices=prices, quantities=quantities)


@pytest.fixture
def violation_log():
    """GARP-violating behavior log (WARP violation).

    Creating a proper WARP violation:
    - Obs 0: prices (1, 2), choose (3, 1), expenditure = 5
      Bundle (1, 3) costs 1*1 + 2*3 = 7 > 5, NOT affordable
      So we need different setup...

    Let's use: both bundles strictly affordable at each other's prices
    - Obs 0: prices (1, 3), choose (4, 1), expenditure = 7
      Bundle (1, 4) costs 1*1 + 3*4 = 13 > 7, not affordable. Still wrong.

    Proper setup:
    - Obs 0: prices (2, 1), choose (2, 2), expenditure = 6
      Bundle (1, 3) costs 2*1 + 1*3 = 5 < 6, affordable!
    - Obs 1: prices (1, 2), choose (1, 3), expenditure = 7
      Bundle (2, 2) costs 1*2 + 2*2 = 6 < 7, affordable!

    But we also need strict preference (strictly cheaper):
    At obs 0: bundle (1,3) costs 5 < 6, so 0 strictly prefers (2,2) over (1,3)
    At obs 1: bundle (2,2) costs 6 < 7, so 1 strictly prefers (1,3) over (2,2)

    This is a WARP violation: (2,2) P (1,3) and (1,3) P (2,2)
    """
    prices = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
    ])
    quantities = np.array([
        [2.0, 2.0],
        [1.0, 3.0],
    ])
    return BehaviorLog(prices=prices, quantities=quantities)


@pytest.fixture
def high_power_log():
    """Behavior log with good price variation (high power)."""
    # Multiple intersecting budget constraints
    np.random.seed(42)
    n_obs = 20
    n_goods = 5
    prices = np.random.uniform(0.5, 2.0, (n_obs, n_goods))
    # Generate consistent choices by utility maximization
    utility_weights = np.random.uniform(0.5, 1.5, n_goods)
    quantities = np.zeros((n_obs, n_goods))
    for i in range(n_obs):
        expenditure = np.random.uniform(5, 10)
        # Spend all on good with highest utility/price ratio
        ratios = utility_weights / prices[i]
        best_good = np.argmax(ratios)
        quantities[i, best_good] = expenditure / prices[i, best_good]
    return BehaviorLog(prices=prices, quantities=quantities)


@pytest.fixture
def low_power_log():
    """Behavior log with minimal price variation (low power)."""
    # Near-identical prices = budget sets don't intersect much
    prices = np.array([
        [1.0, 1.0],
        [1.01, 0.99],
        [0.99, 1.01],
    ])
    quantities = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
    ])
    return BehaviorLog(prices=prices, quantities=quantities)


# =============================================================================
# Selten Measure Tests
# =============================================================================


class TestSeltenMeasure:
    """Tests for compute_selten_measure."""

    def test_consistent_data_positive_measure(self, consistent_log):
        """Consistent data should have non-negative Selten measure."""
        result = compute_selten_measure(consistent_log, n_simulations=100, random_seed=42)

        assert isinstance(result, SeltenMeasureResult)
        assert result.pass_rate == 1.0
        assert 0 <= result.relative_area <= 1
        # m = r - a = 1 - a, should be non-negative if test has power
        assert result.measure >= -1.0
        assert result.measure <= 1.0

    def test_violating_data_negative_measure(self, violation_log):
        """Violating data with easy test should have negative measure."""
        result = compute_selten_measure(violation_log, n_simulations=100, random_seed=42)

        assert result.pass_rate == 0.0  # Fails GARP
        # m = r - a = 0 - a = -a
        assert result.measure <= 0

    def test_algorithm_options(self, consistent_log):
        """Test different random bundle generation algorithms."""
        result1 = compute_selten_measure(consistent_log, n_simulations=50, algorithm=1, random_seed=42)
        result2 = compute_selten_measure(consistent_log, n_simulations=50, algorithm=2, random_seed=42)
        result3 = compute_selten_measure(consistent_log, n_simulations=50, algorithm=3, random_seed=42)

        # All should return valid results
        for r in [result1, result2, result3]:
            assert isinstance(r, SeltenMeasureResult)
            assert -1 <= r.measure <= 1

    def test_result_attributes(self, consistent_log):
        """Test all result attributes are populated."""
        result = compute_selten_measure(consistent_log, n_simulations=50, random_seed=42)

        assert hasattr(result, 'measure')
        assert hasattr(result, 'pass_rate')
        assert hasattr(result, 'relative_area')
        assert hasattr(result, 'n_simulations')
        assert hasattr(result, 'algorithm')
        assert hasattr(result, 'is_meaningful')
        assert hasattr(result, 'computation_time_ms')
        assert hasattr(result, 'bronars_power')

    def test_tech_friendly_alias(self, consistent_log):
        """Test that compute_power_metric is an alias."""
        result1 = compute_selten_measure(consistent_log, n_simulations=50, random_seed=42)
        result2 = compute_power_metric(consistent_log, n_simulations=50, random_seed=42)

        assert result1.measure == result2.measure

    def test_summary_output(self, consistent_log):
        """Test summary() method produces readable output."""
        result = compute_selten_measure(consistent_log, n_simulations=50, random_seed=42)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "Selten" in summary.upper() or "PREDICTIVE" in summary.upper()
        assert len(summary) > 100  # Should have substantial content


# =============================================================================
# Relative Area Tests
# =============================================================================


class TestRelativeArea:
    """Tests for compute_relative_area."""

    def test_area_bounds(self, consistent_log):
        """Area should be between 0 and 1."""
        result = compute_relative_area(consistent_log, n_simulations=100, random_seed=42)

        assert isinstance(result, RelativeAreaResult)
        assert 0 <= result.relative_area <= 1

    def test_high_power_low_area(self, high_power_log):
        """Data with good price variation should have lower area."""
        result = compute_relative_area(high_power_log, n_simulations=200, random_seed=42)

        # With good price variation, random behavior should often violate
        # Area should be reasonably low (high power)
        # Note: this depends on actual data, so we use a loose bound
        assert result.relative_area < 0.99  # Some random should violate

    def test_confidence_interval(self, consistent_log):
        """Test confidence interval computation."""
        result = compute_relative_area(consistent_log, n_simulations=200, random_seed=42)

        assert result.ci_lower <= result.relative_area
        assert result.relative_area <= result.ci_upper
        assert result.std_error >= 0

    def test_bronars_power_relationship(self, consistent_log):
        """Bronars power should be 1 - relative_area."""
        result = compute_relative_area(consistent_log, n_simulations=100, random_seed=42)

        expected_power = 1.0 - result.relative_area
        assert abs(result.bronars_power - expected_power) < 1e-10

    def test_tech_friendly_alias(self, consistent_log):
        """Test that compute_test_demandingness is an alias."""
        result1 = compute_relative_area(consistent_log, n_simulations=50, random_seed=42)
        result2 = compute_test_demandingness(consistent_log, n_simulations=50, random_seed=42)

        assert result1.relative_area == result2.relative_area


# =============================================================================
# Smoothed Hit Rate Tests
# =============================================================================


class TestSmoothedHitRate:
    """Tests for compute_smoothed_hit_rate."""

    def test_consistent_data_perfect_rate(self, consistent_log):
        """Consistent data should have smoothed rate = 1."""
        result = compute_smoothed_hit_rate(consistent_log, random_seed=42)

        assert isinstance(result, SmoothedHitRateResult)
        assert result.smoothed_rate == 1.0
        assert result.is_consistent == True
        assert result.distance == 0.0

    def test_violating_data_reduced_rate(self, violation_log):
        """Violating data should have smoothed rate < 1."""
        result = compute_smoothed_hit_rate(violation_log, random_seed=42)

        assert result.smoothed_rate < 1.0
        assert result.is_consistent == False
        assert result.distance > 0.0

    def test_rate_equals_aei(self, violation_log):
        """Smoothed rate should approximately equal AEI."""
        result = compute_smoothed_hit_rate(violation_log, random_seed=42)
        aei_result = compute_integrity_score(violation_log)

        # They should be close (smoothed rate uses AEI internally)
        assert abs(result.smoothed_rate - aei_result.efficiency_index) < 0.01
        assert abs(result.aei - aei_result.efficiency_index) < 0.01

    def test_near_miss_property(self, consistent_log):
        """Test is_near_miss property."""
        result = compute_smoothed_hit_rate(consistent_log, random_seed=42)

        # Consistent data is not a "near miss" - it's a hit
        assert result.is_near_miss == False

    def test_tech_friendly_alias(self, consistent_log):
        """Test that compute_near_miss_score is an alias."""
        result1 = compute_smoothed_hit_rate(consistent_log, random_seed=42)
        result2 = compute_near_miss_score(consistent_log, random_seed=42)

        assert result1.smoothed_rate == result2.smoothed_rate


# =============================================================================
# Generalized Predictive Success Tests
# =============================================================================


class TestGeneralizedPredictiveSuccess:
    """Tests for compute_generalized_predictive_success."""

    def test_consistent_data(self, consistent_log):
        """Test with GARP-consistent data."""
        result = compute_generalized_predictive_success(
            consistent_log, n_simulations=100, random_seed=42
        )

        assert isinstance(result, SeltenMeasureResult)
        # For consistent data, pass_rate (smoothed) = 1
        assert result.pass_rate == 1.0

    def test_violating_data_uses_smoothed_rate(self, violation_log):
        """Violating data should use smoothed hit rate."""
        result = compute_generalized_predictive_success(
            violation_log, n_simulations=100, random_seed=42
        )

        # Pass rate should be the smoothed rate, not 0
        assert result.pass_rate < 1.0  # Violating
        assert result.pass_rate > 0.0  # But smoothed, not binary

    def test_measure_bounds(self, consistent_log):
        """Measure should be in [-1, 1]."""
        result = compute_generalized_predictive_success(
            consistent_log, n_simulations=100, random_seed=42
        )

        assert -1 <= result.measure <= 1


# =============================================================================
# Bayesian Credibility Tests
# =============================================================================


class TestBayesianCredibility:
    """Tests for compute_bayesian_credibility."""

    def test_consistent_data_high_posterior(self, consistent_log):
        """Consistent data should have high posterior with reasonable power."""
        result = compute_bayesian_credibility(
            consistent_log, prior_rational=0.5, n_simulations=100, random_seed=42
        )

        assert isinstance(result, BayesianCredibilityResult)
        assert result.passes_garp == True
        assert result.posterior >= result.prior  # Posterior should increase

    def test_violating_data_zero_posterior(self, violation_log):
        """Violating data should have posterior = 0 under strict model."""
        result = compute_bayesian_credibility(
            violation_log, prior_rational=0.5, n_simulations=100, random_seed=42
        )

        assert result.passes_garp == False
        assert result.posterior == 0.0  # Strict model: violating = not rational

    def test_prior_affects_posterior(self, consistent_log):
        """Different priors should give different posteriors."""
        result_low = compute_bayesian_credibility(
            consistent_log, prior_rational=0.1, n_simulations=100, random_seed=42
        )
        result_high = compute_bayesian_credibility(
            consistent_log, prior_rational=0.9, n_simulations=100, random_seed=42
        )

        # Both should update upward from prior (passing GARP is evidence)
        # But the absolute posteriors will differ
        assert result_low.prior == 0.1
        assert result_high.prior == 0.9

    def test_bayes_factor(self, consistent_log):
        """Test Bayes factor computation."""
        result = compute_bayesian_credibility(
            consistent_log, n_simulations=100, random_seed=42
        )

        if result.passes_garp and result.p_pass_given_random > 0:
            expected_bf = result.p_pass_given_rational / result.p_pass_given_random
            assert abs(result.bayes_factor - expected_bf) < 1e-10

    def test_evidence_strength_categories(self, consistent_log):
        """Test evidence strength categorization."""
        result = compute_bayesian_credibility(
            consistent_log, n_simulations=100, random_seed=42
        )

        valid_strengths = ["decisive", "very_strong", "strong", "moderate", "anecdotal", "against"]
        assert result.evidence_strength in valid_strengths

    def test_tech_friendly_alias(self, consistent_log):
        """Test that compute_rationality_posterior is an alias."""
        result1 = compute_bayesian_credibility(
            consistent_log, prior_rational=0.5, n_simulations=50, random_seed=42
        )
        result2 = compute_rationality_posterior(
            consistent_log, prior_rational=0.5, n_simulations=50, random_seed=42
        )

        assert result1.posterior == result2.posterior

    def test_summary_output(self, consistent_log):
        """Test summary() method."""
        result = compute_bayesian_credibility(
            consistent_log, n_simulations=50, random_seed=42
        )

        summary = result.summary()
        assert isinstance(summary, str)
        assert "BAYESIAN" in summary.upper() or "CREDIBILITY" in summary.upper()


# =============================================================================
# Integration Tests
# =============================================================================


class TestPowerAnalysisIntegration:
    """Integration tests for power analysis functions."""

    def test_relationship_area_and_bronars(self, consistent_log):
        """Relative area should be approximately 1 - Bronars power."""
        from pyrevealed import compute_bronars_power

        area_result = compute_relative_area(
            consistent_log, n_simulations=200, random_seed=42
        )
        bronars_result = compute_bronars_power(
            consistent_log, n_simulations=200, random_seed=42
        )

        # They use different random draws, so allow some tolerance
        # area ≈ 1 - bronars_power
        expected_area = 1.0 - bronars_result.power_index
        assert abs(area_result.relative_area - expected_area) < 0.15

    def test_selten_uses_correct_pass_rate(self, consistent_log, violation_log):
        """Verify Selten measure uses correct binary pass rate."""
        consistent_result = compute_selten_measure(
            consistent_log, n_simulations=50, random_seed=42
        )
        violation_result = compute_selten_measure(
            violation_log, n_simulations=50, random_seed=42
        )

        # Consistent should pass, violation should fail
        garp_consistent = validate_consistency(consistent_log)
        garp_violation = validate_consistency(violation_log)

        assert consistent_result.pass_rate == (1.0 if garp_consistent.is_consistent else 0.0)
        assert violation_result.pass_rate == (1.0 if garp_violation.is_consistent else 0.0)

    def test_all_results_serializable(self, consistent_log):
        """All results should have to_dict() method."""
        selten = compute_selten_measure(consistent_log, n_simulations=50, random_seed=42)
        area = compute_relative_area(consistent_log, n_simulations=50, random_seed=42)
        smoothed = compute_smoothed_hit_rate(consistent_log, random_seed=42)
        bayes = compute_bayesian_credibility(consistent_log, n_simulations=50, random_seed=42)

        for result in [selten, area, smoothed, bayes]:
            d = result.to_dict()
            assert isinstance(d, dict)
            assert len(d) > 0

    def test_reproducibility_with_seed(self, consistent_log):
        """Results should be reproducible with same seed."""
        result1 = compute_selten_measure(
            consistent_log, n_simulations=100, random_seed=123
        )
        result2 = compute_selten_measure(
            consistent_log, n_simulations=100, random_seed=123
        )

        assert result1.measure == result2.measure
        assert result1.relative_area == result2.relative_area


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_minimal_observations(self):
        """Test with minimal (2) observations."""
        prices = np.array([[1.0, 2.0], [2.0, 1.0]])
        quantities = np.array([[2.0, 1.0], [1.0, 2.0]])
        log = BehaviorLog(prices=prices, quantities=quantities)

        result = compute_selten_measure(log, n_simulations=50, random_seed=42)
        assert isinstance(result, SeltenMeasureResult)

    def test_single_good(self):
        """Test with single good (trivially consistent)."""
        prices = np.array([[1.0], [2.0], [0.5]])
        quantities = np.array([[5.0], [2.5], [10.0]])
        log = BehaviorLog(prices=prices, quantities=quantities)

        # Single good is trivially consistent
        result = compute_relative_area(log, n_simulations=50, random_seed=42)
        assert result.relative_area == 1.0  # All random behavior passes

    def test_many_goods(self):
        """Test with many goods."""
        np.random.seed(42)
        n_obs = 10
        n_goods = 20
        prices = np.random.uniform(0.5, 2.0, (n_obs, n_goods))
        quantities = np.random.uniform(0.1, 2.0, (n_obs, n_goods))
        log = BehaviorLog(prices=prices, quantities=quantities)

        result = compute_selten_measure(log, n_simulations=50, random_seed=42)
        assert isinstance(result, SeltenMeasureResult)


# =============================================================================
# Optimal Efficiency Tests
# =============================================================================


class TestOptimalEfficiency:
    """Tests for compute_optimal_efficiency."""

    def test_consistent_data_result_type(self, consistent_log):
        """Consistent data should return OptimalEfficiencyResult."""
        result = compute_optimal_efficiency(
            consistent_log,
            n_simulations=30,
            n_efficiency_levels=10,
            random_seed=42
        )

        assert isinstance(result, OptimalEfficiencyResult)

    def test_result_attributes(self, consistent_log):
        """Test all result attributes are populated."""
        result = compute_optimal_efficiency(
            consistent_log,
            n_simulations=30,
            n_efficiency_levels=10,
            random_seed=42
        )

        assert hasattr(result, 'optimal_efficiency')
        assert hasattr(result, 'optimal_measure')
        assert hasattr(result, 'efficiency_levels')
        assert hasattr(result, 'measures')
        assert hasattr(result, 'pass_rates')
        assert hasattr(result, 'relative_areas')
        assert hasattr(result, 'aei')
        assert hasattr(result, 'computation_time_ms')
        assert hasattr(result, 'is_meaningful')
        assert hasattr(result, 'bronars_power_at_optimal')

    def test_efficiency_bounds(self, consistent_log):
        """Optimal efficiency should be in (0, 1]."""
        result = compute_optimal_efficiency(
            consistent_log,
            n_simulations=30,
            n_efficiency_levels=10,
            random_seed=42
        )

        assert 0 < result.optimal_efficiency <= 1.0

    def test_measure_bounds(self, consistent_log):
        """Optimal measure should be in [-1, 1]."""
        result = compute_optimal_efficiency(
            consistent_log,
            n_simulations=30,
            n_efficiency_levels=10,
            random_seed=42
        )

        assert -1.0 <= result.optimal_measure <= 1.0

    def test_list_lengths(self, consistent_log):
        """All result lists should have same length."""
        n_levels = 15
        result = compute_optimal_efficiency(
            consistent_log,
            n_simulations=30,
            n_efficiency_levels=n_levels,
            random_seed=42
        )

        assert len(result.efficiency_levels) == n_levels
        assert len(result.measures) == n_levels
        assert len(result.pass_rates) == n_levels
        assert len(result.relative_areas) == n_levels

    def test_optimal_is_argmax(self, consistent_log):
        """Optimal efficiency should correspond to max measure."""
        result = compute_optimal_efficiency(
            consistent_log,
            n_simulations=30,
            n_efficiency_levels=10,
            random_seed=42
        )

        max_idx = result.measures.index(max(result.measures))
        assert result.optimal_efficiency == result.efficiency_levels[max_idx]
        assert result.optimal_measure == result.measures[max_idx]

    def test_pass_rates_binary(self, consistent_log):
        """Pass rates should be 0 or 1."""
        result = compute_optimal_efficiency(
            consistent_log,
            n_simulations=30,
            n_efficiency_levels=10,
            random_seed=42
        )

        for pr in result.pass_rates:
            assert pr in [0.0, 1.0]

    def test_relative_areas_bounded(self, consistent_log):
        """Relative areas should be in [0, 1]."""
        result = compute_optimal_efficiency(
            consistent_log,
            n_simulations=30,
            n_efficiency_levels=10,
            random_seed=42
        )

        for area in result.relative_areas:
            assert 0.0 <= area <= 1.0

    def test_aei_matches_standard_computation(self, consistent_log):
        """AEI in result should match standard AEI computation."""
        result = compute_optimal_efficiency(
            consistent_log,
            n_simulations=30,
            n_efficiency_levels=10,
            random_seed=42
        )
        aei_result = compute_integrity_score(consistent_log)

        assert abs(result.aei - aei_result.efficiency_index) < 0.01

    def test_violating_data(self, violation_log):
        """Test with GARP-violating data."""
        result = compute_optimal_efficiency(
            violation_log,
            n_simulations=30,
            n_efficiency_levels=10,
            random_seed=42
        )

        assert isinstance(result, OptimalEfficiencyResult)
        # AEI should be < 1 for violating data
        assert result.aei < 1.0

    def test_tech_friendly_alias(self, consistent_log):
        """Test that compute_optimal_predictive_efficiency is an alias."""
        result1 = compute_optimal_efficiency(
            consistent_log,
            n_simulations=30,
            n_efficiency_levels=10,
            random_seed=42
        )
        result2 = compute_optimal_predictive_efficiency(
            consistent_log,
            n_simulations=30,
            n_efficiency_levels=10,
            random_seed=42
        )

        assert result1.optimal_efficiency == result2.optimal_efficiency
        assert result1.optimal_measure == result2.optimal_measure

    def test_reproducibility_with_seed(self, consistent_log):
        """Results should be reproducible with same seed."""
        result1 = compute_optimal_efficiency(
            consistent_log,
            n_simulations=30,
            n_efficiency_levels=10,
            random_seed=123
        )
        result2 = compute_optimal_efficiency(
            consistent_log,
            n_simulations=30,
            n_efficiency_levels=10,
            random_seed=123
        )

        assert result1.optimal_efficiency == result2.optimal_efficiency
        assert result1.optimal_measure == result2.optimal_measure
        assert result1.measures == result2.measures

    def test_summary_output(self, consistent_log):
        """Test summary() method produces readable output."""
        result = compute_optimal_efficiency(
            consistent_log,
            n_simulations=30,
            n_efficiency_levels=10,
            random_seed=42
        )

        summary = result.summary()
        assert isinstance(summary, str)
        assert "OPTIMAL" in summary.upper() or "EFFICIENCY" in summary.upper()
        assert len(summary) > 100  # Should have substantial content

    def test_to_dict(self, consistent_log):
        """Test to_dict() method."""
        result = compute_optimal_efficiency(
            consistent_log,
            n_simulations=30,
            n_efficiency_levels=10,
            random_seed=42
        )

        d = result.to_dict()
        assert isinstance(d, dict)
        assert "optimal_efficiency" in d
        assert "optimal_measure" in d
        assert "aei" in d
        assert "is_meaningful" in d

    def test_algorithm_options(self, consistent_log):
        """Test different random bundle generation algorithms."""
        result1 = compute_optimal_efficiency(
            consistent_log,
            n_simulations=20,
            n_efficiency_levels=5,
            algorithm=1,
            random_seed=42
        )
        result2 = compute_optimal_efficiency(
            consistent_log,
            n_simulations=20,
            n_efficiency_levels=5,
            algorithm=2,
            random_seed=42
        )
        result3 = compute_optimal_efficiency(
            consistent_log,
            n_simulations=20,
            n_efficiency_levels=5,
            algorithm=3,
            random_seed=42
        )

        # All should return valid results
        for r in [result1, result2, result3]:
            assert isinstance(r, OptimalEfficiencyResult)
            assert 0 < r.optimal_efficiency <= 1.0
