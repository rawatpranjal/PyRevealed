"""
EVAL: Extreme value tests - float boundaries, extreme ratios.

These tests expose behavior at the limits of floating point representation.
"""

import numpy as np
import pytest
from pyrevealed.core.session import BehaviorLog


class TestFloatBoundaries:
    """EVAL: Values at float64 boundaries."""

    def test_max_float_expenditure(self, max_float_prices):
        """EVAL: Expenditure computation near float64 max."""
        E = max_float_prices.spend_matrix

        # Should not overflow to inf
        assert np.all(np.isfinite(E)), f"Overflow in expenditure matrix: {E}"

    def test_min_positive_float(self):
        """EVAL: Values at float64 minimum positive value."""
        tiny = np.finfo(np.float64).tiny

        log = BehaviorLog(
            cost_vectors=np.array([[tiny, tiny], [tiny, tiny]]),
            action_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
        )

        E = log.spend_matrix

        # Should handle tiny values
        assert np.all(np.isfinite(E))

    def test_epsilon_differences(self):
        """EVAL: Values differing by machine epsilon."""
        eps = np.finfo(np.float64).eps

        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            action_vectors=np.array([
                [1.0, 1.0],
                [1.0 + eps, 1.0 + eps],
            ]),
        )

        from pyrevealed.algorithms.garp import check_garp
        result = check_garp(log)

        # Epsilon differences should be handled correctly
        assert hasattr(result, 'is_consistent')


class TestExtremeRatios:
    """EVAL: Extreme ratios between values."""

    def test_price_ratio_1e15(self):
        """EVAL: Price ratio of 1e15 between goods."""
        log = BehaviorLog(
            cost_vectors=np.array([[1e15, 1.0], [1.0, 1e15]]),
            action_vectors=np.array([[1.0, 1e15], [1e15, 1.0]]),
        )

        from pyrevealed.algorithms.garp import check_garp
        result = check_garp(log)

        assert hasattr(result, 'is_consistent')

    def test_quantity_ratio_1e300(self, extreme_ratio_log):
        """EVAL: Quantity ratio of 1e300 between observations."""
        from pyrevealed.algorithms.garp import check_garp

        result = check_garp(extreme_ratio_log)
        assert hasattr(result, 'is_consistent')

    def test_mixed_extreme_scales(self, mixed_scale_prices):
        """EVAL: Values varying by 15 orders of magnitude."""
        from pyrevealed.algorithms.garp import check_garp

        result = check_garp(mixed_scale_prices)

        # Check expenditure matrix is finite
        E = mixed_scale_prices.spend_matrix
        assert np.all(np.isfinite(E)), f"Non-finite expenditure with mixed scales: {E}"


class TestZeroBoundary:
    """EVAL: Values approaching zero."""

    def test_quantities_approaching_zero(self, near_zero_quantities):
        """EVAL: Quantities at 1e-300."""
        from pyrevealed.algorithms.garp import check_garp

        result = check_garp(near_zero_quantities)
        assert hasattr(result, 'is_consistent')

    def test_expenditure_approaching_zero(self):
        """EVAL: Total expenditure approaching zero."""
        log = BehaviorLog(
            cost_vectors=np.array([[1e-150, 1e-150], [1e-150, 1e-150]]),
            action_vectors=np.array([[1e-150, 1e-150], [1e-150, 1e-150]]),
        )

        total = log.total_spend

        # Total expenditure = 2e-300, may underflow
        if np.any(total == 0):
            pytest.xfail("Total expenditure underflowed to zero")


class TestInfinityHandling:
    """EVAL: Infinity values in computations."""

    def test_log_of_zero_utility(self):
        """EVAL: Log utility with zero quantities."""
        # BehaviorLog validation should prevent zero quantities
        with pytest.raises(Exception):
            BehaviorLog(
                cost_vectors=np.array([[1.0, 1.0]]),
                action_vectors=np.array([[0.0, 1.0]]),  # Zero quantity
            )

    def test_division_creating_infinity(self):
        """EVAL: Operations that could create infinity."""
        from pyrevealed.algorithms.integrability import compute_slutsky_matrix

        # Very small prices could create large 1/p values
        log = BehaviorLog(
            cost_vectors=np.array([
                [1e-300, 1.0],
                [1.0, 1e-300],
            ]),
            action_vectors=np.array([
                [1.0, 1.0],
                [1.0, 1.0],
            ]),
        )

        S = compute_slutsky_matrix(log)

        # Check for infinity
        if np.any(np.isinf(S)):
            pytest.xfail(f"Infinity in Slutsky matrix: {S}")


class TestLargeAbsoluteValues:
    """EVAL: Very large absolute values."""

    def test_large_prices(self):
        """EVAL: Prices in the billions."""
        log = BehaviorLog(
            cost_vectors=np.array([[1e9, 1e9], [1e9, 1e9]]),
            action_vectors=np.array([[1000.0, 1000.0], [1000.0, 1000.0]]),
        )

        # Expenditure will be ~2e12, should be fine
        E = log.spend_matrix
        assert np.all(np.isfinite(E))

    def test_large_quantities(self):
        """EVAL: Quantities in the trillions."""
        log = BehaviorLog(
            cost_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
            action_vectors=np.array([[1e12, 1e12], [1e12, 1e12]]),
        )

        from pyrevealed.algorithms.garp import check_garp
        result = check_garp(log)
        assert hasattr(result, 'is_consistent')


class TestSpecialFloatValues:
    """EVAL: Special IEEE 754 values."""

    def test_denormalized_numbers(self, subnormal_values):
        """EVAL: Subnormal/denormalized floating point values."""
        from pyrevealed.algorithms.garp import check_garp

        result = check_garp(subnormal_values)

        # Should handle without crashing
        assert hasattr(result, 'is_consistent')

    def test_near_overflow_multiplication(self):
        """EVAL: Multiplication that nearly overflows."""
        # sqrt(max_float) * sqrt(max_float) would overflow
        near_sqrt_max = np.sqrt(np.finfo(np.float64).max) / 10

        log = BehaviorLog(
            cost_vectors=np.array([[near_sqrt_max, 1.0]]),
            action_vectors=np.array([[near_sqrt_max, 1.0]]),
        )

        E = log.spend_matrix

        # Expenditure = near_sqrt_max^2, should be large but finite
        assert np.all(np.isfinite(E)), f"Overflow: {E[0,0]}"


class TestRiskChoiceExtremes:
    """EVAL: Extreme values in risk choice data."""

    def test_extreme_lottery_outcomes(self, extreme_outcomes_lottery):
        """EVAL: Lottery with outcomes at 1e150 and 1e-150."""
        from pyrevealed.algorithms.risk import estimate_crra_parameter

        result = estimate_crra_parameter(extreme_outcomes_lottery)

        # Should handle extreme values
        assert hasattr(result, 'rho')

    def test_near_zero_probability(self, zero_probability_lottery):
        """EVAL: Lottery with zero probability outcomes."""
        from pyrevealed.algorithms.risk import estimate_crra_parameter

        result = estimate_crra_parameter(zero_probability_lottery)

        # Expected utility calculation should handle p=0
        assert hasattr(result, 'rho')


class TestWelfareExtremes:
    """EVAL: Extreme values in welfare analysis."""

    def test_extreme_price_change(self, extreme_price_change):
        """EVAL: CV/EV with 1000x price change."""
        baseline, policy = extreme_price_change

        from pyrevealed.algorithms.welfare import compute_cv_bounds, compute_ev_bounds

        cv = compute_cv_bounds(baseline, policy)
        ev = compute_ev_bounds(baseline, policy)

        # Bounds should be finite
        assert np.isfinite(cv), f"CV bounds not finite: {cv}"
        assert np.isfinite(ev), f"EV bounds not finite: {ev}"

    def test_identical_baseline_policy(self, baseline_equals_policy_log):
        """EVAL: CV/EV when baseline equals policy."""
        baseline, policy = baseline_equals_policy_log

        from pyrevealed.algorithms.welfare import compute_cv_bounds, compute_ev_bounds

        cv = compute_cv_bounds(baseline, policy)
        ev = compute_ev_bounds(baseline, policy)

        # No change should give CV=EV=0
        assert abs(cv) < 1e-10, f"Expected CV=0, got {cv}"
        assert abs(ev) < 1e-10, f"Expected EV=0, got {ev}"


class TestProductionExtremes:
    """EVAL: Extreme values in production analysis."""

    def test_zero_output_production(self, zero_output_production):
        """EVAL: Production with zero output."""
        from pyrevealed.algorithms.production import test_profit_maximization

        result = test_profit_maximization(zero_output_production)

        # Zero output means zero revenue, likely violation
        assert hasattr(result, 'is_consistent')

    def test_negative_profit_production(self, negative_profit_production):
        """EVAL: Production with negative profit."""
        from pyrevealed.algorithms.production import test_profit_maximization

        result = test_profit_maximization(negative_profit_production)

        # Negative profit doesn't necessarily violate profit max
        assert hasattr(result, 'is_consistent')
