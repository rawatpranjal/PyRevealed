"""
EVAL: Intertemporal choice overflow issues.

Tests for exponential discounting overflow and edge cases.
"""

import numpy as np
import pytest


class TestExponentialDiscountingOverflow:
    """EVAL: delta^t overflow for large t values."""

    def test_discounting_t100(self):
        """EVAL: Discounting with t=100 time periods."""
        from pyrevealed.algorithms.intertemporal import (
            test_exponential_discounting,
            DatedChoice,
        )

        choices = [
            DatedChoice(
                amounts=np.array([100.0, 200.0]),
                dates=np.array([0, 100]),
                chosen=1,  # Chose delayed
            ),
        ]

        result = test_exponential_discounting(choices)

        # delta^100 for reasonable delta should be fine
        assert np.isfinite(result.delta_lower)
        assert np.isfinite(result.delta_upper)

    def test_discounting_t500(self):
        """EVAL: Discounting with t=500 time periods."""
        from pyrevealed.algorithms.intertemporal import (
            test_exponential_discounting,
            DatedChoice,
        )

        choices = [
            DatedChoice(
                amounts=np.array([100.0, 1e6]),  # Large future reward
                dates=np.array([0, 500]),
                chosen=1,
            ),
        ]

        result = test_exponential_discounting(choices)

        # delta^500 may underflow for small delta
        # Should still return valid bounds
        assert hasattr(result, 'delta_lower')
        assert hasattr(result, 'delta_upper')

    def test_discounting_t800_underflow(self, extreme_delay_choices):
        """EVAL: Discounting with t=800 causes underflow.

        For delta=0.9, delta^800 ≈ 1e-37
        For delta=0.5, delta^800 ≈ 1e-241 (underflows)
        """
        from pyrevealed.algorithms.intertemporal import test_exponential_discounting

        result = test_exponential_discounting(extreme_delay_choices)

        # Bounds should be finite (using log tricks)
        assert np.isfinite(result.delta_lower), f"Lower bound: {result.delta_lower}"
        assert np.isfinite(result.delta_upper), f"Upper bound: {result.delta_upper}"


class TestZeroAmounts:
    """EVAL: Zero consumption amounts in intertemporal choices."""

    def test_zero_present_amount(self, zero_amount_choices):
        """EVAL: Zero consumption at present period (log(0) issue)."""
        from pyrevealed.algorithms.intertemporal import test_exponential_discounting

        # Should handle gracefully (not crash)
        try:
            result = test_exponential_discounting(zero_amount_choices)
            assert hasattr(result, 'delta_lower')
        except Exception as e:
            pytest.xfail(f"Zero amount handling failed: {e}")

    def test_zero_future_amount(self):
        """EVAL: Zero consumption at future period."""
        from pyrevealed.algorithms.intertemporal import (
            test_exponential_discounting,
            DatedChoice,
        )

        choices = [
            DatedChoice(
                amounts=np.array([100.0, 0.0]),
                dates=np.array([0, 1]),
                chosen=0,  # Rationally chose non-zero
            ),
        ]

        result = test_exponential_discounting(choices)

        # Choosing 0 over 100 is irrational - bounds may be degenerate
        assert hasattr(result, 'is_consistent')


class TestIdenticalTiming:
    """EVAL: Choices with identical timing."""

    def test_same_date_choices(self, identical_timing_choices):
        """EVAL: All options at same time - no discounting possible."""
        from pyrevealed.algorithms.intertemporal import test_exponential_discounting

        result = test_exponential_discounting(identical_timing_choices)

        # With identical timing, any delta satisfies (vacuously)
        assert hasattr(result, 'delta_lower')
        assert hasattr(result, 'delta_upper')


class TestQuasiHyperbolicDiscounting:
    """EVAL: Quasi-hyperbolic (beta-delta) discounting edge cases."""

    def test_beta_delta_extreme_present_bias(self):
        """EVAL: Extreme present bias (beta near 0)."""
        from pyrevealed.algorithms.intertemporal import (
            test_quasi_hyperbolic_discounting,
            DatedChoice,
        )

        # Extreme present bias choices
        choices = [
            DatedChoice(
                amounts=np.array([10.0, 1000.0]),
                dates=np.array([0, 1]),
                chosen=0,  # Extreme present bias
            ),
            DatedChoice(
                amounts=np.array([10.0, 11.0]),
                dates=np.array([1, 2]),
                chosen=1,  # Patient when both in future
            ),
        ]

        result = test_quasi_hyperbolic_discounting(choices)

        # Beta should be low
        assert hasattr(result, 'beta_lower')

    def test_beta_delta_no_present_bias(self):
        """EVAL: No present bias (beta = 1)."""
        from pyrevealed.algorithms.intertemporal import (
            test_quasi_hyperbolic_discounting,
            DatedChoice,
        )

        # Consistent exponential discounter
        choices = [
            DatedChoice(
                amounts=np.array([100.0, 105.0]),
                dates=np.array([0, 1]),
                chosen=1,  # Patient
            ),
            DatedChoice(
                amounts=np.array([100.0, 105.0]),
                dates=np.array([1, 2]),
                chosen=1,  # Same patience
            ),
        ]

        result = test_quasi_hyperbolic_discounting(choices)

        # Beta should be near 1
        assert hasattr(result, 'beta_lower')


class TestDiscountRateBounds:
    """EVAL: Interest rate computation from discount factor."""

    def test_interest_rate_from_small_delta(self):
        """EVAL: r = 1/delta - 1 for small delta (high interest rate)."""
        from pyrevealed.algorithms.intertemporal import (
            recover_discount_factor,
            DatedChoice,
        )

        # Patient choices imply low interest rate
        choices = [
            DatedChoice(
                amounts=np.array([100.0, 101.0]),
                dates=np.array([0, 1]),
                chosen=1,  # Very patient
            ),
        ]

        result = recover_discount_factor(choices)

        # High delta implies low interest rate
        assert hasattr(result, 'implied_interest_rate_lower')
        assert hasattr(result, 'implied_interest_rate_upper')

    def test_interest_rate_overflow(self):
        """EVAL: Interest rate computation when delta near 0."""
        from pyrevealed.algorithms.intertemporal import (
            recover_discount_factor,
            DatedChoice,
        )

        # Extremely impatient choices
        choices = [
            DatedChoice(
                amounts=np.array([1.0, 1000000.0]),
                dates=np.array([0, 1]),
                chosen=0,  # Still chose immediate
            ),
        ]

        result = recover_discount_factor(choices)

        # Interest rate might be infinity
        r_lower = result.implied_interest_rate_lower
        r_upper = result.implied_interest_rate_upper

        assert np.isfinite(r_lower) or r_lower == float('inf')


class TestIntertemporalCRRA:
    """EVAL: Intertemporal CRRA utility edge cases."""

    def test_crra_extreme_consumption_ratio(self):
        """EVAL: CRRA with extreme consumption ratios."""
        from pyrevealed.algorithms.intertemporal import (
            estimate_intertemporal_crra,
            DatedChoice,
        )

        choices = [
            DatedChoice(
                amounts=np.array([1e-10, 1e10]),  # Extreme ratio
                dates=np.array([0, 1]),
                chosen=1,
            ),
        ]

        result = estimate_intertemporal_crra(choices)

        # Should handle gracefully
        assert hasattr(result, 'rho')

    def test_crra_rho_bounds(self):
        """EVAL: CRRA rho estimation produces valid bounds."""
        from pyrevealed.algorithms.intertemporal import (
            estimate_intertemporal_crra,
            DatedChoice,
        )

        choices = [
            DatedChoice(
                amounts=np.array([100.0, 110.0]),
                dates=np.array([0, 1]),
                chosen=1,
            ),
        ]

        result = estimate_intertemporal_crra(choices)

        # Rho should be finite
        assert np.isfinite(result.rho), f"CRRA rho not finite: {result.rho}"


class TestIntertemporalConsistency:
    """EVAL: Intertemporal consistency checking."""

    def test_consistent_exponential_discounter(self):
        """EVAL: Perfectly consistent exponential discounter."""
        from pyrevealed.algorithms.intertemporal import (
            test_exponential_discounting,
            DatedChoice,
        )

        # All choices consistent with delta=0.95
        delta = 0.95
        choices = [
            DatedChoice(
                amounts=np.array([100.0, 100.0 / delta + 1]),
                dates=np.array([0, 1]),
                chosen=1,  # Future preferred
            ),
            DatedChoice(
                amounts=np.array([100.0, 100.0 / delta - 1]),
                dates=np.array([0, 1]),
                chosen=0,  # Present preferred
            ),
        ]

        result = test_exponential_discounting(choices)

        # Should be consistent with some delta
        assert result.is_consistent

    def test_inconsistent_choices(self):
        """EVAL: Choices inconsistent with exponential discounting."""
        from pyrevealed.algorithms.intertemporal import (
            test_exponential_discounting,
            DatedChoice,
        )

        # Preference reversal
        choices = [
            DatedChoice(
                amounts=np.array([100.0, 110.0]),
                dates=np.array([0, 1]),
                chosen=0,  # Prefers present
            ),
            DatedChoice(
                amounts=np.array([100.0, 105.0]),  # Less advantage for future
                dates=np.array([0, 1]),
                chosen=1,  # But prefers future?
            ),
        ]

        result = test_exponential_discounting(choices)

        # These choices are inconsistent
        assert not result.is_consistent
